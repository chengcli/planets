import torch
import math
import time
import kintera
import numpy as np
from snapy import (
        index,
        MeshBlockOptions,
        MeshBlock,
        OutputOptions,
        NetcdfOutput,
        )
from kintera import (
        ThermoOptions,
        ThermoX,
        KineticsOptions,
        Kinetics,
        )

torch.set_default_dtype(torch.float64)

def evolve_kinetics(block, kinet, thermo_x):
    eos = block.module("hydro.eos")
    thermo_y = eos.named_modules()["thermo"]

    w = block.buffer("hydro.eos.W")

    temp = eos.compute("W->T", (w,))
    pres = w[index.ipr]
    xfrac = thermo_y.compute("Y->X", (w[ICY:],))
    conc = thermo_x.compute("TPX->V", (temp, pres, xfrac))
    cp_vol = thermo_x.compute("TV->cp", (temp, conc))

    conc_kinet = kinet.options.narrow_copy(conc, thermo_y.options)
    rate, rc_ddC, rc_ddT = kinet.forward(temp, pres, conc_kinet)
    jac = kinet.jacobian(temp, conc_kinet, cp_vol, rate, rc_ddC, rc_ddT)

    stoich = kinet.buffer("stoich")
    del_conc = kintera.evolve_implicit(rate, stoich, jac, dt)

    inv_mu = thermo_y.buffer("inv_mu")
    del_rho = del_conc / inv_mu[1:].view((1, 1, 1, -1))
    return del_rho.permute((3, 0, 1, 2))

if __name__ == '__main__':
    # input file
    infile = "earth.yaml"
    device = "cpu"

    # create meshblock
    op_block = MeshBlockOptions.from_yaml(infile)
    block = MeshBlock(op_block)
    block.to(torch.device(device))

    # create thermo module
    op_thermo = ThermoOptions.from_yaml(infile)
    thermo_x = ThermoX(op_thermo)
    thermo_x.to(torch.device(device))

    # create kinetics module
    op_kinet = KineticsOptions.from_yaml(infile)
    kinet = Kinetics(op_kinet)
    kinet.to(torch.device(device))

    # create output fields
    op_out = OutputOptions().file_basename("earth")
    out2 = NetcdfOutput(op_out.fid(2).variable("prim"))
    out3 = NetcdfOutput(op_out.fid(3).variable("uov"))
    out4 = NetcdfOutput(op_out.fid(4).variable("diag"))
    outs = [out2, out4]

    # set up initial condition
    w = setup_initial_condition(block, thermo_x)
    print("w = ", w[:,0,0,:])

    # integration
    current_time = 0.0
    count = 0
    start_time = time.time()
    interior = block.part((0, 0, 0))
    while not block.intg.stop(count, current_time):
        dt = block.max_time_step()
        u = block.buffer("hydro.eos.U")

        if count % 1 == 0:
            print(f"count = {count}, dt = {dt}, time = {current_time}")
            print("mass = ", u[interior][index.idn].sum())

            for out in outs:
                out.increment_file_number()
                out.write_output_file(block, current_time)
                out.combine_blocks()

        for stage in range(len(block.intg.stages)):
            block.forward(dt, stage)

        # evolve kinetics
        u[index.icy:] += evolve_kinetics(block, kinet, thermo_x)

        current_time += dt
        count += 1
