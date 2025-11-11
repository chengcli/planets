import math
import time
import numpy as np
import yaml
import torch
import snapy
import kintera
from pathlib import Path
from snapy import (
        index,
        MeshBlockOptions,
        MeshBlock,
        OutputOptions,
        NetcdfOutput
        )
from kintera import (
        ThermoX,
        KineticsOptions,
        Kinetics,
        evolve_implicit
        )

torch.set_default_dtype(torch.float64)

RESTART_FOLDER = Path("31.90N_34.30N_107.35W_105.55W") / "regridded_white-sands_20251002"
RESTART_FILE = "regridded_white-sands_20251002_block_0_0.restart"

def load_restart_block(block_vars):
    restart = torch.jit.load(RESTART_FOLDER / RESTART_FILE)
    print('shape = ', restart.hydro_w.shape)

    block_vars["hydro_w"] = restart.hydro_w[0,...]

if __name__ == "__main__":
    block_vars = {}
    load_restart_block(block_vars)
    print("hydro_w shape = ", block_vars["hydro_w"].shape)
    exit()

    #infile = "jupiter3d.yaml"
    #config = yaml.safe_load(open(infile, "r"))

    # device
    device = torch.device("cuda:0")
    #device = torch.device("cpu")

    # set hydrodynamic options
    op = MeshBlockOptions.from_yaml(infile)
    block = MeshBlock(op)
    block.to(device)

    # get handles to modules
    coord = block.hydro.module("coord")
    thermo_y = block.hydro.module("eos.thermo")
    eos = block.hydro.get_eos()
    #thermo_y.options.max_iter(100)

    thermo_x = ThermoX(thermo_y.options)
    thermo_x.to(device)

    block_vars = {}
    interior = block.part((0, 0, 0))

    # initialize
    block_vars = block.initialize(block_vars)

    # kinetics model
    op_kinet = KineticsOptions.from_yaml(infile)
    kinet = Kinetics(op_kinet)
    kinet.to(device)

    # integration
    count, current_time, start_time = 0, 0, time.time()
    while not block.intg.stop(count, current_time):
        dt = block.max_time_step(block_vars)

        # make output
        if count % 1000 == 0:
            print(f"count = {count}, dt = {dt}, time = {current_time}", flush=True)
            u = block_vars["hydro_u"]
            print("mass = ", u[interior][index.idn].sum(), flush=True)

            qtol = block_vars["hydro_w"][index.icy:, :, :, :].sum(dim=0)
            block.set_uov("qtol", qtol)

            for out in [out2, out3, out4]:
                out.increment_file_number()
                out.write_output_file(block, block_vars, current_time)
                out.combine_blocks()

        # evolve dynamics
        for stage in range(len(block.intg.stages)):
            block.forward(dt, stage, block_vars)

        evolve_kinetics(block_vars, eos, thermo_x, thermo_y, kinet, dt)

        count += 1
        current_time += dt
