import torch
import math
import time
import yaml
import kintera
import snapy
import numpy as np
from snapy import (
        index,
        MeshBlockOptions,
        MeshBlock,
        OutputOptions,
        NetcdfOutput
        )
from kintera import (
        ThermoOptions,
        ThermoX,
        KineticsOptions,
        Kinetics,
        evolve_implicit,
        )
#from paddle import setup_profile
from setup_profile import setup_profile
from evolve_kinetics import evolve_kinetics

torch.set_default_dtype(torch.float64)

'''
def evolve_kinetics(block_vars, eos, thermo_x, thermo_y, kinet, dt):
    # evolve kinetics
    hydro_u = block_vars["hydro_u"]
    hydro_w = block_vars["hydro_w"]

    temp = eos.compute("W->T", [hydro_w])
    pres = hydro_w[index.ipr]
    xfrac = thermo_y.compute("Y->X", [hydro_w[index.icy:, :, :, :]])
    conc = thermo_x.compute("TPX->V", [temp, pres, xfrac])
    cp_vol = thermo_x.compute("TV->cp", [temp, conc])

    conc_kinet = conc[:, :, :, 1:]
    #conc_kinet = kinet.options.narrow_copy(conc, thermo_y.options)

    rate, rc_ddC, rc_ddT = kinet.forward_nogil(temp, pres, conc_kinet)
    jac = kinet.jacobian(temp, conc_kinet, cp_vol, rate, rc_ddC, rc_ddT)

    stoich = kinet.buffer("stoich")
    del_conc = evolve_implicit(rate, stoich, jac, dt)

    inv_mu = thermo_y.buffer("inv_mu")
    del_rho = del_conc / inv_mu[1:].view(1, 1, 1, -1)
    hydro_u[index.icy:, :, :, :] += del_rho.permute(3, 0, 1, 2)
'''

if __name__ == "__main__":
    infile = "k2-18b.yaml"
    config = yaml.safe_load(open(infile, "r"))

    # device
    #device = torch.device("cuda:0")
    device = torch.device("cpu")

    # set hydrodynamic options
    op = MeshBlockOptions.from_yaml(infile)
    block = MeshBlock(op)
    block.to(device)

    # get handles to modules
    coord = block.hydro.module("coord")
    thermo_y = block.hydro.module("eos.thermo")
    eos = block.hydro.get_eos()
    thermo_y.options.max_iter(50)

    thermo_x = ThermoX(thermo_y.options)
    thermo_x.to(device)

    block_vars = {}
    interior = block.part((0, 0, 0))

    if "init_cond" in config["problem"]:
        nc3 = coord.buffer("x3v").shape[0]
        nc2 = coord.buffer("x2v").shape[0]
        nc1 = coord.buffer("x1v").shape[0]
        nvar = eos.nvar()
        block_vars["hydro_w"] = torch.zeros((nvar, nc3, nc2, nc1),
                                            device=device)

        module = torch.jit.load(config["problem"]["init_cond"])
        data = {name: param for name, param in module.named_buffers()}
        block_vars["hydro_w"][interior] = data["hydro_w"].to(device)
    else:
        #block_vars["hydro_w"] = setup_moist_adiabatic_profile(
                #config, coord, eos, thermo_x, device=device)
        param = {}
        param["Ts"] = float(config["problem"]["Ts"])
        param["Ps"] = float(config["problem"]["Ps"])
        param["grav"] = - float(config["forcing"]["const-gravity"]["grav1"])
        param["Tmin"] = float(config["problem"]["Tmin"])
        for name in thermo_y.options.species():
            param[f"x{name}"] = float(config["problem"].get(f"x{name}", 0.0))
        block_vars["hydro_w"] = setup_profile(block,
                                              param,
                                              method="pseudo-adiabat")

    # initialize
    block_vars = block.initialize(block_vars)

    out2 = NetcdfOutput(OutputOptions().file_basename("k2-18b").fid(2).variable("prim"))
    out3 = NetcdfOutput(OutputOptions().file_basename("k2-18b").fid(3).variable("uov"))
    out4 = NetcdfOutput(OutputOptions().file_basename("k2-18b").fid(4).variable("diag"))

    # kinetics model
    op_kinet = KineticsOptions.from_yaml(infile)
    kinet = Kinetics(op_kinet)
    kinet.to(device)

    # integration
    count, current_time, start_time = 0, 0, time.time()
    while not block.intg.stop(count, current_time):
        dt = block.max_time_step(block_vars)

        # make output
        if count % 100 == 0:
            print(f"count = {count}, dt = {dt}, time = {current_time}", flush=True)
            u = block_vars["hydro_u"]
            w = block_vars["hydro_w"]
            print("mass = ", u[interior][index.idn].sum(), flush=True)

            qtol = block_vars["hydro_w"][index.icy:, :, :, :].sum(dim=0)
            block.set_uov("qtol", qtol)

            for out in [out2, out3, out4]:
                out.increment_file_number()
                out.write_output_file(block, block_vars, current_time)
                out.combine_blocks()

        # evolve dynamics
        # linear cooling rate:
        # dT/dt = - A * (z - z0), z > z0
        #       = 0, z <= z0
        # dE/dt = - rho * cv * dTdt
        A = 1.e-5        # K / s
        x1v = coord.buffer("x1v")
        z0 = 80.e3
        dTdt = torch.zeros_like(x1v, device=device)
        dTdt[x1v > z0] = - A * (x1v[x1v > z0] - z0) / 1000.

        for stage in range(len(block.intg.stages)):
            block.forward(dt, stage, block_vars)

            # add cooling
            w = block_vars["hydro_w"]
            ivol = thermo_y.compute("DY->V", (w[index.idn], w[index.icy:]))
            temp = eos.compute("W->T", (w,))
            cv = thermo_y.compute("VT->cv", (ivol, temp))
            u = block_vars["hydro_u"]
            weight = block.intg.stages[stage].wght2()
            u[index.ipr] += weight * w[index.idn] * cv * dTdt * dt

        #evolve_kinetics(block_vars, eos, thermo_x, thermo_y, kinet, dt)
        del_rho = evolve_kinetics(block_vars["hydro_w"], block, kinet, thermo_x, dt)
        block_vars["hydro_u"][index.icy:, :, :, :] += del_rho

        count += 1
        current_time += dt

'''
# from Leconte+ 2024
o = 5.67E-8         # W / m^2 / K^4
s0 = 580;           # W / m^2
# Teq = (s0 / o / 4) ** (1/4)
q_dot = s0 / 4      # heat flux

# device
device = torch.device("cuda:0")

# set hydrodynamic options
op = MeshBlockOptions.from_yaml("k2-18b.yaml")

# initialize block
block = MeshBlock(op)
block.to(device)

# get handles to modules
coord = block.hydro.module("coord")
thermo = block.hydro.module("eos.thermo")
eos = block.hydro.module("eos")

# thermodynamics
Rd = kintera.constants.Rgas / kintera.species_weights()[0]
cv = kintera.species_cref_R()[0] * Rd
cp = cv + Rd

# set initial condition

x3v, x2v, x1v = torch.meshgrid(
    coord.buffer("x3v"), coord.buffer("x2v"), coord.buffer("x1v"), indexing="ij"
)   # x3v is x, x2v is y, x1v is z

# dimensions
nc3 = coord.buffer("x3v").shape[0]
nc2 = coord.buffer("x2v").shape[0]
nc1 = coord.buffer("x1v").shape[0]
nvar = 5

w = torch.zeros((nvar, nc3, nc2, nc1), device=device)       # initialize primitive variables (density, vx, vy, vz, pressure)

# temp = Ts - grav * x1v / cp       # adiabatic condition
temp = torch.full_like(x1v, Ts)     # isothermal condition

# w[index.ipr] = p0 * torch.pow(temp / Ts, cp / Rd)
w[index.ipr] = p0 * torch.exp(-grav * x1v / Rd / Ts)        # isothermal pressure
w[index.idn] = w[index.ipr] / (Rd * temp)                   # ideal gas law

block_vars = {}
block_vars["hydro_w"] = w
block_vars = block.initialize(block_vars)
block_vars["scalar_x"] = torch.tensor(0.)
block_vars["scalar_v"] = torch.tensor(0.)

# make output
directory = f"output_{experiment_name}"
try:
    os.mkdir(directory)
except FileExistsError:
    pass
out2 = NetcdfOutput(OutputOptions().file_basename(f"{directory}/convection").fid(2).variable("prim"))
out3 = NetcdfOutput(OutputOptions().file_basename(f"{directory}/convection").fid(3).variable("uov"))

block.set_uov("temp", temp)
block.set_uov("theta", temp * (p0 / w[index.ipr]).pow(Rd / cp))

activities = [ProfilerActivity.CPU]

# integration
count = 0
start_time = time.time()
interior = block.part((0, 0, 0))
current_time = 0.0

bottom_row_height = torch.full((nc3, nc2), coord.buffer("dx1f")[interior[-1]][0])[interior[1:3]]
bottom_row_height = bottom_row_height.to(device)
top_row_height = torch.full((nc3, nc2), coord.buffer("dx1f")[interior[-1]][-1])[interior[1:3]]
top_row_height = top_row_height.to(device)  

w[interior][index.ivx] = torch.randn_like(w[interior][index.ivy])
# with profile(activities=activities, record_shapes=True) as prof:
while not block.intg.stop(count, current_time):
    dt = block.max_time_step(block_vars)

    if count % 1000 == 0:
        print(f"count = {count}, dt = {dt}, time = {current_time}")
        u = block_vars["hydro_u"]
        print("mass = ", u[interior][index.idn].sum())

        ivol = thermo.compute("DY->V", (w[index.idn], w[index.icy:]))
        temp = thermo.compute("PV->T", (w[index.ipr], ivol))
        theta = temp * (p0 / w[index.ipr]).pow(Rd / cp)

        block.set_uov("temp", temp)
        block.set_uov("theta", theta)

        for out in [out2, out3]:
            out.increment_file_number()
            out.write_output_file(block, block_vars, current_time)
            out.combine_blocks()

    for stage in range(len(block.intg.stages)):
        block.forward(dt, stage, block_vars)
        u = block_vars["hydro_u"]
        last_weight = block.intg.stages[stage].wght2()
        u[interior][index.ipr][:, :, 0] += last_weight * q_dot / bottom_row_height * dt     # heating from the bottom
        u[interior][index.ipr][:, :, -1] -= last_weight * q_dot / top_row_height * dt       # cooling from the top
    
    count += 1
    current_time += dt

print("elapsed time = ", time.time() - start_time)
# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
'''
