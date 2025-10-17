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

torch.set_default_dtype(torch.float64)

def setup_moist_adiabatic_profile(config, coord, eos, thermo_x,
                                  device=torch.device("cpu")):
    Tmin = float(config["problem"]["Tmin"])
    grav = - float(config["forcing"]["const-gravity"]["grav1"])
    Ps = float(config["problem"]["Ps"])
    Ts = float(config["problem"]["Ts"])

    # dimensions
    nc3 = coord.buffer("x3v").shape[0]
    nc2 = coord.buffer("x2v").shape[0]
    nc1 = coord.buffer("x1v").shape[0]
    ny = len(thermo_x.options.species()) - 1
    nvar = eos.nvar()

    temp = Ts * torch.ones((nc3, nc2), device=device)
    pres = Ps * torch.ones((nc3, nc2), device=device)
    xfrac = torch.zeros((nc3, nc2, 1 + ny), device=device)

    # read in compositions
    for i in range(1, 1 + ny):
        name = 'x' + thermo_x.options.species()[i]
        if name in config["problem"]:
            xfrac[:, :, i] = float(config["problem"][name])

    # dry air mole fraction
    xfrac[:, :, 0] = 1.0 - xfrac[:, :, 1:].sum(dim=2)

    # adiabatic extrapolate half a grid
    ifirst = coord.ifirst()
    dx1f = coord.buffer("dx1f")
    dz = dx1f[ifirst].item()
    thermo_x.extrapolate_ad(temp, pres, xfrac, grav, dz / 2.0)

    nvapor = len(thermo_x.options.vapor_ids())
    ncloud = len(thermo_x.options.cloud_ids())
    i = ifirst
    w = torch.zeros((nvar, nc3, nc2, nc1), device=device)
    while i < coord.ilast():
        print("i = ", i)
        conc = thermo_x.compute("TPX->V", [temp, pres, xfrac])

        w[index.ipr, :, :, i] = pres
        w[index.idn, :, :, i] = thermo_x.compute("V->D", [conc])
        w[index.icy:, :, :, i] = thermo_x.compute("X->Y", [xfrac])

        # drop clouds
        #w[int(index.icy) + nvapor:,:,:,i] = 0.

        if (temp < Tmin).any().item():
            break
            raise ValueError("Temperature below minimum")
        dz = dx1f[i].item()
        thermo_x.extrapolate_ad(temp, pres, xfrac, grav, dz)
        i += 1

    # isothermal extrapolation
    while i < coord.ilast():
        mu = (thermo_x.buffer("mu") * xfrac).sum(-1);
        dz = dx1f[i].item()
        pres *= torch.exp(-grav * mu * dz / (kintera.constants.Rgas * temp))
        conc = thermo_x.compute("TPX->V", [temp, pres, xfrac])
        w[index.ipr, :, :, i] = pres
        w[index.idn, :, :, i] = thermo_x.compute("V->D", [conc])
        w[index.icy:, :, :, i] = thermo_x.compute("X->Y", [xfrac])
        i += 1

    # add noise
    w[index.ivx] += 0.01 * torch.rand_like(w[index.ivx])
    w[index.ivy] += 0.01 * torch.rand_like(w[index.ivy])

    return w;

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
    thermo_y.options.max_iter(100)

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
        block_vars["hydro_w"] = setup_moist_adiabatic_profile(
                config, coord, eos, thermo_x, device=device)

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
        if count % 1000 == 0:
            print(f"count = {count}, dt = {dt}, time = {current_time}", flush=True)
            u = block_vars["hydro_u"]
            w = block_vars["hydro_w"]
            print("mass = ", u[interior][index.idn].sum(), flush=True)

            qtol = block_vars["hydro_w"][index.icy:, :, :, :].sum(dim=0)
            ivol = thermo_y.compute("DY->V", (w[index.idn], w[index.icy:]))
            temp = thermo_y.compute("PV->T", (w[index.ipr], ivol))
            #theta = temp * (p0 / w[index.ipr]).pow(Rd / cp)

            block.set_uov("qtol", qtol)
            block.set_uov("temp", temp)

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
