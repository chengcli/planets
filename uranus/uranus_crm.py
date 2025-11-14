import torch
import math
import time
import numpy as np
import yaml
import torch
import snapy
import kintera
from snapy import MeshBlockOptions, MeshBlock
from kintera import ThermoX, KineticsOptions, Kinetics
from paddle import evolve_kinetics, setup_profile

torch.set_default_dtype(torch.float64)

if __name__ == "__main__":
    infile = "uranus.yaml"
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
    thermo_y.options.max_iter(40)

    # construct thermo_x
    thermo_x = ThermoX(thermo_y.options)
    thermo_x.to(device)

    # initialize
    param = {
        "Ts": float(config["problem"]["Ts"]),
        "Ps": float(config["problem"]["Ps"]),
        "Tmin": float(config["problem"]["Tmin"]),
        "xH2S": float(config["problem"]["xH2S"]),
        "xCH4": float(config["problem"]["xCH4"]),
        "grav": - float(config["forcing"]["const-gravity"]["grav1"]),
    }

    block_vars = {}
    block.make_outputs(block_vars, current_time)

    block_vars["hydro_w"] = setup_profile(block, param, "pseudo-adiabat", verbose=True)
    block_vars = block.initialize(block_vars)

    # kinetics model
    op_kinet = KineticsOptions.from_yaml(infile)
    kinet = Kinetics(op_kinet)
    kinet.to(device)

    # integration
    start_time = time.time()
    current_time = 0.0
    #block.make_outputs(block_vars, current_time)

    while not block.intg.stop(block.inc_cycle(), current_time):
        dt = block.max_time_step(block_vars)
        block.print_cycle_info(current_time, dt)

        # evolve dynamics
        for stage in range(len(block.intg.stages)):
            block.forward(dt, stage, block_vars)
        
        # evolve kinetics
        du = evolve_kinetics(block_vars["hydro_w"], block, kinet, thermo_x, dt)
        block_vars["hydro_u"] += du

        current_time += dt
        #block.make_outputs(block_vars, current_time)

    print("elapsed time = ", time.time() - start_time)
