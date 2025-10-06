#! /usr/bin/env python3

from snapy import (
        MeshBlockOptions,
        MeshBlock,
        )
from kintera import ThermoX
from paddle import (
        setup_profile,
        write_profile,
        find_init_params,
        )

def setup_saturn_profile(fname: str, method: str = "moist-adiabat"):
    print(f"Reading input file: {fname}")

    op_block = MeshBlockOptions.from_yaml(str(fname))
    block = MeshBlock(op_block)

    # Initial guess for parameters
    param = {
        "Ts": 500.,
        "Ps": 100.e5,
        "Tmin": 85.,
        "xH2O": 8.91e-3,
        "xNH3": 3.52e-4,
        "xH2S": 8.08e-5,
        "grav": 10.44,
    }

    param = find_init_params(
            block,
            param,
            target_T=134.,
            target_P=1.e5,
            method=method,
            max_iter=50,
            ftol=1.e-2,
            verbose=True)

    w = setup_profile(block, param, method=method)

    return w, block

if __name__ == "__main__":
    # Fix 1bar temperature to 134K
    print("Setting up Saturn moist-adiabat profiles")
    w, block = setup_saturn_profile("saturn1d.yaml", "moist-adiabat")
    write_profile(f"saturn_profile_moist.txt", w, block)

    print("Setting up Saturn pseudo-adiabat profiles")
    w, block = setup_saturn_profile("saturn1d.yaml", "pseudo-adiabat")
    write_profile(f"saturn_profile_pseudo.txt", w, block)

    print("Setting up Saturn dry-adiabat profiles")
    w, block = setup_saturn_profile("saturn1d.yaml", "dry-adiabat")
    write_profile(f"saturn_profile_dry.txt", w, block)

    print("Setting up Saturn neutral density profiles")
    w, block = setup_saturn_profile("saturn1d.yaml", "neutral")
    write_profile(f"saturn_profile_neutral.txt", w, block)

    print("Saturn profiles setup complete.")
