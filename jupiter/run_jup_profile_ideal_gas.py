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

def setup_jupiter_profile(fname: str, method: str = "moist-adiabat"):
    print(f"Reading input file: {fname}")

    op_block = MeshBlockOptions.from_yaml(str(fname))
    block = MeshBlock(op_block)

    # Initial guess for parameters
    param = {
        "Ts": 600.,
        "Ps": 100.e5,
        "Tmin": 110.,
        "xH2O": 3.e-3,
        "xNH3": 3.2e-4,
        "xH2S": 1.e-5,
        "grav": 24.79,
    }

    param = find_init_params(
            block,
            param,
            target_T=166.,
            target_P=1.e5,
            method=method,
            max_iter=100,
            ftol=1.e-1,
            verbose=True)

    print("parameters found:", param)

    w = setup_profile(block, param, method=method)

    return w, block

if __name__ == "__main__":
    # Fix 1bar temperature to be 166 K
    print("Setting up Jupiter pseudo-adiabat profiles")
    w, block = setup_jupiter_profile("jupiter1d.yaml", "pseudo-adiabat")
    write_profile(f"jupiter_profile_pseudo_100bar.txt", w, block)

    print("Setting up Jupiter dry-adiabat profiles")
    w, block = setup_jupiter_profile("jupiter1d.yaml", "dry-adiabat")
    write_profile(f"jupiter_profile_dry_100bar.txt", w, block)

    print("Setting up Jupiter neutral density profiles")
    w, block = setup_jupiter_profile("jupiter1d.yaml", "neutral")
    write_profile(f"jupiter_profile_neutral_100bar.txt", w, block)

    print("Jupiter profiles setup complete.")
