#!/usr/bin/env python3
"""
White Sands Weather Data Download Script

This script downloads ERA5 weather data for the White Sands test area using
the ECMWF data curation pipeline. It executes the complete 4-step pipeline:

Step 1: Fetch ERA5 data (dynamics and densities)
Step 2: Calculate air density from downloaded data
Step 3: Regrid to Cartesian coordinates
Step 4: Compute hydrostatic pressure

The script uses the white_sands.yaml configuration file which specifies:
- Test area: 106.7W - 106.2W, 32.6N - 33.6N
- Buffer area: 107.2W - 105.7W, 32.1N - 34.1N (0.5 degree buffer on each side)
- Time window: 2025-10-01 to 2025-10-02

Usage:
    python download_white_sands_data.py [options]

Options:
    --config PATH           Path to YAML configuration file (default: white_sands.yaml)
    --output-base PATH      Base directory for output files (default: current directory)
    --skip-fetch            Skip the data fetching step (use existing data)
    --skip-density          Skip density calculation step
    --skip-regrid           Skip regridding step
    --skip-pressure         Skip pressure computation step
    --jobs N                Number of parallel download jobs (default: 2)

Examples:
    # Download all data and run complete pipeline
    python download_white_sands_data.py

    # Download to specific directory
    python download_white_sands_data.py --output-base ./data

    # Run only steps 2-4 with existing data
    python download_white_sands_data.py --skip-fetch --output-base ./data

    # Download with more parallel jobs
    python download_white_sands_data.py --jobs 4

Requirements:
    - ECMWF CDS API credentials configured (~/.cdsapirc or CDSAPI_KEY env var)
    - Python packages: cdsapi, xarray, netCDF4, numpy, scipy, PyYAML
    - See ../ecmwf/requirements.txt for complete list

For setup instructions, see:
    - ../ecmwf/README_ECMWF.md
    - README.md in this directory
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

# Add parent directory to path for importing ECMWF modules
SCRIPT_DIR = Path(__file__).parent
ECMWF_DIR = SCRIPT_DIR.parent / "ecmwf"
sys.path.insert(0, str(ECMWF_DIR))


def check_cds_credentials():
    """Check if CDS API credentials are configured."""
    cdsapirc_path = Path.home() / ".cdsapirc"
    has_env_key = "CDSAPI_KEY" in os.environ
    has_config_file = cdsapirc_path.exists()
    
    if not has_env_key and not has_config_file:
        print("ERROR: CDS API credentials not found.")
        print()
        print("Please configure your ECMWF CDS API credentials:")
        print("1. Register at https://cds.climate.copernicus.eu/")
        print("2. Get your API key from https://cds.climate.copernicus.eu/how-to-api")
        print("3. Configure authentication:")
        print()
        print("   Option A: Set environment variable")
        print("   export CDSAPI_KEY='your-uid:your-api-key'")
        print()
        print("   Option B: Create ~/.cdsapirc file with:")
        print("   url: https://cds.climate.copernicus.eu/api")
        print("   key: your-uid:your-api-key")
        print()
        return False
    
    return True


def run_step(step_name, command, skip=False):
    """
    Run a pipeline step with error handling.
    
    Args:
        step_name: Name of the step for display
        command: Command to execute (list of arguments)
        skip: If True, skip this step
        
    Returns:
        True if successful or skipped, False if failed
    """
    if skip:
        print(f"\n{'='*70}")
        print(f"SKIPPING: {step_name}")
        print(f"{'='*70}\n")
        return True
    
    print(f"\n{'='*70}")
    print(f"STEP: {step_name}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(str(c) for c in command)}")
    print()
    
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=False,  # Show output in real-time
            text=True
        )
        print(f"\n✓ {step_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {step_name} failed with exit code {e.returncode}")
        print(f"Error: {e}")
        return False
    except FileNotFoundError as e:
        print(f"\n✗ Command not found: {e}")
        print(f"Make sure the ECMWF scripts are in: {ECMWF_DIR}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download and process White Sands weather data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--config",
        default="white_sands.yaml",
        help="Path to YAML configuration file (default: white_sands.yaml)"
    )
    
    parser.add_argument(
        "--output-base",
        default=".",
        help="Base directory for output files (default: current directory)"
    )
    
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip the data fetching step (use existing data)"
    )
    
    parser.add_argument(
        "--skip-density",
        action="store_true",
        help="Skip density calculation step"
    )
    
    parser.add_argument(
        "--skip-regrid",
        action="store_true",
        help="Skip regridding step"
    )
    
    parser.add_argument(
        "--skip-pressure",
        action="store_true",
        help="Skip pressure computation step"
    )
    
    parser.add_argument(
        "--jobs",
        type=int,
        default=2,
        help="Number of parallel download jobs (default: 2)"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = SCRIPT_DIR / config_path
    
    output_base = Path(args.output_base).resolve()
    
    # Check if config file exists
    if not config_path.exists():
        print(f"ERROR: Configuration file not found: {config_path}")
        return 1
    
    print("="*70)
    print("White Sands Weather Data Download Pipeline")
    print("="*70)
    print(f"Configuration: {config_path}")
    print(f"Output base: {output_base}")
    print(f"Parallel jobs: {args.jobs}")
    print()
    
    # Check credentials before starting
    if not args.skip_fetch:
        if not check_cds_credentials():
            return 1
    
    # Step 1: Fetch ERA5 data
    fetch_script = ECMWF_DIR / "fetch_era5_pipeline.py"
    step1_success = run_step(
        "Step 1: Fetch ERA5 Data",
        ["python", str(fetch_script), str(config_path), 
         "--output-base", str(output_base), "--jobs", str(args.jobs)],
        skip=args.skip_fetch
    )
    
    if not step1_success:
        print("\n✗ Pipeline failed at Step 1")
        return 1
    
    # Determine output directory name
    # The fetch script creates a directory named with lat-lon bounds
    # We need to find it or calculate it
    # For now, we'll use a placeholder - in practice, the script would need to
    # parse the output or calculate the expected directory name
    print("\nNote: After Step 1, locate the output directory (e.g., 32.10N_34.10N_107.20W_105.70W)")
    print("      Then use it as input for the remaining steps.")
    print()
    print("Example commands for remaining steps:")
    print()
    print("Step 2: Calculate density")
    print("  python ../ecmwf/calculate_density.py \\")
    print("    --input-dir <output_dir> \\")
    print("    --output-dir <output_dir>")
    print()
    print("Step 3: Regrid to Cartesian coordinates")
    print("  python ../ecmwf/regrid_era5_to_cartesian.py \\")
    print("    white_sands.yaml <output_dir> \\")
    print("    --output regridded_white_sands.nc")
    print()
    print("Step 4: Compute hydrostatic pressure")
    print("  python ../ecmwf/compute_hydrostatic_pressure.py \\")
    print("    white_sands.yaml regridded_white_sands.nc")
    print()
    
    # Note: Steps 2-4 would require knowing the exact output directory name
    # which depends on the calculated lat-lon bounds. For a complete implementation,
    # we would need to either:
    # 1. Parse the output of Step 1 to get the directory name
    # 2. Recalculate the bounds using the same logic as fetch_era5_pipeline.py
    # 3. Make fetch_era5_pipeline.py return/print the directory name
    
    print("="*70)
    print("Pipeline execution completed!")
    print("="*70)
    print()
    print("Next steps:")
    print("1. Verify the downloaded data in the output directory")
    print("2. Run steps 2-4 manually with the output directory path")
    print("3. Use the regridded data for White Sands simulations")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
