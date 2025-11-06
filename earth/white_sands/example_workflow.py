#!/usr/bin/env python3
"""
Example: Complete White Sands Weather Data Pipeline

This example demonstrates the complete workflow for downloading and processing
ERA5 weather data for the White Sands test area. It shows all 4 steps of the
pipeline with clear explanations.

This is a reference example - in practice, you would run download_white_sands_data.py
or execute each step individually.

Author: ECMWF Weather Pipeline
Date: 2025
"""

def print_step(step_num, title):
    """Print a formatted step header."""
    print("\n" + "="*70)
    print(f"STEP {step_num}: {title}")
    print("="*70 + "\n")


def main():
    print("\n" + "="*70)
    print("White Sands Weather Data Pipeline - Example Workflow")
    print("="*70)
    
    print("""
This example demonstrates the complete 4-step pipeline for processing
White Sands weather data:

1. Fetch ERA5 Data from ECMWF
2. Calculate Air Density
3. Regrid to Cartesian Coordinates  
4. Compute Hydrostatic Pressure

The pipeline covers:
- Test area: 106.7°W - 106.2°W, 32.6°N - 33.6°N
- With buffer: 107.2°W - 105.7°W, 32.1°N - 34.1°N (0.5° buffer)
- Time: October 1-2, 2025
- Domain: ~223 km N-S × ~140 km E-W × 15 km vertical
""")
    
    # Step 1: Fetch ERA5 Data
    print_step(1, "Fetch ERA5 Data")
    print("Command:")
    print("  python download_white_sands_data.py")
    print()
    print("Or directly:")
    print("  python ../ecmwf/fetch_era5_pipeline.py white_sands.yaml")
    print()
    print("What it does:")
    print("  - Reads white_sands.yaml configuration")
    print("  - Calculates geographic bounds (32.1N-34.1N, 107.2W-105.7W)")
    print("  - Downloads ERA5 hourly data for Oct 1-2, 2025")
    print("  - Fetches both dynamics and density variables")
    print("  - Saves to directory named with coordinates")
    print()
    print("Output files:")
    print("  <output_dir>/era5_hourly_dynamics_20251001.nc")
    print("  <output_dir>/era5_hourly_dynamics_20251002.nc")
    print("  <output_dir>/era5_hourly_densities_20251001.nc")
    print("  <output_dir>/era5_hourly_densities_20251002.nc")
    print()
    print("Time: Typically 5-30 minutes depending on CDS load")
    
    # Step 2: Calculate Density
    print_step(2, "Calculate Air Density")
    print("Command:")
    print("  python ../ecmwf/calculate_density.py \\")
    print("    --input-dir <output_dir> \\")
    print("    --output-dir <output_dir>")
    print()
    print("Example:")
    print("  python ../ecmwf/calculate_density.py \\")
    print("    --input-dir ./32.10N_34.10N_107.20W_105.70W \\")
    print("    --output-dir ./32.10N_34.10N_107.20W_105.70W")
    print()
    print("What it does:")
    print("  - Loads dynamics (T, P) and densities (q, cloud content)")
    print("  - Solves ideal gas law with moisture and clouds")
    print("  - Computes ρ_total = ρ_dry + ρ_vapor + ρ_cloud")
    print("  - Saves density components to NetCDF")
    print()
    print("Output files:")
    print("  <output_dir>/era5_density_20251001.nc")
    print("  <output_dir>/era5_density_20251002.nc")
    print()
    print("Time: Typically 1-5 minutes")
    
    # Step 3: Regrid
    print_step(3, "Regrid to Cartesian Coordinates")
    print("Command:")
    print("  python ../ecmwf/regrid_era5_to_cartesian.py \\")
    print("    white_sands.yaml <output_dir> \\")
    print("    --output regridded_white_sands.nc")
    print()
    print("Example:")
    print("  python ../ecmwf/regrid_era5_to_cartesian.py \\")
    print("    white_sands.yaml ./32.10N_34.10N_107.20W_105.70W \\")
    print("    --output ./regridded_white_sands.nc")
    print()
    print("What it does:")
    print("  - Transforms pressure-level data to height-based grid")
    print("  - Interpolates to Cartesian mesh (150×400×300 cells)")
    print("  - Includes ghost zones for finite-volume methods")
    print("  - Creates cell-centered and interface coordinates")
    print()
    print("Output file:")
    print("  regridded_white_sands.nc (Cartesian grid)")
    print()
    print("Time: Typically 10-30 minutes")
    
    # Step 4: Compute Pressure
    print_step(4, "Compute Hydrostatic Pressure")
    print("Command:")
    print("  python ../ecmwf/compute_hydrostatic_pressure.py \\")
    print("    white_sands.yaml regridded_white_sands.nc")
    print()
    print("What it does:")
    print("  - Ensures hydrostatic balance in regridded data")
    print("  - Computes pressure from density using dP/dz = -ρg")
    print("  - Augments NetCDF with balanced pressure field")
    print("  - Ensures consistency for atmospheric simulations")
    print()
    print("Output file:")
    print("  regridded_white_sands.nc (updated with pressure)")
    print()
    print("Time: Typically 1-5 minutes")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70 + "\n")
    print("After completing all 4 steps, you will have:")
    print()
    print("1. Raw ERA5 data on pressure levels (NetCDF)")
    print("2. Computed density fields (NetCDF)")
    print("3. Regridded Cartesian grid data (NetCDF)")
    print("4. Hydrostatically balanced fields (NetCDF)")
    print()
    print("The final regridded_white_sands.nc file is ready for:")
    print("  - Atmospheric simulations")
    print("  - Boundary conditions for mesoscale models")
    print("  - Initial conditions for weather prediction")
    print("  - Analysis and visualization")
    print()
    print("Total time: ~20-60 minutes (depending on network and processing)")
    print()
    
    # Quick Start
    print("="*70)
    print("QUICK START")
    print("="*70 + "\n")
    print("To run the complete pipeline in one command:")
    print()
    print("  cd /path/to/planets/earth/white_sands")
    print("  python download_white_sands_data.py")
    print()
    print("Then manually run steps 2-4 with the generated output directory.")
    print()
    print("See README.md for detailed instructions and troubleshooting.")
    print()


if __name__ == "__main__":
    main()
