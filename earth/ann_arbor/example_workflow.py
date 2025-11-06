#!/usr/bin/env python3
"""
Example: Complete Ann Arbor Weather Data Pipeline

This example demonstrates the complete workflow for downloading and processing
ERA5 weather data for Ann Arbor, Michigan. It shows all 4 steps of the
pipeline with clear explanations.

This is a reference example - in practice, you would run download_ann_arbor_data.py
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
    print("Ann Arbor Weather Data Pipeline - Example Workflow")
    print("="*70)
    
    print("""
This example demonstrates the complete 4-step pipeline for processing
Ann Arbor weather data:

1. Fetch ERA5 Data from ECMWF
2. Calculate Air Density
3. Regrid to Cartesian Coordinates  
4. Compute Hydrostatic Pressure

The pipeline covers:
- Test area: Ann Arbor, Michigan (42.3°N, 83.7°W)
- Domain: 125 km × 125 km horizontal, 15 km vertical
- Time: November 1, 2025
- Grid: 150 × 200 × 200 interior cells + 3 ghost cells
""")
    
    # Step 1: Fetch ERA5 Data
    print_step(1, "Fetch ERA5 Data")
    print("Command:")
    print("  python download_ann_arbor_data.py")
    print()
    print("Or directly:")
    print("  python ../ecmwf/fetch_era5_pipeline.py ann_arbor.yaml")
    print()
    print("What it does:")
    print("  - Reads ann_arbor.yaml configuration")
    print("  - Calculates geographic bounds (~41.75N-42.85N, 84.25W-83.15W)")
    print("  - Downloads ERA5 hourly data for Nov 1, 2025")
    print("  - Fetches both dynamics and density variables")
    print("  - Saves to directory named with coordinates")
    print()
    print("Output files:")
    print("  <output_dir>/era5_hourly_dynamics_20251101.nc")
    print("  <output_dir>/era5_hourly_densities_20251101.nc")
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
    print("    --input-dir ./41.75N_42.85N_84.25W_83.15W \\")
    print("    --output-dir ./41.75N_42.85N_84.25W_83.15W")
    print()
    print("What it does:")
    print("  - Loads dynamics (T, P) and densities (q, cloud content)")
    print("  - Solves ideal gas law with moisture and clouds")
    print("  - Computes ρ_total = ρ_dry + ρ_vapor + ρ_cloud")
    print("  - Saves density components to NetCDF")
    print()
    print("Output files:")
    print("  <output_dir>/era5_density_20251101.nc")
    print()
    print("Time: Typically 1-5 minutes")
    
    # Step 3: Regrid
    print_step(3, "Regrid to Cartesian Coordinates")
    print("Command:")
    print("  python ../ecmwf/regrid_era5_to_cartesian.py \\")
    print("    ann_arbor.yaml <output_dir> \\")
    print("    --output regridded_ann_arbor.nc")
    print()
    print("Example:")
    print("  python ../ecmwf/regrid_era5_to_cartesian.py \\")
    print("    ann_arbor.yaml ./41.75N_42.85N_84.25W_83.15W \\")
    print("    --output ./regridded_ann_arbor.nc")
    print()
    print("What it does:")
    print("  - Transforms pressure-level data to height-based grid")
    print("  - Interpolates to Cartesian mesh (150×200×200 cells)")
    print("  - Includes ghost zones for finite-volume methods")
    print("  - Creates cell-centered and interface coordinates")
    print()
    print("Output file:")
    print("  regridded_ann_arbor.nc (Cartesian grid)")
    print()
    print("Time: Typically 10-30 minutes")
    
    # Step 4: Compute Pressure
    print_step(4, "Compute Hydrostatic Pressure")
    print("Command:")
    print("  python ../ecmwf/compute_hydrostatic_pressure.py \\")
    print("    ann_arbor.yaml regridded_ann_arbor.nc")
    print()
    print("What it does:")
    print("  - Ensures hydrostatic balance in regridded data")
    print("  - Computes pressure from density using dP/dz = -ρg")
    print("  - Augments NetCDF with balanced pressure field")
    print("  - Ensures consistency for atmospheric simulations")
    print()
    print("Output file:")
    print("  regridded_ann_arbor.nc (updated with pressure)")
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
    print("The final regridded_ann_arbor.nc file is ready for:")
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
    print("  cd /path/to/planets/earth/ann_arbor")
    print("  python download_ann_arbor_data.py")
    print()
    print("This will execute all 4 steps automatically.")
    print()
    print("See README.md for detailed instructions and troubleshooting.")
    print()
    
    # Ann Arbor Specific Notes
    print("="*70)
    print("ANN ARBOR WEATHER NOTES")
    print("="*70 + "\n")
    print("November weather in Ann Arbor typically features:")
    print("  - Cooling temperatures (2-10°C typical)")
    print("  - Increased cloudiness and precipitation")
    print("  - Lake-effect influences from Great Lakes")
    print("  - First snow events possible")
    print("  - Frequent frontal passages")
    print()
    print("This test case is ideal for studying:")
    print("  - Lake-land interactions")
    print("  - Fall convective systems")
    print("  - Synoptic-scale weather patterns")
    print("  - Urban-rural boundary layer effects")
    print()


if __name__ == "__main__":
    main()
