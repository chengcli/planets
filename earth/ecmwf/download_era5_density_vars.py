#!/usr/bin/env python3
"""
Convenient script to download density-related variables from ERA5.

This script downloads specific humidity and cloud/precipitation water content
variables at all standard ERA5 pressure levels for a specified region and time period.

The download is split into separate jobs - one per pressure level - which can
run in parallel for faster data retrieval. Once all jobs complete, the data
is combined into a single NetCDF file.

Variables downloaded:
    1. Specific cloud ice water content
    2. Specific humidity
    3. Specific snow water content
    4. Specific cloud liquid water content
    5. Specific rain water content

Usage:
    python download_era5_density_vars.py --latmin 30.0 --latmax 40.0 \
                                          --lonmin -110.0 --lonmax -100.0 \
                                          --start-date 2024-01-01 \
                                          --end-date 2024-01-02 \
                                          --output era5_density_vars.nc

Requirements:
    - ECMWF CDS API key configured (see README_ECMWF.md)
    - Required packages: cdsapi, xarray, netCDF4, numpy
"""

import argparse
import sys
import os
import tempfile
import shutil
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add current directory to path for importing local module
sys.path.insert(0, os.path.dirname(__file__))

from ecmwf_weather_api import ECMWFWeatherAPI

try:
    import xarray as xr
    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Download ERA5 density-related variables at all pressure levels.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Variables downloaded:
  1. Specific cloud ice water content
  2. Specific humidity
  3. Specific snow water content
  4. Specific cloud liquid water content
  5. Specific rain water content

Examples:
  # Download data for White Sands, New Mexico
  python download_era5_density_vars.py --latmin 32.0 --latmax 33.5 \\
                                        --lonmin -106.8 --lonmax -105.8 \\
                                        --start-date 2024-01-01 \\
                                        --end-date 2024-01-02 \\
                                        --output white_sands_density.nc

  # Download data for a larger region with specific times
  python download_era5_density_vars.py --latmin 30.0 --latmax 40.0 \\
                                        --lonmin -120.0 --lonmax -100.0 \\
                                        --start-date 2024-01-01 \\
                                        --end-date 2024-01-03 \\
                                        --times 00:00 12:00 \\
                                        --output region_density.nc

Note: Download may take several minutes depending on request size and CDS server load.
        """
    )
    
    # Required arguments
    parser.add_argument('--latmin', type=float, required=True,
                        help='Minimum latitude (-90 to 90)')
    parser.add_argument('--latmax', type=float, required=True,
                        help='Maximum latitude (-90 to 90)')
    parser.add_argument('--lonmin', type=float, required=True,
                        help='Minimum longitude (-180 to 180)')
    parser.add_argument('--lonmax', type=float, required=True,
                        help='Maximum longitude (-180 to 180)')
    parser.add_argument('--start-date', type=str, required=True,
                        help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end-date', type=str, required=True,
                        help='End date in YYYY-MM-DD format')
    parser.add_argument('--output', type=str, required=True,
                        help='Output NetCDF file path')
    
    # Optional arguments
    parser.add_argument('--times', nargs='+', default=None,
                        help='Specific times to download (e.g., 00:00 06:00 12:00 18:00). '
                             'Default: all available times (every 6 hours)')
    parser.add_argument('--jobs', type=int, default=4,
                        help='Number of parallel download jobs (default: 4). '
                             'Each job downloads one pressure level.')
    parser.add_argument('--api-key', type=str, default=None,
                        help='CDS API key (if not using environment variable or ~/.cdsapirc)')
    parser.add_argument('--api-url', type=str, default=None,
                        help='CDS API URL (default: https://cds.climate.copernicus.eu/api)')
    
    return parser.parse_args()


def validate_date_format(date_str):
    """Validate date format."""
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False


def download_single_level(api, level, variables, latmin, latmax, lonmin, lonmax,
                          start_date, end_date, times, temp_dir, level_idx, total_levels):
    """
    Download data for a single pressure level.
    
    Args:
        api: ECMWFWeatherAPI instance
        level: Pressure level in hPa
        variables: List of variable names
        latmin, latmax, lonmin, lonmax: Geographic bounds
        start_date, end_date: Date range
        times: List of times
        temp_dir: Temporary directory for storing individual files
        level_idx: Index of this level (for progress tracking)
        total_levels: Total number of levels
    
    Returns:
        Tuple of (level, output_file_path) or (level, None) on error
    """
    try:
        output_file = os.path.join(temp_dir, f'level_{level:04d}hPa.nc')
        
        print(f"[{level_idx+1}/{total_levels}] Downloading {level} hPa...")
        
        api.fetch_weather_data(
            latmin=latmin,
            latmax=latmax,
            lonmin=lonmin,
            lonmax=lonmax,
            start_date=start_date,
            end_date=end_date,
            variables=variables,
            pressure_levels=[level],
            output_file=output_file,
            times=times
        )
        
        print(f"[{level_idx+1}/{total_levels}] ✓ Completed {level} hPa")
        return (level, output_file)
    
    except Exception as e:
        print(f"[{level_idx+1}/{total_levels}] ✗ Failed {level} hPa: {e}")
        return (level, None)


def combine_netcdf_files(file_list, output_file):
    """
    Combine multiple NetCDF files (one per pressure level) into a single file.
    
    Args:
        file_list: List of tuples (level, filepath)
        output_file: Path for the combined output file
    
    Returns:
        True if successful, False otherwise
    """
    if not XARRAY_AVAILABLE:
        print("Error: xarray is required for combining files.")
        print("Install with: pip install xarray netCDF4")
        return False
    
    try:
        print(f"\nCombining {len(file_list)} pressure level files...")
        
        # Filter out failed downloads
        valid_files = [(level, path) for level, path in file_list if path is not None]
        
        if not valid_files:
            print("Error: No files to combine (all downloads failed)")
            return False
        
        if len(valid_files) < len(file_list):
            failed = len(file_list) - len(valid_files)
            print(f"Warning: {failed} pressure level(s) failed to download")
        
        # Load all datasets
        datasets = []
        for level, filepath in valid_files:
            ds = xr.open_dataset(filepath)
            datasets.append(ds)
        
        # Combine along the level dimension
        combined_ds = xr.concat(datasets, dim='level')
        
        # Sort by pressure level (descending - from low pressure/high altitude to high pressure/low altitude)
        combined_ds = combined_ds.sortby('level', ascending=False)
        
        # Save to output file
        combined_ds.to_netcdf(output_file)
        combined_ds.close()
        
        # Close individual datasets
        for ds in datasets:
            ds.close()
        
        print(f"✓ Successfully combined into: {output_file}")
        return True
    
    except Exception as e:
        print(f"Error combining files: {e}")
        return False


def main():
    """Main function to download ERA5 density-related variables."""
    args = parse_arguments()
    
    # Validate date formats
    if not validate_date_format(args.start_date):
        print(f"Error: Invalid start date format '{args.start_date}'. Use YYYY-MM-DD.")
        sys.exit(1)
    
    if not validate_date_format(args.end_date):
        print(f"Error: Invalid end date format '{args.end_date}'. Use YYYY-MM-DD.")
        sys.exit(1)
    
    print("=" * 70)
    print("ERA5 Density-Related Variables Download")
    print("=" * 70)
    print(f"\nRegion:")
    print(f"  Latitude:  [{args.latmin}, {args.latmax}]")
    print(f"  Longitude: [{args.lonmin}, {args.lonmax}]")
    print(f"\nTime Period:")
    print(f"  Start: {args.start_date}")
    print(f"  End:   {args.end_date}")
    print(f"\nVariables:")
    print(f"  1. Specific cloud ice water content")
    print(f"  2. Specific humidity")
    print(f"  3. Specific snow water content")
    print(f"  4. Specific cloud liquid water content")
    print(f"  5. Specific rain water content")
    print(f"\nPressure Levels: All 37 standard ERA5 levels")
    print(f"  (1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200,")
    print(f"   225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750,")
    print(f"   775, 800, 825, 850, 875, 900, 925, 950, 975, 1000 hPa)")
    
    if args.times:
        print(f"\nTimes: {', '.join(args.times)}")
    else:
        print(f"\nTimes: All available (00:00, 06:00, 12:00, 18:00 UTC)")
    
    print(f"\nOutput file: {args.output}")
    print("\n" + "=" * 70)
    print("\nInitializing ECMWF API...")
    
    try:
        # Initialize the API
        api = ECMWFWeatherAPI(api_key=args.api_key, api_url=args.api_url)
        
        print("✓ API initialized successfully")
        
        # Define density-related variables to download
        # These are the official ECMWF ERA5 variable names
        variables = [
            'specific_cloud_ice_water_content',
            'specific_humidity',
            'specific_snow_water_content',
            'specific_cloud_liquid_water_content',
            'specific_rain_water_content'
        ]
        
        # All standard ERA5 pressure levels
        pressure_levels = [
            1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200,
            225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750,
            775, 800, 825, 850, 875, 900, 925, 950, 975, 1000
        ]
        
        print(f"\nDownloading {len(pressure_levels)} pressure levels using {args.jobs} parallel jobs...")
        print("(Each job downloads one pressure level. This may take several minutes.)")
        
        # Create temporary directory for individual level files
        temp_dir = tempfile.mkdtemp(prefix='era5_download_')
        
        try:
            # Download each pressure level in parallel
            results = []
            with ThreadPoolExecutor(max_workers=args.jobs) as executor:
                futures = {}
                for idx, level in enumerate(pressure_levels):
                    future = executor.submit(
                        download_single_level,
                        api, level, variables,
                        args.latmin, args.latmax, args.lonmin, args.lonmax,
                        args.start_date, args.end_date, args.times,
                        temp_dir, idx, len(pressure_levels)
                    )
                    futures[future] = level
                
                # Collect results as they complete
                for future in as_completed(futures):
                    level, filepath = future.result()
                    results.append((level, filepath))
            
            # Sort results by pressure level for consistent ordering
            results.sort(key=lambda x: x[0])
            
            # Combine all files into one
            if not combine_netcdf_files(results, args.output):
                print("\n✗ Failed to combine NetCDF files")
                sys.exit(1)
            
            # Load and display summary
            print("\nLoading data for verification...")
            data = api.load_data(args.output)
            
            print("\nData Summary:")
            print(f"  Variables: {list(data['variables'].keys())}")
            print(f"  Coordinates: {list(data['coordinates'].keys())}")
            
            for coord_name, coord_values in data['coordinates'].items():
                if len(coord_values.shape) == 1:
                    print(f"  {coord_name}: {len(coord_values)} points, "
                          f"range [{coord_values.min():.2f}, {coord_values.max():.2f}]")
        
        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print(f"\nCleaned up temporary files")
        
        print("\n" + "=" * 70)
        print("Download completed successfully!")
        print("=" * 70)
        
    except ValueError as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease check your input parameters and try again.")
        sys.exit(1)
    
    except RuntimeError as e:
        print(f"\n✗ Error: {e}")
        print("\nData retrieval failed. This could be due to:")
        print("  - Invalid CDS API credentials")
        print("  - Network connectivity issues")
        print("  - CDS server problems")
        print("  - Need to accept CDS license terms")
        print("\nPlease check your configuration and try again.")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
