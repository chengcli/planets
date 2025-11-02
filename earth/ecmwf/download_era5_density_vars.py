#!/usr/bin/env python3
"""
Convenient script to download density-related variables from ERA5.

This script downloads specific humidity and cloud/precipitation water content
variables at all standard ERA5 pressure levels for a specified region and time period.

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
from datetime import datetime

# Add current directory to path for importing local module
sys.path.insert(0, os.path.dirname(__file__))

from ecmwf_weather_api import ECMWFWeatherAPI


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
        print("\nFetching data from CDS...")
        print("(This may take several minutes depending on request size and server load)")
        
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
        
        # Fetch the data
        output_file = api.fetch_weather_data(
            latmin=args.latmin,
            latmax=args.latmax,
            lonmin=args.lonmin,
            lonmax=args.lonmax,
            start_date=args.start_date,
            end_date=args.end_date,
            variables=variables,
            pressure_levels=pressure_levels,
            output_file=args.output,
            times=args.times
        )
        
        print(f"\n✓ Data successfully downloaded!")
        print(f"\nOutput file: {output_file}")
        
        # Load and display summary
        print("\nLoading data for verification...")
        data = api.load_data(output_file)
        
        print("\nData Summary:")
        print(f"  Variables: {list(data['variables'].keys())}")
        print(f"  Coordinates: {list(data['coordinates'].keys())}")
        
        for coord_name, coord_values in data['coordinates'].items():
            if len(coord_values.shape) == 1:
                print(f"  {coord_name}: {len(coord_values)} points, "
                      f"range [{coord_values.min():.2f}, {coord_values.max():.2f}]")
        
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
