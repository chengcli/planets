#!/usr/bin/env python3
"""
Plot US city polygons on a map with coastal lines.

This script reads the us_cities.csv file and plots city boundaries on
a map using a specified projection with coastal lines.

Requirements:
    pip install matplotlib cartopy

Usage:
    # Plot all cities on default projection
    python plot_us_cities.py

    # Plot specific cities
    python plot_us_cities.py --cities pasadena-ca austin-tx boston-ma

    # Plot all cities in specific states
    python plot_us_cities.py --states California Texas

    # Use different projection
    python plot_us_cities.py --projection LambertConformal --cities seattle-wa

    # Save to file instead of displaying
    python plot_us_cities.py --output us_cities_map.png --states California

Available projections:
    - PlateCarree (default): Simple lat-lon projection
    - LambertConformal: Lambert conformal conic
    - Mercator: Mercator projection
    - Orthographic: Orthographic (globe) projection
    - Robinson: Robinson projection
    - AlbersEqualArea: Albers equal-area conic
"""

import argparse
import sys
from pathlib import Path

# Add earth directory to path
EARTH_DIR = Path(__file__).parent
sys.path.insert(0, str(EARTH_DIR))

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.path import Path as MPath
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ImportError as e:
    print("Error: Required plotting libraries not installed.")
    print("\nPlease install dependencies:")
    print("  pip install matplotlib cartopy")
    print("\nOr add to requirements.txt:")
    print("  matplotlib")
    print("  cartopy")
    sys.exit(1)

import generate_config


def get_projection(projection_name):
    """
    Get cartopy projection by name.
    
    Args:
        projection_name: Name of the projection
        
    Returns:
        Cartopy projection object
    """
    projections = {
        'PlateCarree': ccrs.PlateCarree(),
        'LambertConformal': ccrs.LambertConformal(central_longitude=-96, central_latitude=39),
        'Mercator': ccrs.Mercator(),
        'Orthographic': ccrs.Orthographic(central_longitude=-96, central_latitude=39),
        'Robinson': ccrs.Robinson(central_longitude=-96),
        'AlbersEqualArea': ccrs.AlbersEqualArea(central_longitude=-96, central_latitude=39),
    }
    
    if projection_name not in projections:
        print(f"Error: Unknown projection '{projection_name}'")
        print(f"Available projections: {', '.join(projections.keys())}")
        sys.exit(1)
    
    return projections[projection_name]


def plot_cities(locations_file, cities_to_plot=None, states_filter=None, 
                projection_name='PlateCarree', output_file=None, 
                show_city_names=True, figsize=(15, 10)):
    """
    Plot city polygons on a map.
    
    Args:
        locations_file: Path to CSV file with city polygons
        cities_to_plot: List of city IDs to plot (None = all cities)
        states_filter: List of state abbreviations to filter by
        projection_name: Name of map projection to use
        output_file: Path to save figure (None = display)
        show_city_names: Whether to show city name labels
        figsize: Figure size as (width, height) tuple
    """
    
    # Load locations
    print(f"Loading locations from {locations_file}...")
    locations = generate_config.load_locations(locations_file)
    
    # Filter cities if specified
    if cities_to_plot:
        cities_to_plot = [c.lower() for c in cities_to_plot]
        locations = {k: v for k, v in locations.items() if k in cities_to_plot}
        if not locations:
            print(f"Error: None of the specified cities found in {locations_file}")
            sys.exit(1)
    
    # Filter by state if specified
    if states_filter:
        states_filter = [s.lower() for s in states_filter]
        # City location IDs end with state abbreviation (e.g., 'pasadena-ca')
        locations = {k: v for k, v in locations.items() 
                    if k.split('-')[-1] in states_filter}
        if not locations:
            print(f"Error: No cities found for specified states in {locations_file}")
            sys.exit(1)
    
    print(f"Plotting {len(locations)} cities...")
    
    # Create figure and axis with projection
    projection = get_projection(projection_name)
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': projection})
    
    # Calculate extent based on selected cities
    all_lons = []
    all_lats = []
    for city_data in locations.values():
        polygon = city_data['polygon']
        all_lons.extend([p[0] for p in polygon])
        all_lats.extend([p[1] for p in polygon])
    
    # Set map extent with margin
    margin = 2  # degrees
    extent = [min(all_lons) - margin, max(all_lons) + margin,
              min(all_lats) - margin, max(all_lats) + margin]
    
    # Clip extent to reasonable bounds
    extent[0] = max(extent[0], -180)
    extent[1] = min(extent[1], 180)
    extent[2] = max(extent[2], -90)
    extent[3] = min(extent[3], 90)
    
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    # Add coastlines and features
    ax.coastlines(resolution='110m', linewidth=0.5, color='black')
    ax.add_feature(cfeature.BORDERS.with_scale('110m'), linewidth=0.5, edgecolor='gray')
    ax.add_feature(cfeature.STATES.with_scale('110m'), linewidth=0.3, edgecolor='lightgray')
    
    # Plot each city polygon
    for city_id, city_data in locations.items():
        name = city_data['name']
        polygon = city_data['polygon']
        
        # Convert polygon to matplotlib path
        vertices = [(lon, lat) for lon, lat in polygon]
        # Close the polygon
        vertices.append(vertices[0])
        
        # Create path
        codes = [MPath.MOVETO] + [MPath.LINETO] * (len(vertices) - 2) + [MPath.CLOSEPOLY]
        path = MPath(vertices, codes)
        
        # Create patch
        patch = mpatches.PathPatch(path, facecolor='lightblue', edgecolor='darkblue', 
                                   linewidth=1.5, alpha=0.6, transform=ccrs.PlateCarree())
        ax.add_patch(patch)
        
        # Add city name label at center
        if show_city_names:
            center = generate_config.calculate_center(polygon)
            # Extract just city name (without state)
            city_name_only = name.split(',')[0] if ',' in name else name
            ax.text(center['longitude'], center['latitude'], city_name_only, 
                   transform=ccrs.PlateCarree(),
                   ha='center', va='center', fontsize=7, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    # Add title
    if len(locations) <= 5:
        title = f"City Boundaries: {', '.join([locations[c]['name'] for c in list(locations.keys())[:5]])}"
        if len(locations) > 5:
            title += f" (+{len(locations) - 5} more)"
    else:
        title = f"US City Boundaries ({len(locations)} cities)"
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Save or display
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {output_file}")
    else:
        print("Displaying figure...")
        plt.show()
    
    plt.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Plot US city polygons on a map with coastal lines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--locations-file',
        default='us_cities.csv',
        help="Path to CSV file with city polygons (default: us_cities.csv)"
    )
    
    parser.add_argument(
        '--cities',
        nargs='+',
        help="Specific cities to plot by location ID (e.g., pasadena-ca austin-tx)"
    )
    
    parser.add_argument(
        '--states',
        nargs='+',
        help="Plot all cities in specified states (e.g., CA TX NY)"
    )
    
    parser.add_argument(
        '--projection',
        default='PlateCarree',
        help="Map projection to use (default: PlateCarree)"
    )
    
    parser.add_argument(
        '--output',
        help="Output file path. If not specified, displays the plot interactively."
    )
    
    parser.add_argument(
        '--no-labels',
        action='store_true',
        help="Don't show city name labels"
    )
    
    parser.add_argument(
        '--figsize',
        nargs=2,
        type=float,
        default=[15, 10],
        help="Figure size as width height (default: 15 10)"
    )
    
    args = parser.parse_args()
    
    # Resolve locations file path
    locations_file = Path(args.locations_file)
    if not locations_file.is_absolute():
        locations_file = EARTH_DIR / locations_file
    
    if not locations_file.exists():
        print(f"Error: Locations file not found: {locations_file}")
        print("\nGenerate the cities database first:")
        print("  python generate_us_cities_polygons.py")
        sys.exit(1)
    
    plot_cities(
        locations_file=locations_file,
        cities_to_plot=args.cities,
        states_filter=args.states,
        projection_name=args.projection,
        output_file=args.output,
        show_city_names=not args.no_labels,
        figsize=tuple(args.figsize)
    )


if __name__ == '__main__':
    main()
