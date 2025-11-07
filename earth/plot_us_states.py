#!/usr/bin/env python3
"""
Plot US state polygons on a map with coastal lines.

This script reads the us_states.csv file and plots state boundaries on
a map using a specified projection with coastal lines.

Requirements:
    pip install matplotlib cartopy

Usage:
    # Plot all states on default projection
    python plot_us_states.py

    # Plot specific states
    python plot_us_states.py --states california texas florida

    # Use different projection
    python plot_us_states.py --projection PlateCarree

    # Save to file instead of displaying
    python plot_us_states.py --output us_states_map.png

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


def plot_states(locations_file, states_to_plot=None, projection_name='PlateCarree', 
                output_file=None, show_state_names=True, figsize=(15, 10)):
    """
    Plot state polygons on a map.
    
    Args:
        locations_file: Path to CSV file with state polygons
        states_to_plot: List of state IDs to plot (None = all states)
        projection_name: Name of map projection to use
        output_file: Path to save figure (None = display)
        show_state_names: Whether to show state name labels
        figsize: Figure size as (width, height) tuple
    """
    
    # Load locations
    print(f"Loading locations from {locations_file}...")
    locations = generate_config.load_locations(locations_file)
    
    # Filter states if specified
    if states_to_plot:
        states_to_plot = [s.lower() for s in states_to_plot]
        locations = {k: v for k, v in locations.items() if k in states_to_plot}
        if not locations:
            print(f"Error: None of the specified states found in {locations_file}")
            sys.exit(1)
    
    print(f"Plotting {len(locations)} states...")
    
    # Create figure and axis with projection
    projection = get_projection(projection_name)
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': projection})
    
    # Set map extent for US
    if states_to_plot and len(states_to_plot) == 1:
        # For single state, zoom to that state's bounds
        state_id = list(locations.keys())[0]
        polygon = locations[state_id]['polygon']
        lons = [p[0] for p in polygon]
        lats = [p[1] for p in polygon]
        margin = 2  # degrees
        ax.set_extent([min(lons) - margin, max(lons) + margin, 
                      min(lats) - margin, max(lats) + margin], 
                      crs=ccrs.PlateCarree())
    else:
        # Full US extent
        ax.set_extent([-130, -65, 24, 50], crs=ccrs.PlateCarree())
    
    # Add coastlines and features
    # Note: Cartopy may download Natural Earth data on first use if not cached locally
    ax.coastlines(resolution='110m', linewidth=0.5, color='black')
    ax.add_feature(cfeature.BORDERS.with_scale('110m'), linewidth=0.5, edgecolor='gray')
    
    # Plot each state polygon
    for state_id, state_data in locations.items():
        name = state_data['name']
        polygon = state_data['polygon']
        
        # Convert polygon to matplotlib path
        vertices = [(lon, lat) for lon, lat in polygon]
        # Close the polygon
        vertices.append(vertices[0])
        
        # Create path
        codes = [MPath.MOVETO] + [MPath.LINETO] * (len(vertices) - 2) + [MPath.CLOSEPOLY]
        path = MPath(vertices, codes)
        
        # Create patch
        patch = mpatches.PathPatch(path, facecolor='lightcoral', edgecolor='darkred', 
                                   linewidth=1.5, alpha=0.6, transform=ccrs.PlateCarree())
        ax.add_patch(patch)
        
        # Add state name label at center
        if show_state_names:
            center = generate_config.calculate_center(polygon)
            ax.text(center['longitude'], center['latitude'], name, 
                   transform=ccrs.PlateCarree(),
                   ha='center', va='center', fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    # Add title
    if states_to_plot and len(states_to_plot) <= 3:
        title = f"State Boundaries: {', '.join([locations[s]['name'] for s in locations.keys()])}"
    else:
        title = f"US State Boundaries ({len(locations)} states)"
    
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
        description="Plot US state polygons on a map with coastal lines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--locations-file',
        default='us_states.csv',
        help="Path to CSV file with state polygons (default: us_states.csv)"
    )
    
    parser.add_argument(
        '--states',
        nargs='+',
        help="Specific states to plot (e.g., california texas). If not specified, plots all states."
    )
    
    parser.add_argument(
        '--projection',
        default='PlateCarree',
        help="Map projection to use (default: PlateCarree). See --help for available projections."
    )
    
    parser.add_argument(
        '--output',
        help="Output file path. If not specified, displays the plot interactively."
    )
    
    parser.add_argument(
        '--no-labels',
        action='store_true',
        help="Don't show state name labels"
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
        sys.exit(1)
    
    plot_states(
        locations_file=locations_file,
        states_to_plot=args.states,
        projection_name=args.projection,
        output_file=args.output,
        show_state_names=not args.no_labels,
        figsize=tuple(args.figsize)
    )


if __name__ == '__main__':
    main()
