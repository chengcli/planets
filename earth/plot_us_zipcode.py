#!/usr/bin/env python3
"""
Plot US zipcode polygons on a map with coastal lines.

This script queries the State-zip-code-GeoJSON database and plots zipcode
boundaries on a map using a specified projection with coastal lines.

Requirements:
    pip install matplotlib cartopy

Usage:
    # Plot a specific zipcode
    python plot_us_zipcode.py 48104

    # Plot multiple zipcodes
    python plot_us_zipcode.py 48104 90210 10001

    # Use different projection
    python plot_us_zipcode.py 48104 --projection PlateCarree

    # Save to file instead of displaying
    python plot_us_zipcode.py 48104 --output ann_arbor_zipcode.png
    
    # Provide state hint for faster querying
    python plot_us_zipcode.py 48104 --state mi --output ann_arbor.png

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

# Import the query function
try:
    from query_zipcode import get_zipcode_polygon
except ImportError:
    print("Error: query_zipcode module not found.")
    print("Make sure query_zipcode.py is in the same directory.")
    sys.exit(1)


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


def plot_zipcodes(zipcodes_to_plot, state_hint=None, projection_name='PlateCarree', 
                  output_file=None, show_labels=True, figsize=(12, 10), max_vertices=None):
    """
    Plot zipcode polygons on a map.
    
    Args:
        zipcodes_to_plot: List of zipcodes to plot
        state_hint: Optional state abbreviation to speed up queries
        projection_name: Name of map projection to use
        output_file: Path to save figure (None = display)
        show_labels: Whether to show zipcode labels
        figsize: Figure size as (width, height) tuple
        max_vertices: Maximum vertices per polygon (simplifies if needed)
    """
    
    # Query zipcodes from database
    print(f"Querying {len(zipcodes_to_plot)} zipcodes from State-zip-code-GeoJSON database...")
    zipcodes = {}
    for z in zipcodes_to_plot:
        z = str(z).zfill(5)
        print(f"  Querying {z}...", end=' ')
        polygon = get_zipcode_polygon(z, state_hint, max_vertices)
        if polygon:
            zipcodes[z] = polygon
            print(f"✓ ({len(polygon)} vertices)")
        else:
            print(f"✗ not found")
    
    if not zipcodes:
        print(f"Error: None of the specified zipcodes found in database")
        sys.exit(1)
    
    print(f"\nPlotting {len(zipcodes)} zipcodes...")
    
    # Create figure and axis with projection
    projection = get_projection(projection_name)
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': projection})
    
    # Calculate extent based on all polygons
    all_lons = []
    all_lats = []
    for polygon in zipcodes.values():
        all_lons.extend([p[0] for p in polygon])
        all_lats.extend([p[1] for p in polygon])
    
    # Set map extent with margin
    if len(zipcodes) == 1:
        margin = 0.5  # degrees
    else:
        margin = 1.0  # degrees
    
    ax.set_extent([min(all_lons) - margin, max(all_lons) + margin,
                   min(all_lats) - margin, max(all_lats) + margin],
                  crs=ccrs.PlateCarree())
    
    # Add coastlines and features
    ax.coastlines(resolution='50m', linewidth=0.5, color='black')
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.5, edgecolor='gray')
    ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.3, edgecolor='lightgray')
    
    # Plot each zipcode polygon
    for zipcode, polygon in zipcodes.items():
        # Convert polygon to matplotlib path
        vertices = [(lon, lat) for lon, lat in polygon]
        # Close the polygon if not already closed
        if vertices[0] != vertices[-1]:
            vertices.append(vertices[0])
        
        # Create path
        codes = [MPath.MOVETO] + [MPath.LINETO] * (len(vertices) - 2) + [MPath.CLOSEPOLY]
        path = MPath(vertices, codes)
        
        # Create patch
        patch = mpatches.PathPatch(path, facecolor='lightblue', edgecolor='darkblue',
                                   linewidth=2, alpha=0.6, transform=ccrs.PlateCarree())
        ax.add_patch(patch)
        
        # Add zipcode label at center
        if show_labels:
            center_lon = sum(p[0] for p in polygon) / len(polygon)
            center_lat = sum(p[1] for p in polygon) / len(polygon)
            ax.text(center_lon, center_lat, zipcode,
                   transform=ccrs.PlateCarree(),
                   ha='center', va='center', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    # Add title
    if len(zipcodes) == 1:
        title = f"Zipcode Boundary: {list(zipcodes.keys())[0]}"
    else:
        title = f"Zipcode Boundaries: {', '.join(sorted(zipcodes.keys()))}"
    
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
        description="Plot US zipcode polygons on a map with coastal lines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        'zipcodes',
        nargs='+',
        help="Zipcode(s) to plot (e.g., 48104 90210 10001)"
    )
    
    parser.add_argument(
        '--state',
        help="2-letter state abbreviation to speed up queries (e.g., mi, ca, ny)"
    )
    
    parser.add_argument(
        '--projection',
        default='PlateCarree',
        help="Map projection to use (default: PlateCarree). See --help for available projections."
    )
    
    parser.add_argument(
        '--max-vertices',
        type=int,
        help="Maximum vertices per polygon (simplifies if needed)"
    )
    
    parser.add_argument(
        '--output',
        help="Output file path. If not specified, displays the plot interactively."
    )
    
    parser.add_argument(
        '--no-labels',
        action='store_true',
        help="Don't show zipcode labels"
    )
    
    parser.add_argument(
        '--figsize',
        nargs=2,
        type=float,
        default=[12, 10],
        help="Figure size as width height (default: 12 10)"
    )
    
    args = parser.parse_args()
    
    plot_zipcodes(
        zipcodes_to_plot=args.zipcodes,
        state_hint=args.state,
        projection_name=args.projection,
        output_file=args.output,
        show_labels=not args.no_labels,
        figsize=tuple(args.figsize),
        max_vertices=args.max_vertices
    )


if __name__ == '__main__':
    main()
