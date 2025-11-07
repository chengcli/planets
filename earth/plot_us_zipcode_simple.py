#!/usr/bin/env python3
"""
Plot US zipcode polygons using simple matplotlib (no internet required).

This script reads the us_zipcode.csv file and plots zipcode boundaries
using simple matplotlib without map projections. This is useful when
you don't have cartopy installed or internet access.

Requirements:
    pip install matplotlib

Usage:
    # Plot a specific zipcode
    python plot_us_zipcode_simple.py 48104

    # Plot multiple zipcodes
    python plot_us_zipcode_simple.py 48104 90210 10001

    # Save to file instead of displaying
    python plot_us_zipcode_simple.py 48104 --output ann_arbor_zipcode.png
"""

import argparse
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.path import Path as MPath
except ImportError:
    print("Error: matplotlib not installed.")
    print("\nPlease install:")
    print("  pip install matplotlib")
    sys.exit(1)


def load_zipcodes(zipcode_file):
    """
    Load zipcode polygons from CSV file.
    
    Args:
        zipcode_file: Path to CSV file with zipcode polygons
        
    Returns:
        dict: Dictionary mapping zipcode to polygon coordinates
    """
    zipcodes = {}
    
    with open(zipcode_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip comments and header
            if line.startswith('#') or line.startswith('zipcode'):
                continue
            
            if not line:
                continue
            
            # Parse line
            parts = line.split('\t')
            if len(parts) != 2:
                continue
            
            zipcode = parts[0]
            vertices_str = parts[1]
            
            # Parse vertices
            polygon = []
            for vertex in vertices_str.split(';'):
                if ',' in vertex:
                    lon, lat = vertex.split(',')
                    polygon.append((float(lon), float(lat)))
            
            zipcodes[zipcode] = polygon
    
    return zipcodes


def plot_zipcodes_simple(zipcode_file, zipcodes_to_plot, output_file=None, 
                        show_labels=True, figsize=(12, 10)):
    """
    Plot zipcode polygons using simple matplotlib.
    
    Args:
        zipcode_file: Path to CSV file with zipcode polygons
        zipcodes_to_plot: List of zipcodes to plot
        output_file: Path to save figure (None = display)
        show_labels: Whether to show zipcode labels
        figsize: Figure size as (width, height) tuple
    """
    
    # Load zipcodes
    print(f"Loading zipcodes from {zipcode_file}...")
    all_zipcodes = load_zipcodes(zipcode_file)
    
    # Filter to requested zipcodes
    zipcodes = {}
    for z in zipcodes_to_plot:
        z = str(z).zfill(5)
        if z in all_zipcodes:
            zipcodes[z] = all_zipcodes[z]
        else:
            print(f"Warning: Zipcode {z} not found in {zipcode_file}")
    
    if not zipcodes:
        print(f"Error: None of the specified zipcodes found in {zipcode_file}")
        print(f"Available zipcodes: {', '.join(sorted(all_zipcodes.keys()))}")
        sys.exit(1)
    
    print(f"Plotting {len(zipcodes)} zipcodes...")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate extent based on all polygons
    all_lons = []
    all_lats = []
    for polygon in zipcodes.values():
        all_lons.extend([p[0] for p in polygon])
        all_lats.extend([p[1] for p in polygon])
    
    # Plot each zipcode polygon
    # Use a more robust color generation for many zipcodes
    if len(zipcodes) <= 12:
        # Use Set3 colormap for small number of zipcodes (up to 12 distinct colors)
        colors = plt.cm.Set3.colors
    else:
        # Use tab20 colormap for more zipcodes (up to 20 distinct colors)
        colors = plt.cm.tab20.colors
    
    for i, (zipcode, polygon) in enumerate(zipcodes.items()):
        # Convert polygon to matplotlib path
        vertices = [(lon, lat) for lon, lat in polygon]
        # Close the polygon if not already closed
        if vertices[0] != vertices[-1]:
            vertices.append(vertices[0])
        
        # Create path
        codes = [MPath.MOVETO] + [MPath.LINETO] * (len(vertices) - 2) + [MPath.CLOSEPOLY]
        path = MPath(vertices, codes)
        
        # Create patch with different colors for each zipcode
        color = colors[i % len(colors)]
        patch = mpatches.PathPatch(path, facecolor=color, edgecolor='darkblue',
                                   linewidth=2, alpha=0.6)
        ax.add_patch(patch)
        
        # Add zipcode label at center
        if show_labels:
            center_lon = sum(p[0] for p in polygon) / len(polygon)
            center_lat = sum(p[1] for p in polygon) / len(polygon)
            ax.text(center_lon, center_lat, zipcode,
                   ha='center', va='center', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                            edgecolor='darkblue', alpha=0.9, linewidth=1.5))
    
    # Set map extent with margin
    if len(zipcodes) == 1:
        margin = 0.1  # degrees
    else:
        margin = 0.2  # degrees
    
    ax.set_xlim(min(all_lons) - margin, max(all_lons) + margin)
    ax.set_ylim(min(all_lats) - margin, max(all_lats) + margin)
    
    # Add grid
    ax.grid(True, linewidth=0.5, color='gray', alpha=0.3, linestyle='--')
    
    # Add labels
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    
    # Add title
    if len(zipcodes) == 1:
        title = f"Zipcode Boundary: {list(zipcodes.keys())[0]}"
    else:
        title = f"Zipcode Boundaries: {', '.join(sorted(zipcodes.keys()))}"
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Set aspect ratio to be equal
    ax.set_aspect('equal', adjustable='box')
    
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
    # Add earth directory to path
    EARTH_DIR = Path(__file__).parent
    
    parser = argparse.ArgumentParser(
        description="Plot US zipcode polygons using simple matplotlib",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        'zipcodes',
        nargs='+',
        help="Zipcode(s) to plot (e.g., 48104 90210 10001)"
    )
    
    parser.add_argument(
        '--zipcode-file',
        default='us_zipcode.csv',
        help="Path to CSV file with zipcode polygons (default: us_zipcode.csv)"
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
    
    # Resolve zipcode file path
    zipcode_file = Path(args.zipcode_file)
    if not zipcode_file.is_absolute():
        zipcode_file = EARTH_DIR / zipcode_file
    
    if not zipcode_file.exists():
        print(f"Error: Zipcode file not found: {zipcode_file}")
        sys.exit(1)
    
    plot_zipcodes_simple(
        zipcode_file=zipcode_file,
        zipcodes_to_plot=args.zipcodes,
        output_file=args.output,
        show_labels=not args.no_labels,
        figsize=tuple(args.figsize)
    )


if __name__ == '__main__':
    main()
