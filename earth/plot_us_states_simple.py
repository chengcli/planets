#!/usr/bin/env python3
"""
Plot US state polygons using matplotlib (simpler version without cartopy).

This script reads the us_states.csv file and plots state boundaries
using basic matplotlib without map projections.

Requirements:
    pip install matplotlib

Usage:
    # Plot all states
    python plot_us_states_simple.py

    # Plot specific states
    python plot_us_states_simple.py --states california texas florida

    # Save to file
    python plot_us_states_simple.py --output us_states_map.png
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
except ImportError as e:
    print("Error: matplotlib not installed.")
    print("\nPlease install: pip install matplotlib")
    sys.exit(1)

import generate_config


def plot_states_simple(locations_file, states_to_plot=None, output_file=None, 
                       show_state_names=True, figsize=(15, 10)):
    """
    Plot state polygons using simple lat-lon coordinates.
    
    Args:
        locations_file: Path to CSV file with state polygons
        states_to_plot: List of state IDs to plot (None = all states)
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
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each state polygon
    for state_id, state_data in locations.items():
        name = state_data['name']
        polygon = state_data['polygon']
        
        # Extract lons and lats
        lons = [p[0] for p in polygon]
        lats = [p[1] for p in polygon]
        
        # Close the polygon
        lons_closed = lons + [lons[0]]
        lats_closed = lats + [lats[0]]
        
        # Plot polygon boundary
        ax.plot(lons_closed, lats_closed, 'darkred', linewidth=1.5)
        
        # Fill polygon
        ax.fill(lons_closed, lats_closed, color='lightcoral', alpha=0.6)
        
        # Add state name label at center
        if show_state_names:
            center = generate_config.calculate_center(polygon)
            ax.text(center['longitude'], center['latitude'], name,
                   ha='center', va='center', fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Set aspect ratio and labels
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel('Longitude (°)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latitude (°)', fontsize=12, fontweight='bold')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.5, color='gray')
    
    # Set background color
    ax.set_facecolor('lightblue')
    
    # Add title
    if states_to_plot and len(states_to_plot) <= 3:
        title = f"State Boundaries: {', '.join([locations[s]['name'] for s in locations.keys()])}"
    else:
        title = f"US State Boundaries ({len(locations)} states)"
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Adjust layout
    plt.tight_layout()
    
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
        description="Plot US state polygons (simple version without map projections)",
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
    
    plot_states_simple(
        locations_file=locations_file,
        states_to_plot=args.states,
        output_file=args.output,
        show_state_names=not args.no_labels,
        figsize=tuple(args.figsize)
    )


if __name__ == '__main__':
    main()
