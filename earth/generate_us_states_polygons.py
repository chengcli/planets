#!/usr/bin/env python3
"""
Generate US states location table with actual polygon boundaries.

This script downloads GeoJSON data for US states and converts it to the
locations.csv format, preserving the actual state boundaries (not just
bounding boxes).

Usage:
    python generate_us_states_polygons.py [--output OUTPUT_FILE] [--max-vertices N]

Options:
    --output FILE         Output CSV file path (default: us_states.csv)
    --max-vertices N      Maximum vertices per polygon (default: 50)
"""

import json
import urllib.request
import argparse


def location_id_from_name(name):
    """Convert state name to location ID."""
    return name.lower().replace(' ', '-')


def format_polygon_vertices(polygon):
    """Format polygon vertices for CSV."""
    return ';'.join([f"{lon},{lat}" for lon, lat in polygon])


def simplify_polygon(coordinates, target_vertices=50):
    """
    Simplify a polygon using even sampling.
    
    For polygons with more vertices than the target, sample evenly spaced
    points to reduce vertex count while maintaining shape.
    
    Args:
        coordinates: List of [lon, lat] pairs
        target_vertices: Maximum number of points to keep
    
    Returns:
        List of [lon, lat] pairs (simplified)
    """
    if len(coordinates) <= target_vertices:
        return coordinates
    
    # Sample evenly spaced points
    step = len(coordinates) / target_vertices
    simplified = []
    for i in range(target_vertices):
        idx = int(i * step)
        simplified.append(coordinates[idx])
    
    return simplified


def process_geojson(output_file, max_vertices=50):
    """
    Download GeoJSON and create CSV with actual polygon boundaries.
    
    Args:
        output_file: Path to output CSV file
        max_vertices: Maximum vertices per state polygon
    """
    
    # Download US states GeoJSON
    url = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
    print(f"Downloading data from {url}...")
    
    try:
        with urllib.request.urlopen(url) as response:
            data = json.load(response)
    except Exception as e:
        print(f"Error downloading data: {e}")
        return 1
    
    print(f"Processing {len(data['features'])} states/territories...")
    
    # Write CSV header
    with open(output_file, 'w') as f:
        # Write comments explaining the format
        f.write("# US States Location Table\n")
        f.write("# Format: location_id\tname\tpolygon_vertices\n")
        f.write("# polygon_vertices format: lon1,lat1;lon2,lat2;lon3,lat3;...\n")
        f.write(f"# Vertices represent actual state boundaries (simplified to up to {max_vertices} vertices per state)\n")
        f.write("# Vertices are in counterclockwise order\n")
        f.write("# Longitude: negative for West, positive for East\n")
        f.write("# Latitude: negative for South, positive for North\n")
        f.write("# Note: Alaska uses extended western longitude (around -189° to -130°) for consistency\n")
        f.write("#       across the International Date Line. Scripts calculate rectangular\n")
        f.write("#       bounds from these polygons automatically.\n")
        f.write("location_id\tname\tpolygon_vertices\n")
        
        # Process each state
        for feature in data['features']:
            name = feature['properties']['name']
            location_id = location_id_from_name(name)
            
            # Get coordinates - handle both Polygon and MultiPolygon
            geometry = feature['geometry']
            coordinates = []
            
            if geometry['type'] == 'Polygon':
                # Polygon has one outer ring
                coordinates = geometry['coordinates'][0]
            elif geometry['type'] == 'MultiPolygon':
                # MultiPolygon - use the largest polygon
                largest = max(geometry['coordinates'], key=lambda p: len(p[0]))
                coordinates = largest[0]
            
            # Simplify if too many vertices
            if len(coordinates) > max_vertices:
                coordinates = simplify_polygon(coordinates, target_vertices=max_vertices)
            
            # Format vertices
            vertices_str = format_polygon_vertices(coordinates)
            
            # Write the row
            f.write(f"{location_id}\t{name}\t{vertices_str}\n")
            
            print(f"  {name}: {len(coordinates)} vertices")
    
    print(f"\nOutput written to: {output_file}")
    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate US states location table with actual polygon boundaries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--output',
        default='us_states.csv',
        help="Output CSV file path (default: us_states.csv)"
    )
    
    parser.add_argument(
        '--max-vertices',
        type=int,
        default=50,
        help="Maximum vertices per polygon (default: 50)"
    )
    
    args = parser.parse_args()
    
    return process_geojson(args.output, args.max_vertices)


if __name__ == '__main__':
    import sys
    sys.exit(main())
