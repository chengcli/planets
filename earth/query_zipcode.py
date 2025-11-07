#!/usr/bin/env python3
"""
Query US zipcode polygon boundaries from the State-zip-code-GeoJSON database.

This module provides functions to query zipcode polygon data from the 
chengcli/State-zip-code-GeoJSON repository on GitHub, which contains 
GeoJSON files with zipcode boundaries for all 50 US states plus DC.

Usage:
    from query_zipcode import get_zipcode_polygon
    
    # Get polygon for a specific zipcode
    polygon = get_zipcode_polygon('48104')
    
    # Or use as a command-line tool
    python query_zipcode.py 48104 90210 10001

Repository: https://github.com/chengcli/State-zip-code-GeoJSON
"""

import json
import urllib.request
import urllib.error
import argparse
import sys
from typing import Optional, List, Tuple, Dict


# Mapping of state abbreviations to full names (used in repository file names)
STATE_MAPPING = {
    'al': 'alabama', 'ak': 'alaska', 'az': 'arizona', 'ar': 'arkansas',
    'ca': 'california', 'co': 'colorado', 'ct': 'connecticut', 'de': 'delaware',
    'fl': 'florida', 'ga': 'georgia', 'hi': 'hawaii', 'id': 'idaho',
    'il': 'illinois', 'in': 'indiana', 'ia': 'iowa', 'ks': 'kansas',
    'ky': 'kentucky', 'la': 'louisiana', 'me': 'maine', 'md': 'maryland',
    'ma': 'massachusetts', 'mi': 'michigan', 'mn': 'minnesota', 'ms': 'mississippi',
    'mo': 'missouri', 'mt': 'montana', 'ne': 'nebraska', 'nv': 'nevada',
    'nh': 'new_hampshire', 'nj': 'new_jersey', 'nm': 'new_mexico', 'ny': 'new_york',
    'nc': 'north_carolina', 'nd': 'north_dakota', 'oh': 'ohio', 'ok': 'oklahoma',
    'or': 'oregon', 'pa': 'pennsylvania', 'ri': 'rhode_island', 'sc': 'south_carolina',
    'sd': 'south_dakota', 'tn': 'tennessee', 'tx': 'texas', 'ut': 'utah',
    'vt': 'vermont', 'va': 'virginia', 'wa': 'washington', 'wv': 'west_virginia',
    'wi': 'wisconsin', 'wy': 'wyoming', 'dc': 'district_of_columbia'
}

GITHUB_BASE_URL = "https://raw.githubusercontent.com/chengcli/State-zip-code-GeoJSON/master"


def get_zipcode_polygon(zipcode: str, state_hint: Optional[str] = None, 
                       max_vertices: Optional[int] = None) -> Optional[List[Tuple[float, float]]]:
    """
    Query the State-zip-code-GeoJSON database and return polygon boundary for a zipcode.
    
    Args:
        zipcode: 5-digit ZIP code as string (e.g., '48104')
        state_hint: Optional 2-letter state abbreviation to narrow search (e.g., 'mi')
        max_vertices: If specified, simplify polygon to this many vertices
        
    Returns:
        List of (longitude, latitude) tuples representing polygon boundary,
        or None if zipcode not found
        
    Example:
        >>> polygon = get_zipcode_polygon('48104', 'mi')
        >>> if polygon:
        ...     print(f"Found {len(polygon)} vertices")
    """
    # Ensure zipcode is 5 digits
    zipcode = str(zipcode).zfill(5)
    
    # If state hint provided, try that state first
    if state_hint:
        state_hint = state_hint.lower()
        if state_hint in STATE_MAPPING:
            polygon = _search_state_for_zipcode(state_hint, zipcode)
            if polygon:
                return _simplify_if_needed(polygon, max_vertices)
    
    # Try to infer state from zipcode prefix (rough approximation)
    inferred_states = _infer_states_from_zipcode(zipcode)
    for state_abbr in inferred_states:
        polygon = _search_state_for_zipcode(state_abbr, zipcode)
        if polygon:
            return _simplify_if_needed(polygon, max_vertices)
    
    # If still not found, search all states (slow but thorough)
    print(f"Searching all states for zipcode {zipcode}...", file=sys.stderr)
    for state_abbr in STATE_MAPPING.keys():
        if state_hint and state_abbr == state_hint:
            continue  # Already tried
        if state_abbr in inferred_states:
            continue  # Already tried
            
        polygon = _search_state_for_zipcode(state_abbr, zipcode)
        if polygon:
            return _simplify_if_needed(polygon, max_vertices)
    
    return None


def _search_state_for_zipcode(state_abbr: str, zipcode: str) -> Optional[List[Tuple[float, float]]]:
    """
    Search a specific state's GeoJSON file for a zipcode.
    
    Args:
        state_abbr: 2-letter state abbreviation
        zipcode: 5-digit ZIP code
        
    Returns:
        List of (lon, lat) tuples or None if not found
    """
    state_name = STATE_MAPPING.get(state_abbr)
    if not state_name:
        return None
    
    url = f"{GITHUB_BASE_URL}/{state_abbr}_{state_name}_zip_codes_geo.min.json"
    
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.load(response)
            
        # Search for the zipcode in features
        for feature in data.get('features', []):
            props = feature.get('properties', {})
            zcta = props.get('ZCTA5CE10', '')
            
            if zcta == zipcode:
                # Found it! Extract polygon coordinates
                geometry = feature.get('geometry', {})
                return _extract_coordinates(geometry)
                
    except urllib.error.URLError:
        # State file not accessible, skip
        pass
    except Exception as e:
        print(f"Warning: Error reading {state_abbr}: {e}", file=sys.stderr)
    
    return None


def _extract_coordinates(geometry: dict) -> Optional[List[Tuple[float, float]]]:
    """
    Extract coordinates from GeoJSON geometry.
    
    Args:
        geometry: GeoJSON geometry object
        
    Returns:
        List of (lon, lat) tuples
    """
    geom_type = geometry.get('type')
    coords = geometry.get('coordinates', [])
    
    if geom_type == 'Polygon':
        # Polygon: use outer ring (first array)
        if coords and len(coords) > 0:
            return [(lon, lat) for lon, lat in coords[0]]
    elif geom_type == 'MultiPolygon':
        # MultiPolygon: use the largest polygon
        if coords:
            largest = max(coords, key=lambda p: len(p[0]) if p else 0)
            if largest and len(largest) > 0:
                return [(lon, lat) for lon, lat in largest[0]]
    
    return None


def _infer_states_from_zipcode(zipcode: str) -> List[str]:
    """
    Infer likely states based on ZIP code prefix.
    
    This is a rough heuristic based on ZIP code ranges.
    
    Args:
        zipcode: 5-digit ZIP code
        
    Returns:
        List of likely state abbreviations
    """
    if not zipcode or len(zipcode) < 1:
        return []
    
    first_digit = zipcode[0]
    
    # Rough mapping of first digit to regions
    region_map = {
        '0': ['ct', 'ma', 'me', 'nh', 'nj', 'ny', 'ri', 'vt'],  # Northeast
        '1': ['de', 'ny', 'pa'],  # NY, PA area
        '2': ['dc', 'md', 'nc', 'sc', 'va', 'wv'],  # Mid-Atlantic
        '3': ['al', 'fl', 'ga', 'ms', 'tn'],  # Southeast
        '4': ['in', 'ky', 'mi', 'oh'],  # Great Lakes
        '5': ['ia', 'mn', 'mt', 'nd', 'sd', 'wi'],  # Upper Midwest
        '6': ['il', 'ks', 'mo', 'ne'],  # Central
        '7': ['ar', 'la', 'ok', 'tx'],  # South Central
        '8': ['az', 'co', 'id', 'nm', 'nv', 'ut', 'wy'],  # Mountain
        '9': ['ak', 'ca', 'hi', 'or', 'wa'],  # Pacific
    }
    
    return region_map.get(first_digit, [])


def _simplify_if_needed(polygon: List[Tuple[float, float]], 
                       max_vertices: Optional[int]) -> List[Tuple[float, float]]:
    """
    Simplify polygon if max_vertices is specified.
    
    Args:
        polygon: List of (lon, lat) tuples
        max_vertices: Maximum vertices to keep
        
    Returns:
        Simplified polygon
    """
    if max_vertices is None or len(polygon) <= max_vertices:
        return polygon
    
    # Simple uniform sampling
    step = len(polygon) / max_vertices
    simplified = []
    for i in range(max_vertices):
        idx = int(i * step)
        simplified.append(polygon[idx])
    
    return simplified


def format_polygon_vertices(polygon: List[Tuple[float, float]]) -> str:
    """
    Format polygon vertices as semicolon-separated string.
    
    Args:
        polygon: List of (lon, lat) tuples
        
    Returns:
        String in format "lon1,lat1;lon2,lat2;..."
    """
    return ';'.join([f"{lon},{lat}" for lon, lat in polygon])


def main():
    """Command-line interface for querying zipcodes."""
    parser = argparse.ArgumentParser(
        description="Query US zipcode polygon boundaries from State-zip-code-GeoJSON database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        'zipcodes',
        nargs='+',
        help="Zipcode(s) to query (e.g., 48104 90210 10001)"
    )
    
    parser.add_argument(
        '--state',
        help="2-letter state abbreviation to narrow search (e.g., mi, ca, ny)"
    )
    
    parser.add_argument(
        '--max-vertices',
        type=int,
        help="Maximum vertices per polygon (simplifies if needed)"
    )
    
    parser.add_argument(
        '--format',
        choices=['json', 'csv', 'wkt'],
        default='csv',
        help="Output format (default: csv)"
    )
    
    args = parser.parse_args()
    
    results = {}
    for zipcode in args.zipcodes:
        print(f"Querying zipcode {zipcode}...", file=sys.stderr)
        polygon = get_zipcode_polygon(zipcode, args.state, args.max_vertices)
        
        if polygon:
            results[zipcode] = polygon
            print(f"✓ Found {zipcode}: {len(polygon)} vertices", file=sys.stderr)
        else:
            print(f"✗ Zipcode {zipcode} not found", file=sys.stderr)
    
    # Output results
    if args.format == 'csv':
        print("zipcode\tpolygon_vertices")
        for zipcode, polygon in results.items():
            vertices_str = format_polygon_vertices(polygon)
            print(f"{zipcode}\t{vertices_str}")
    
    elif args.format == 'json':
        output = {
            zipcode: [{'lon': lon, 'lat': lat} for lon, lat in polygon]
            for zipcode, polygon in results.items()
        }
        print(json.dumps(output, indent=2))
    
    elif args.format == 'wkt':
        for zipcode, polygon in results.items():
            coords_str = ', '.join([f"{lon} {lat}" for lon, lat in polygon])
            print(f"{zipcode}\tPOLYGON(({coords_str}))")
    
    return 0 if results else 1


if __name__ == '__main__':
    sys.exit(main())
