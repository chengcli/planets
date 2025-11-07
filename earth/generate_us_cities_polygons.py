#!/usr/bin/env python3
"""
Generate US cities location table with polygon boundaries.

This script fetches GeoJSON data for major US cities from OpenStreetMap's
Nominatim API and converts it to the locations.csv format.

Usage:
    python generate_us_cities_polygons.py [options]

Examples:
    # Generate with default settings (top 3-5 cities per state)
    python generate_us_cities_polygons.py

    # Generate for specific states only
    python generate_us_cities_polygons.py --states California Texas

    # Customize output file and vertex count
    python generate_us_cities_polygons.py --output custom_cities.csv --max-vertices 40

    # Use custom delay between API requests
    python generate_us_cities_polygons.py --delay 1.5

Options:
    --output FILE         Output CSV file path (default: us_cities.csv)
    --max-vertices N      Maximum vertices per polygon (default: 50)
    --states STATE [...]  Only fetch cities for specified states
    --delay SECONDS       Delay between API requests (default: 1.0 second)
    --dry-run            Show what would be fetched without making requests

Note: This script uses OpenStreetMap's Nominatim API which has usage limits.
      Please be respectful and use appropriate delays between requests.
      See: https://operations.osmfoundation.org/policies/nominatim/
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error
import urllib.parse
import re
from pathlib import Path


# US State name to abbreviation mapping
STATE_ABBREVIATIONS = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
    'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
    'District of Columbia': 'DC', 'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI',
    'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
    'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME',
    'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN',
    'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE',
    'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM',
    'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH',
    'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Puerto Rico': 'PR',
    'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD', 'Tennessee': 'TN',
    'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA',
    'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY',
}


# Major cities per state (typically 3-5 largest cities, more for populous states)
MAJOR_CITIES = {
    'Alabama': ['Birmingham', 'Montgomery', 'Mobile', 'Huntsville'],
    'Alaska': ['Anchorage', 'Fairbanks', 'Juneau'],
    'Arizona': ['Phoenix', 'Tucson', 'Mesa', 'Chandler', 'Scottsdale'],
    'Arkansas': ['Little Rock', 'Fort Smith', 'Fayetteville'],
    'California': ['Los Angeles', 'San Diego', 'San Jose', 'San Francisco', 'Fresno', 
                   'Sacramento', 'Oakland', 'Bakersfield', 'Anaheim', 'Riverside', 'Pasadena'],
    'Colorado': ['Denver', 'Colorado Springs', 'Aurora', 'Fort Collins'],
    'Connecticut': ['Bridgeport', 'New Haven', 'Hartford', 'Stamford'],
    'Delaware': ['Wilmington', 'Dover', 'Newark'],
    'District of Columbia': ['Washington'],
    'Florida': ['Jacksonville', 'Miami', 'Tampa', 'Orlando', 'St Petersburg', 'Tallahassee'],
    'Georgia': ['Atlanta', 'Augusta', 'Columbus', 'Savannah', 'Athens'],
    'Hawaii': ['Honolulu', 'Hilo', 'Kailua'],
    'Idaho': ['Boise', 'Meridian', 'Nampa'],
    'Illinois': ['Chicago', 'Aurora', 'Naperville', 'Joliet', 'Rockford', 'Springfield'],
    'Indiana': ['Indianapolis', 'Fort Wayne', 'Evansville', 'South Bend'],
    'Iowa': ['Des Moines', 'Cedar Rapids', 'Davenport'],
    'Kansas': ['Wichita', 'Overland Park', 'Kansas City', 'Topeka'],
    'Kentucky': ['Louisville', 'Lexington', 'Bowling Green', 'Frankfort'],
    'Louisiana': ['New Orleans', 'Baton Rouge', 'Shreveport', 'Lafayette'],
    'Maine': ['Portland', 'Lewiston', 'Bangor', 'Augusta'],
    'Maryland': ['Baltimore', 'Frederick', 'Rockville', 'Annapolis'],
    'Massachusetts': ['Boston', 'Worcester', 'Springfield', 'Cambridge', 'Lowell'],
    'Michigan': ['Detroit', 'Grand Rapids', 'Warren', 'Sterling Heights', 'Ann Arbor', 'Lansing'],
    'Minnesota': ['Minneapolis', 'St Paul', 'Rochester', 'Duluth'],
    'Mississippi': ['Jackson', 'Gulfport', 'Southaven', 'Biloxi'],
    'Missouri': ['Kansas City', 'St Louis', 'Springfield', 'Columbia', 'Jefferson City'],
    'Montana': ['Billings', 'Missoula', 'Great Falls', 'Helena'],
    'Nebraska': ['Omaha', 'Lincoln', 'Bellevue'],
    'Nevada': ['Las Vegas', 'Henderson', 'Reno', 'Carson City'],
    'New Hampshire': ['Manchester', 'Nashua', 'Concord'],
    'New Jersey': ['Newark', 'Jersey City', 'Paterson', 'Elizabeth', 'Trenton'],
    'New Mexico': ['Albuquerque', 'Las Cruces', 'Rio Rancho', 'Santa Fe'],
    'New York': ['New York', 'Buffalo', 'Rochester', 'Yonkers', 'Syracuse', 'Albany'],
    'North Carolina': ['Charlotte', 'Raleigh', 'Greensboro', 'Durham', 'Winston-Salem', 'Fayetteville'],
    'North Dakota': ['Fargo', 'Bismarck', 'Grand Forks'],
    'Ohio': ['Columbus', 'Cleveland', 'Cincinnati', 'Toledo', 'Akron', 'Dayton'],
    'Oklahoma': ['Oklahoma City', 'Tulsa', 'Norman'],
    'Oregon': ['Portland', 'Salem', 'Eugene', 'Gresham'],
    'Pennsylvania': ['Philadelphia', 'Pittsburgh', 'Allentown', 'Erie', 'Harrisburg'],
    'Puerto Rico': ['San Juan', 'Bayamon', 'Carolina', 'Ponce'],
    'Rhode Island': ['Providence', 'Warwick', 'Cranston'],
    'South Carolina': ['Charleston', 'Columbia', 'North Charleston', 'Greenville'],
    'South Dakota': ['Sioux Falls', 'Rapid City', 'Aberdeen', 'Pierre'],
    'Tennessee': ['Nashville', 'Memphis', 'Knoxville', 'Chattanooga'],
    'Texas': ['Houston', 'San Antonio', 'Dallas', 'Austin', 'Fort Worth', 'El Paso', 
              'Arlington', 'Corpus Christi', 'Plano'],
    'Utah': ['Salt Lake City', 'West Valley City', 'Provo', 'West Jordan'],
    'Vermont': ['Burlington', 'South Burlington', 'Rutland', 'Montpelier'],
    'Virginia': ['Virginia Beach', 'Norfolk', 'Chesapeake', 'Richmond', 'Newport News', 'Arlington'],
    'Washington': ['Seattle', 'Spokane', 'Tacoma', 'Vancouver', 'Bellevue', 'Olympia'],
    'West Virginia': ['Charleston', 'Huntington', 'Morgantown'],
    'Wisconsin': ['Milwaukee', 'Madison', 'Green Bay', 'Kenosha'],
    'Wyoming': ['Cheyenne', 'Casper', 'Laramie'],
}


def location_id_from_city_state(city, state_abbr):
    """
    Convert city name and state abbreviation to location ID.
    
    Format: cityname-stateabbr (e.g., 'pasadena-ca')
    """
    # Remove special characters and convert to lowercase
    city_clean = city.lower()
    # Replace spaces, apostrophes, and other punctuation with hyphens
    city_clean = re.sub(r"['\s.,()\[\]]+", '-', city_clean)
    # Remove any remaining non-alphanumeric characters except hyphens
    city_clean = re.sub(r'[^a-z0-9-]', '', city_clean)
    # Remove leading/trailing hyphens and collapse multiple hyphens
    city_clean = re.sub(r'-+', '-', city_clean).strip('-')
    state_clean = state_abbr.lower()
    return f"{city_clean}-{state_clean}"


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


def fetch_city_boundary(city, state, delay=1.0):
    """
    Fetch city boundary from OpenStreetMap Nominatim API.
    
    Args:
        city: City name
        state: State name
        delay: Delay in seconds before making request (for rate limiting)
    
    Returns:
        List of [lon, lat] coordinate pairs, or None if failed
    """
    # Be respectful of API limits - add delay
    time.sleep(delay)
    
    # Build Nominatim query
    base_url = "https://nominatim.openstreetmap.org/search"
    params = {
        'city': city,
        'state': state,
        'country': 'USA',
        'format': 'json',
        'polygon_geojson': '1',
        'limit': '1'
    }
    
    # Build URL with parameters
    query_parts = [f"{k}={urllib.parse.quote(str(v))}" for k, v in params.items()]
    url = f"{base_url}?{'&'.join(query_parts)}"
    
    try:
        # Add User-Agent header as required by Nominatim usage policy
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'PlanetsLocationFetcher/1.0 (Educational/Research)'}
        )
        
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.load(response)
            
            if not data:
                return None
            
            result = data[0]
            
            # Check if we have geometry
            if 'geojson' not in result:
                return None
            
            geojson = result['geojson']
            geom_type = geojson.get('type', '')
            
            # Extract coordinates based on geometry type
            if geom_type == 'Polygon':
                # Polygon has one outer ring
                coordinates = geojson['coordinates'][0]
            elif geom_type == 'MultiPolygon':
                # MultiPolygon - use the largest polygon
                largest = max(geojson['coordinates'], key=lambda p: len(p[0]))
                coordinates = largest[0]
            elif geom_type == 'Point':
                # If only a point, create a small bounding box
                lon, lat = geojson['coordinates']
                offset = 0.05  # about 5km
                coordinates = [
                    [lon - offset, lat - offset],
                    [lon + offset, lat - offset],
                    [lon + offset, lat + offset],
                    [lon - offset, lat + offset],
                ]
            else:
                return None
            
            return coordinates
            
    except urllib.error.HTTPError as e:
        print(f"    HTTP Error {e.code}: {e.reason}")
        return None
    except urllib.error.URLError as e:
        print(f"    URL Error: {e.reason}")
        return None
    except Exception as e:
        print(f"    Error: {e}")
        return None


def process_cities(output_file, max_vertices=50, states_filter=None, delay=1.0, dry_run=False):
    """
    Fetch city boundaries and create CSV.
    
    Args:
        output_file: Path to output CSV file
        max_vertices: Maximum vertices per city polygon
        states_filter: List of state names to process (None = all)
        delay: Delay between API requests in seconds
        dry_run: If True, don't make API calls, just show what would be done
    """
    
    # Filter states if specified
    states_to_process = MAJOR_CITIES.keys()
    if states_filter:
        states_to_process = [s for s in states_to_process if s in states_filter]
        if not states_to_process:
            print(f"Error: No matching states found. Available: {', '.join(MAJOR_CITIES.keys())}")
            return 1
    
    # Count total cities
    total_cities = sum(len(MAJOR_CITIES[state]) for state in states_to_process)
    
    print(f"Processing {total_cities} cities across {len(states_to_process)} states...")
    print(f"Rate limit delay: {delay} seconds between requests")
    print(f"Estimated time: {total_cities * delay / 60:.1f} minutes")
    print()
    
    if dry_run:
        print("DRY RUN - No API calls will be made")
        print()
    
    # Open output file for writing
    if not dry_run:
        f = open(output_file, 'w')
        
        # Write CSV header
        f.write("# US Cities Location Table\n")
        f.write("# Format: location_id\tname\tpolygon_vertices\n")
        f.write("# location_id format: cityname-stateabbr (e.g., 'pasadena-ca')\n")
        f.write("# polygon_vertices format: lon1,lat1;lon2,lat2;lon3,lat3;...\n")
        f.write(f"# Vertices represent city boundaries (simplified to up to {max_vertices} vertices)\n")
        f.write("# Vertices are in counterclockwise order\n")
        f.write("# Longitude: negative for West, positive for East\n")
        f.write("# Latitude: negative for South, positive for North\n")
        f.write("# Data source: OpenStreetMap via Nominatim API\n")
        f.write("location_id\tname\tpolygon_vertices\n")
    
    success_count = 0
    failed_count = 0
    
    # Process each state
    for state in sorted(states_to_process):
        state_abbr = STATE_ABBREVIATIONS[state]
        cities = MAJOR_CITIES[state]
        
        print(f"\n{state} ({state_abbr}):")
        
        for city in cities:
            location_id = location_id_from_city_state(city, state_abbr)
            display_name = f"{city}, {state}"
            
            if dry_run:
                print(f"  Would fetch: {display_name} -> {location_id}")
                continue
            
            print(f"  Fetching: {display_name}...", end=' ', flush=True)
            
            # Fetch boundary
            coordinates = fetch_city_boundary(city, state, delay=delay)
            
            if coordinates is None:
                print("✗ Failed")
                failed_count += 1
                continue
            
            # Simplify if too many vertices
            if len(coordinates) > max_vertices:
                coordinates = simplify_polygon(coordinates, target_vertices=max_vertices)
            
            # Format vertices
            vertices_str = format_polygon_vertices(coordinates)
            
            # Write the row
            f.write(f"{location_id}\t{display_name}\t{vertices_str}\n")
            f.flush()  # Flush after each city in case of interruption
            
            print(f"✓ ({len(coordinates)} vertices)")
            success_count += 1
    
    if not dry_run:
        f.close()
    
    print()
    print("="*70)
    if dry_run:
        print(f"DRY RUN complete. Would process {total_cities} cities.")
    else:
        print(f"Processing complete!")
        print(f"  Success: {success_count} cities")
        print(f"  Failed:  {failed_count} cities")
        print(f"  Total:   {success_count + failed_count} cities")
        print()
        print(f"Output written to: {output_file}")
    
    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate US cities location table with polygon boundaries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--output',
        default='us_cities.csv',
        help="Output CSV file path (default: us_cities.csv)"
    )
    
    parser.add_argument(
        '--max-vertices',
        type=int,
        default=50,
        help="Maximum vertices per polygon (default: 50)"
    )
    
    parser.add_argument(
        '--states',
        nargs='+',
        help="Only process cities in specified states (e.g., California Texas)"
    )
    
    parser.add_argument(
        '--delay',
        type=float,
        default=1.0,
        help="Delay between API requests in seconds (default: 1.0)"
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Show what would be fetched without making API requests"
    )
    
    args = parser.parse_args()
    
    # Validate delay
    if args.delay < 0.5:
        print("Warning: Delay should be at least 0.5 seconds to respect API limits.")
        print("         Setting delay to 0.5 seconds.")
        args.delay = 0.5
    
    return process_cities(
        output_file=args.output,
        max_vertices=args.max_vertices,
        states_filter=args.states,
        delay=args.delay,
        dry_run=args.dry_run
    )


if __name__ == '__main__':
    sys.exit(main())
