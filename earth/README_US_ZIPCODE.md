# US Zipcode Polygon Query Tools

This directory contains tools for querying and visualizing US zipcode boundary polygons from the [State-zip-code-GeoJSON](https://github.com/chengcli/State-zip-code-GeoJSON) repository.

## Overview

Rather than maintaining a static database, these tools query zipcode polygon data on-demand from the State-zip-code-GeoJSON repository, which contains GeoJSON files with zipcode boundaries for all 50 US states plus DC.

## Scripts

### query_zipcode.py

Query zipcode polygon boundaries from the State-zip-code-GeoJSON database.

**Usage:**
```bash
# Query a single zipcode
python query_zipcode.py 48104

# Query multiple zipcodes
python query_zipcode.py 48104 90210 10001

# Provide state hint for faster lookup
python query_zipcode.py 48104 --state mi

# Simplify polygon to fewer vertices
python query_zipcode.py 48104 --max-vertices 50

# Output in different formats
python query_zipcode.py 48104 --format json
python query_zipcode.py 48104 --format wkt
```

**As a Python library:**
```python
from query_zipcode import get_zipcode_polygon

# Get polygon for Ann Arbor, MI
polygon = get_zipcode_polygon('48104', state_hint='mi')
if polygon:
    print(f"Found {len(polygon)} vertices")
    for lon, lat in polygon[:3]:
        print(f"  ({lon}, {lat})")
```

**Options:**
- `--state STATE`: 2-letter state abbreviation to narrow search
- `--max-vertices N`: Maximum vertices per polygon (simplifies if needed)
- `--format {json,csv,wkt}`: Output format (default: csv)

### plot_us_zipcode_simple.py

Plot zipcode polygons using simple matplotlib (no cartopy dependency).

**Requirements:**
```bash
pip install matplotlib
```

**Usage:**
```bash
# Plot a single zipcode
python plot_us_zipcode_simple.py 48104

# Plot multiple zipcodes
python plot_us_zipcode_simple.py 48104 90210 10001

# Provide state hint for faster querying
python plot_us_zipcode_simple.py 48104 --state mi

# Simplify polygons for faster rendering
python plot_us_zipcode_simple.py 48104 --max-vertices 100

# Save to file
python plot_us_zipcode_simple.py 48104 --output ann_arbor.png
```

**Options:**
- `--state STATE`: 2-letter state abbreviation to speed up queries
- `--max-vertices N`: Maximum vertices per polygon (simplifies if needed)
- `--output FILE`: Save figure to file instead of displaying
- `--no-labels`: Don't show zipcode labels
- `--figsize WIDTH HEIGHT`: Figure size in inches (default: 12 10)

### plot_us_zipcode.py

Plot zipcode polygons on a map with coastal lines using cartopy (requires internet for map data).

**Requirements:**
```bash
pip install matplotlib cartopy
```

**Note:** This script requires cartopy which downloads Natural Earth data for coastlines. If you don't have internet access or want a simpler alternative, use `plot_us_zipcode_simple.py` instead.

**Usage:**
```bash
# Plot a single zipcode
python plot_us_zipcode.py 48104

# Plot multiple zipcodes
python plot_us_zipcode.py 48104 90210 10001

# Use different projection
python plot_us_zipcode.py 48104 --projection LambertConformal

# Save to file
python plot_us_zipcode.py 48104 --output ann_arbor.png

# Provide state hint and simplify
python plot_us_zipcode.py 48104 --state mi --max-vertices 100
```

**Options:**
- `--state STATE`: 2-letter state abbreviation to speed up queries
- `--projection NAME`: Map projection to use (default: PlateCarree)
- `--max-vertices N`: Maximum vertices per polygon (simplifies if needed)
- `--output FILE`: Save figure to file instead of displaying
- `--no-labels`: Don't show zipcode labels
- `--figsize WIDTH HEIGHT`: Figure size in inches (default: 12 10)

**Available Projections:**
- `PlateCarree`: Simple lat-lon projection (default)
- `LambertConformal`: Lambert conformal conic
- `Mercator`: Mercator projection
- `Orthographic`: Orthographic (globe) projection
- `Robinson`: Robinson projection
- `AlbersEqualArea`: Albers equal-area conic

## Examples

### Query zipcode data
```bash
# Get Ann Arbor, MI zipcode polygon
python query_zipcode.py 48104 --state mi

# Get multiple California zipcodes
python query_zipcode.py 90210 94102 --state ca --format json

# Get simplified polygon (fewer vertices)
python query_zipcode.py 48104 --max-vertices 50
```

### Plot single zipcode (simple)
```bash
python plot_us_zipcode_simple.py 48104 --state mi --output ann_arbor.png
```

### Plot multiple zipcodes (simple)
```bash
python plot_us_zipcode_simple.py 48104 90210 10001 --output three_zipcodes.png
```

### Plot with cartopy and projection
```bash
python plot_us_zipcode.py 48104 --state mi --projection LambertConformal --output ann_arbor_map.png
```

### Compare multiple zipcodes on map
```bash
python plot_us_zipcode.py 48104 90210 33101 --projection AlbersEqualArea --output coast_comparison.png
```

## How It Works

The tools query the [State-zip-code-GeoJSON](https://github.com/chengcli/State-zip-code-GeoJSON) repository, which contains GeoJSON files for each US state with ZCTA (ZIP Code Tabulation Area) boundaries.

**Lookup Strategy:**
1. If a state hint is provided, search that state first
2. Infer likely states based on ZIP code prefix (first digit)
3. If not found, search all states (slower but comprehensive)

**Performance Tips:**
- Always provide `--state` hint when known for faster queries
- Use `--max-vertices` to simplify complex polygons for faster rendering
- The first query to a state downloads the GeoJSON file (~1-10MB per state)

## Data Source

**Repository:** https://github.com/chengcli/State-zip-code-GeoJSON

The repository contains GeoJSON files with zipcode boundaries derived from US Census TIGER/Line shapefiles. Each state's file includes:
- ZCTA5CE10: 5-digit ZIP code
- Polygon/MultiPolygon geometry with actual boundaries

**Note:** ZCTAs (ZIP Code Tabulation Areas) are generalized areal representations and may not exactly match postal ZIP code boundaries.

## State Abbreviations

Use 2-letter state abbreviations for the `--state` parameter:

- `al` - Alabama, `ak` - Alaska, `az` - Arizona, `ar` - Arkansas
- `ca` - California, `co` - Colorado, `ct` - Connecticut, `de` - Delaware
- `fl` - Florida, `ga` - Georgia, `hi` - Hawaii, `id` - Idaho
- `il` - Illinois, `in` - Indiana, `ia` - Iowa, `ks` - Kansas
- `ky` - Kentucky, `la` - Louisiana, `me` - Maine, `md` - Maryland
- `ma` - Massachusetts, `mi` - Michigan, `mn` - Minnesota, `ms` - Mississippi
- `mo` - Missouri, `mt` - Montana, `ne` - Nebraska, `nv` - Nevada
- `nh` - New Hampshire, `nj` - New Jersey, `nm` - New Mexico, `ny` - New York
- `nc` - North Carolina, `nd` - North Dakota, `oh` - Ohio, `ok` - Oklahoma
- `or` - Oregon, `pa` - Pennsylvania, `ri` - Rhode Island, `sc` - South Carolina
- `sd` - South Dakota, `tn` - Tennessee, `tx` - Texas, `ut` - Utah
- `vt` - Vermont, `va` - Virginia, `wa` - Washington, `wv` - West Virginia
- `wi` - Wisconsin, `wy` - Wyoming, `dc` - District of Columbia

## Integration

The `get_zipcode_polygon()` function can be imported and used in other scripts:

```python
from query_zipcode import get_zipcode_polygon

# Query a zipcode
polygon = get_zipcode_polygon('48104', state_hint='mi', max_vertices=100)

if polygon:
    # polygon is a list of (longitude, latitude) tuples
    for lon, lat in polygon:
        print(f"Vertex: ({lon:.6f}, {lat:.6f})")
```

The function returns `None` if the zipcode is not found.
