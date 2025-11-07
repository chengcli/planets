# US Zipcode Polygon Tools

This directory contains tools for generating and visualizing US zipcode boundary polygons.

## Scripts

### generate_us_zipcode_polygons.py

Generate or regenerate the `us_zipcode.csv` file with zipcode boundary polygons.

**Usage:**
```bash
# Generate with default settings (uses sample data for major cities)
python generate_us_zipcode_polygons.py

# Generate specific zipcodes only
python generate_us_zipcode_polygons.py --zipcodes 48104 90210 10001

# Customize output file and vertex count
python generate_us_zipcode_polygons.py --output custom_zipcode.csv --max-vertices 20

# Generate sample of random zipcodes (for testing)
python generate_us_zipcode_polygons.py --sample 100
```

**Options:**
- `--output FILE`: Output CSV file path (default: `us_zipcode.csv`)
- `--max-vertices N`: Maximum vertices per polygon (default: 30)
- `--zipcodes ZIP [ZIP ...]`: Specific zipcodes to include
- `--sample N`: Sample N random zipcodes for testing

**Data Source:**
Currently uses sample data for major US cities. For complete data, you can:
1. Download US Census TIGER/Line shapefiles
2. Use a geocoding API
3. Manually download GeoJSON from public repositories

**Note:** The script uses ZCTA (ZIP Code Tabulation Areas) which are generalized areal representations and may not exactly match postal ZIP code boundaries.

### plot_us_zipcode.py

Plot zipcode polygons on a map with coastal lines using various map projections (requires cartopy).

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

# Save to file instead of displaying
python plot_us_zipcode.py 48104 --output ann_arbor.png

# Plot without zipcode labels
python plot_us_zipcode.py 48104 90210 --no-labels
```

**Options:**
- `--zipcode-file FILE`: Path to CSV file with zipcode polygons (default: `us_zipcode.csv`)
- `--projection NAME`: Map projection to use (default: `PlateCarree`)
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

### plot_us_zipcode_simple.py

Plot zipcode polygons using simple matplotlib without map projections (no internet required).

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

# Save to file
python plot_us_zipcode_simple.py 48104 --output ann_arbor.png
```

**Options:**
- `--zipcode-file FILE`: Path to CSV file with zipcode polygons (default: `us_zipcode.csv`)
- `--output FILE`: Save figure to file instead of displaying
- `--no-labels`: Don't show zipcode labels
- `--figsize WIDTH HEIGHT`: Figure size in inches (default: 12 10)

## Examples

### Generate zipcode data for specific zipcodes
```bash
python generate_us_zipcode_polygons.py --zipcodes 48104 90210 10001 02101 60601
```

### Plot Ann Arbor zipcode on a Lambert Conformal projection (requires cartopy)
```bash
python plot_us_zipcode.py 48104 --projection LambertConformal --output ann_arbor.png
```

### Plot multiple zipcodes (simple version, no cartopy needed)
```bash
python plot_us_zipcode_simple.py 48104 90210 10001 --output three_zipcodes.png
```

### Plot Beverly Hills and save high-res image
```bash
python plot_us_zipcode_simple.py 90210 --output beverly_hills.png
```

### Compare East and West coast cities
```bash
python plot_us_zipcode_simple.py 10001 90210 33101 94102 --output coast_cities.png
```

## File Format

The generated `us_zipcode.csv` file follows this format:

```
zipcode	polygon_vertices
48104	-83.78,42.25;-83.78,42.31;-83.70,42.31;-83.70,42.25;-83.78,42.25
```

Where:
- **zipcode**: 5-digit ZIP code (zero-padded)
- **polygon_vertices**: Semicolon-separated lon,lat pairs representing the zipcode boundary

## Available Zipcodes

The default `us_zipcode.csv` includes sample data for major US cities:

- `48104` - Ann Arbor, MI
- `90210` - Beverly Hills, CA
- `10001` - New York, NY
- `60601` - Chicago, IL
- `02101` - Boston, MA
- `33101` - Miami, FL
- `94102` - San Francisco, CA
- `98101` - Seattle, WA
- `75201` - Dallas, TX
- `30301` - Atlanta, GA
- `80201` - Denver, CO
- `85001` - Phoenix, AZ
- `19101` - Philadelphia, PA
- `77001` - Houston, TX
- `20001` - Washington, DC
- `32801` - Orlando, FL
- `89101` - Las Vegas, NV
- `97201` - Portland, OR
- `55401` - Minneapolis, MN
- `63101` - St. Louis, MO

## Notes

- Zipcode boundaries are based on ZCTA (ZIP Code Tabulation Areas) from the US Census Bureau
- ZCTAs are generalized areal representations and may not exactly match postal ZIP code boundaries
- All longitude/latitude values follow standard conventions:
  - Longitude: negative for West, positive for East
  - Latitude: negative for South, positive for North
- For complete zipcode coverage, download the full TIGER/Line shapefiles from the US Census Bureau

## Integration with Other Scripts

The zipcode data format is designed to be compatible with the location database system:

**Format Comparison:**
- `locations.csv`: `location_id`, `name`, `polygon_vertices`
- `us_states.csv`: `location_id`, `name`, `polygon_vertices`
- `us_zipcode.csv`: `zipcode`, `polygon_vertices` (simplified - zipcode is self-descriptive)

**Note:** While the zipcode CSV uses a simplified format (without a separate name column), the polygon format is identical to other location databases. The zipcode visualization scripts (`plot_us_zipcode.py` and `plot_us_zipcode_simple.py`) handle this format directly.

**For custom integrations:** If you need to integrate zipcode data with `generate_config.py` or similar tools that expect a `name` field, you can either:
1. Use the zipcode visualization scripts provided (recommended)
2. Create a custom loader that adds the zipcode as both the ID and name
3. Extend the CSV to include a name column (e.g., "Zipcode 48104")
