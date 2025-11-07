# Unified Earth Location Configuration System

This directory contains a unified system for managing atmospheric simulation configurations across multiple Earth locations. The system replaces repetitive location-specific code with a flexible, data-driven approach.

## Overview

The unified system consists of:

1. **Location Table** (`locations.yaml`): Defines geographic locations with polygon bounds, center coordinates, and default settings
2. **Configuration Template** (`config_template.yaml`): Template YAML file with placeholders for location-specific values
3. **Configuration Generator** (`generate_config.py`): Generates location-specific YAML files from the template
4. **Unified Download Script** (`download_location_data.py`): Single script that works with any configured location

## Quick Start

### List Available Locations

```bash
python generate_config.py --list
```

This shows all configured locations with their geographic bounds, default settings, and descriptions.

### Generate a Configuration File

```bash
# Generate config for Ann Arbor with defaults
python generate_config.py ann-arbor

# Generate config for White Sands with custom time window
python generate_config.py white-sands --start-date 2025-10-15 --end-date 2025-10-16

# Generate config with custom grid resolution
python generate_config.py ann-arbor --nx2 300 --nx3 300 --output custom_config.yaml
```

### Download and Process Data

```bash
# Download Ann Arbor data (uses ann-arbor/ann_arbor.yaml)
python download_location_data.py ann-arbor

# Download White Sands data (uses white_sands/white_sands.yaml)
python download_location_data.py white-sands

# Use custom configuration file
python download_location_data.py ann-arbor --config my_custom_config.yaml

# Run only first 2 steps
python download_location_data.py white-sands --stop-after 2
```

## Location Table Format

The `locations.yaml` file defines available locations:

```yaml
locations:
  location-id:  # Unique identifier (letters, numbers, dashes, underscores)
    name: "Human-readable name"
    description: "Location description"
    
    # Geographic bounds (polygon vertices)
    polygon:
      - [lon1, lat1]  # Southwest corner
      - [lon2, lat2]  # Northwest corner
      - [lon3, lat3]  # Northeast corner
      - [lon4, lat4]  # Southeast corner
    
    # Domain center
    center:
      longitude: -83.7  # Negative for West
      latitude: 42.3    # Positive for North
    
    elevation: 280  # Meters above sea level
    
    # Default domain size (meters)
    default_domain:
      x2_extent: 125000.0  # North-south
      x3_extent: 125000.0  # East-west
      x1_max: 15000.0      # Vertical
    
    # Default grid resolution
    default_grid:
      nx1: 150  # Vertical cells
      nx2: 200  # North-south cells
      nx3: 200  # East-west cells
      nghost: 3 # Ghost cells per side
    
    # Default time settings
    default_time:
      start_date: "2025-11-01"
      end_date: "2025-11-01"
      tlim: 43200  # Seconds
```

## Configuration Generator Options

The `generate_config.py` script accepts the following options:

### Basic Options
- `location_id`: Required location identifier (e.g., `ann-arbor`)
- `--list`: List all available locations
- `--output PATH`: Output file path (default: `<location-id>.yaml`)

### Time Options
- `--start-date YYYY-MM-DD`: Simulation start date
- `--end-date YYYY-MM-DD`: Simulation end date
- `--tlim SECONDS`: Simulation time limit

### Grid Resolution Options
- `--nx1 INT`: Vertical cells
- `--nx2 INT`: North-south cells
- `--nx3 INT`: East-west cells
- `--nghost INT`: Ghost cells per side (default: 3)

### Domain Size Options
- `--x1-max METERS`: Vertical extent
- `--x2-extent METERS`: North-south extent
- `--x3-extent METERS`: East-west extent

### File Paths
- `--locations-file PATH`: Custom locations table (default: `locations.yaml`)
- `--template-file PATH`: Custom template file (default: `config_template.yaml`)

## Download Script Options

The `download_location_data.py` script accepts:

- `location_id`: Required location identifier
- `--config PATH`: Configuration file (default: searches location subdirectory)
- `--output-base PATH`: Base directory for output (default: current directory)
- `--stop-after N`: Stop after step N (1-4)
- `--timeout SECONDS`: Timeout per step (default: 3600)
- `--locations-file PATH`: Custom locations table

## Adding New Locations

To add a new location:

1. Edit `locations.yaml` and add a new entry under `locations:`
2. Define the polygon bounds (vertices in counterclockwise order)
3. Set center coordinates, elevation, and default parameters
4. Generate a config file: `python generate_config.py <new-location-id>`
5. Download data: `python download_location_data.py <new-location-id>`

### Example: Adding a New Location

```yaml
locations:
  my-location:
    name: "My Test Site"
    description: "Description of the test site"
    
    polygon:
      - [-100.0, 30.0]  # SW corner
      - [-100.0, 31.0]  # NW corner
      - [-99.0, 31.0]   # NE corner
      - [-99.0, 30.0]   # SE corner
    
    center:
      longitude: -99.5
      latitude: 30.5
    
    elevation: 500
    
    default_domain:
      x2_extent: 111000.0
      x3_extent: 111000.0
      x1_max: 15000.0
    
    default_grid:
      nx1: 150
      nx2: 200
      nx3: 200
      nghost: 3
    
    default_time:
      start_date: "2025-01-01"
      end_date: "2025-01-01"
      tlim: 43200
```

## Location Identifier Rules

Location identifiers must:
- Contain only letters, numbers, dashes (`-`), and underscores (`_`)
- Be unique across all locations
- Not contain spaces or special characters
- Be used consistently in file names and directory names

Valid examples:
- `ann-arbor`
- `white-sands`
- `test_site_123`
- `location-A1`

Invalid examples:
- `Ann Arbor` (contains space)
- `white.sands` (contains period)
- `site@location` (contains @)

## Simulated Domain vs. Geographic Bounds

- **Geographic bounds (polygon)**: The actual area of interest (can be any shape)
- **Simulated domain (rectangular)**: The computational grid (always rectangular)
- The simulated rectangular domain **must contain** the polygon bounds

For example, if your polygon defines an irregular test area, the rectangular simulation domain will encompass it with some buffer space.

## Pipeline Steps

The download script executes 4 steps:

1. **Fetch ERA5 Data**: Download from ECMWF Climate Data Store
2. **Calculate Air Density**: Compute density from dynamics and moisture
3. **Regrid to Cartesian**: Transform to height-based grid with ghost zones
4. **Compute Hydrostatic Pressure**: Ensure hydrostatic balance

Each step can be run independently or stopped at any point using `--stop-after`.

## Backward Compatibility

For existing workflows using location-specific scripts:

### Ann Arbor
```bash
# Old way (still works)
cd ann_arbor
python download_ann_arbor_data.py

# New way (recommended)
cd earth
python download_location_data.py ann-arbor --config ann_arbor/ann_arbor.yaml
```

### White Sands
```bash
# Old way (still works)
cd white_sands
python download_white_sands_data.py

# New way (recommended)
cd earth
python download_location_data.py white-sands --config white_sands/white_sands.yaml
```

Wrapper scripts are provided in location subdirectories for backward compatibility.

## Examples

### Generate Custom Configuration

```bash
# High-resolution Ann Arbor simulation
python generate_config.py ann-arbor \
    --nx1 200 --nx2 400 --nx3 400 \
    --output ann_arbor_highres.yaml

# Multi-day White Sands simulation
python generate_config.py white-sands \
    --start-date 2025-10-01 \
    --end-date 2025-10-05 \
    --tlim 432000 \
    --output white_sands_5day.yaml
```

### Download with Custom Settings

```bash
# Download with increased timeout (2 hours per step)
python download_location_data.py ann-arbor --timeout 7200

# Download to specific directory
python download_location_data.py white-sands --output-base ./my_data

# Run only data download step
python download_location_data.py ann-arbor --stop-after 1
```

## File Organization

```
earth/
├── locations.yaml                      # Location table
├── config_template.yaml                # Configuration template
├── generate_config.py                  # Config generator script
├── download_location_data.py           # Unified download script
├── ann_arbor/
│   ├── ann_arbor.yaml                 # Ann Arbor config (generated)
│   ├── download_ann_arbor_data.py     # Original (kept for compatibility)
│   ├── download_ann_arbor_data_new.py # Wrapper script
│   └── README.md
├── white_sands/
│   ├── white_sands.yaml               # White Sands config (generated)
│   ├── download_white_sands_data.py   # Original (kept for compatibility)
│   ├── download_white_sands_data_new.py # Wrapper script
│   └── README.md
└── ecmwf/                             # ECMWF pipeline scripts
```

## Benefits of Unified System

1. **Reduced Code Duplication**: Single download script instead of per-location copies
2. **Easy Location Addition**: Add new locations by editing YAML, no code changes needed
3. **Flexible Configuration**: Command-line overrides for any parameter
4. **Consistent Interface**: Same commands work for all locations
5. **Maintainability**: Updates to pipeline logic happen in one place
6. **Data-Driven**: Location properties defined in data files, not code

## Requirements

- Python 3.6+
- PyYAML
- ECMWF CDS API credentials
- See `ecmwf/requirements.txt` for complete list

## Support

For issues or questions:
- Check location-specific README files (`ann_arbor/README.md`, `white_sands/README.md`)
- See ECMWF pipeline documentation (`ecmwf/README_ECMWF.md`)
- Open an issue in the planets repository

## References

- ERA5 Reanalysis: https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5
- ECMWF CDS: https://cds.climate.copernicus.eu/
- Python YAML: https://pyyaml.org/
