# Ann Arbor Weather Prediction Pipeline

This directory contains the configuration for the Ann Arbor, Michigan weather data curation and processing pipeline.

## Overview

The Ann Arbor location uses the unified Earth location system to download and process ERA5 reanalysis data from ECMWF for atmospheric modeling and simulation. The pipeline covers:

- **Test Area**: Ann Arbor, Michigan (University of Michigan campus)
- **Center Coordinates**: 42.3°N, -83.7°W
- **Domain Size**: 125 km × 125 km horizontal, 15 km vertical
- **Time Window**: November 1, 2025
- **Vertical Extent**: Surface to 15 km altitude
- **Grid Resolution**: 150 (vertical) × 200 (N-S) × 200 (E-W) interior cells

**Note on Data Availability**: ERA5 data is typically released with a 5-day delay. If November 2025 data is not yet available, you can modify the dates in `ann_arbor.yaml` to use a different time period while keeping the same spatial configuration.

## Quick Start

### Prerequisites

1. **ECMWF CDS Account**: Register at [https://cds.climate.copernicus.eu/](https://cds.climate.copernicus.eu/)

2. **API Key Configuration**: Set up your credentials following [https://cds.climate.copernicus.eu/how-to-api](https://cds.climate.copernicus.eu/how-to-api)

   Option A: Environment variable
   ```bash
   export CDSAPI_KEY="your-uid:your-api-key"
   ```

   Option B: Configuration file `~/.cdsapirc`
   ```
   url: https://cds.climate.copernicus.eu/api
   key: your-uid:your-api-key
   ```

3. **Python Dependencies**: Install required packages
   ```bash
   pip install -r ../ecmwf/requirements.txt
   ```

### Download and Process Data

Run the unified pipeline from the earth directory:

```bash
cd ..
python download_location_data.py ann-arbor
```

This will automatically execute all 4 steps:
1. Download ERA5 data (Step 1)
2. Calculate air density (Step 2)
3. Regrid to Cartesian coordinates (Step 3)
4. Compute hydrostatic pressure (Step 4)

The script includes:
- **Automatic step detection**: Finds output directory from Step 1
- **Progress checking**: Waits for files from each step before proceeding
- **Timeout handling**: Aborts with error if steps take too long (default: 1 hour per step)
- **Partial execution**: Can stop after any step with `--stop-after`

Advanced options:

```bash
# Run only first 2 steps
python download_location_data.py ann-arbor --stop-after 2

# Use custom timeout (2 hours per step)
python download_location_data.py ann-arbor --timeout 7200

# Download to specific directory
python download_location_data.py ann-arbor --output-base ./data
```

### Generate Custom Configuration

To create a custom configuration file with different parameters:

```bash
cd ..
python generate_config.py ann-arbor --start-date 2025-11-15 --end-date 2025-11-17 --output custom_ann_arbor.yaml
python download_location_data.py ann-arbor --config custom_ann_arbor.yaml
```

## Pipeline Steps

The data curation pipeline consists of 4 steps that are now automated:

### Step 1: Fetch ERA5 Data

Downloads ERA5 reanalysis data from ECMWF Climate Data Store.

**Manual option**:
```bash
python ../ecmwf/fetch_era5_pipeline.py ann_arbor.yaml
```

**Output**: Directory containing:
- `era5_hourly_dynamics_20251101.nc`
- `era5_hourly_densities_20251101.nc`

**Variables downloaded**:
- Dynamics: temperature, wind components, vertical velocity, geopotential, etc.
- Densities: humidity, cloud water content, etc.

### Step 2: Calculate Air Density

Computes total air density from dynamics and density variables.

**Manual option**:
```bash
python ../ecmwf/calculate_density.py \
    --input-dir <output_dir> \
    --output-dir <output_dir>
```

**Output**: 
- `era5_density_20251101.nc`

**Variables**: Total density (ρ), dry air density (ρ_d), water vapor density (ρ_v), cloud density (ρ_c)

### Step 3: Regrid to Cartesian Coordinates

Transforms pressure-level data to height-based Cartesian grid suitable for finite-volume methods.

**Manual option**:
```bash
python ../ecmwf/regrid_era5_to_cartesian.py \
    ann_arbor.yaml <output_dir> \
    --output regridded_ann_arbor.nc
```

**Output**: 
- `regridded_ann_arbor.nc` (Cartesian grid with ghost zones)

### Step 4: Compute Hydrostatic Pressure

Ensures hydrostatic balance in the regridded data.

**Manual option**:
```bash
python ../ecmwf/compute_hydrostatic_pressure.py \
    ann_arbor.yaml regridded_ann_arbor.nc
```

**Output**: Augmented NetCDF file with balanced pressure field

## Configuration File

The `ann_arbor.yaml` file defines:

- **Geometry**: Cartesian domain with 150×200×200 interior cells plus 3 ghost cells on each side
- **Bounds**: 15 km vertical, 125 km × 125 km horizontal
- **Center**: 42.3°N, -83.7°W (University of Michigan campus)
- **Time**: November 1, 2025
- **Grid**: High-resolution atmospheric grid for mesoscale modeling

To generate a custom configuration:
```bash
cd ..
python generate_config.py ann-arbor [options]
```

See `../README_UNIFIED_SYSTEM.md` for all options.

## Output Data Structure

After running the complete pipeline, you will have:

```
<output_dir>/
├── era5_hourly_dynamics_20251101.nc     # Raw dynamics data
├── era5_hourly_densities_20251101.nc    # Raw density variables
├── era5_density_20251101.nc             # Computed densities
└── regridded_ann_arbor.nc               # Final Cartesian grid data
```

## Ann Arbor Characteristics

### Geography
- **Location**: Southeastern Michigan, USA
- **Elevation**: 260-300 meters above sea level
- **Terrain**: Gently rolling hills, part of the Huron River watershed
- **Notable Features**: Great Lakes influence, urban-rural transition zone

### Climate
- **November Weather**: 
  - Transitional fall weather with increasing cold
  - Average temperatures: 2-10°C (35-50°F)
  - Increased cloudiness and precipitation
  - First snow events possible
  - Lake-effect influences from Lake Erie and Lake Huron
- **Surface Pressure**: ~980-990 hPa (due to elevation)
- **Boundary Layer**: Variable depth due to changing synoptic conditions

### Atmospheric Modeling Considerations
- Humid continental climate with lake influences
- Frequent synoptic-scale weather systems
- Lake-effect snow and rain events
- Urban heat island effects from Detroit metro area to the east
- Complex mesoscale interactions between lakes and land
- Frontal passages common in November

## Usage Examples

### Run Complete Pipeline
```bash
cd ..
# Run complete pipeline with all 4 steps
python download_location_data.py ann-arbor
```

### Partial Pipeline Execution
```bash
cd ..
# Run only Step 1 (download data)
python download_location_data.py ann-arbor --stop-after 1

# Run Steps 1-2 (download and calculate density)
python download_location_data.py ann-arbor --stop-after 2

# Run with custom timeout (2 hours per step)
python download_location_data.py ann-arbor --timeout 7200
```

### Custom Configuration
```bash
cd ..
# Generate custom configuration
python generate_config.py ann-arbor \
    --start-date 2025-11-15 \
    --end-date 2025-11-17 \
    --output custom_ann_arbor.yaml

# Use custom configuration
python download_location_data.py ann-arbor --config custom_ann_arbor.yaml
```

### Manual Pipeline Execution (Advanced)

If you need to run steps manually (e.g., for debugging):

```bash
# Step 1: Download
python ../ecmwf/fetch_era5_pipeline.py ann_arbor.yaml --output-base ./data

# Step 2: Calculate density (assuming output dir is 41.75N_42.85N_84.25W_83.15W)
python ../ecmwf/calculate_density.py \
    --input-dir ./data/41.75N_42.85N_84.25W_83.15W \
    --output-dir ./data/41.75N_42.85N_84.25W_83.15W

# Step 3: Regrid
python ../ecmwf/regrid_era5_to_cartesian.py \
    ann_arbor.yaml ./data/41.75N_42.85N_84.25W_83.15W \
    --output ./data/41.75N_42.85N_84.25W_83.15W/regridded_ann_arbor.nc

# Step 4: Compute pressure
python ../ecmwf/compute_hydrostatic_pressure.py \
    ann_arbor.yaml ./data/41.75N_42.85N_84.25W_83.15W/regridded_ann_arbor.nc
```

## Monitoring Downloads

Monitor your CDS data requests at:
[https://cds.climate.copernicus.eu/requests](https://cds.climate.copernicus.eu/requests)

Download times depend on:
- Region size
- Number of variables
- Time period
- CDS server load

## Troubleshooting

### API Key Issues
```
ERROR: CDS API credentials not found
```
**Solution**: Configure your API key as described in Prerequisites

### License Agreement Required
```
ERROR: You must accept the license agreement
```
**Solution**: Log in to CDS website and accept terms for ERA5 pressure level data

### Download Timeout
**Solution**: 
- Increase timeout with `--timeout` option
- Try again later when CDS load is lower

Example:
```bash
cd ..
python download_location_data.py ann-arbor --timeout 7200
```

### Missing Dependencies
```
ModuleNotFoundError: No module named 'cdsapi'
```
**Solution**: Install requirements
```bash
pip install -r ../ecmwf/requirements.txt
```

### Data Not Available
```
ERROR: Data not available for requested date
```
**Solution**: ERA5 data has a 5-day delay. Generate a new configuration with a different date:
```bash
cd ..
python generate_config.py ann-arbor --start-date 2025-10-01 --output ann_arbor.yaml
```

## Advanced Configuration

Use the configuration generator to customize:
```bash
cd ..
# Customize domain size
python generate_config.py ann-arbor --x2-extent 150000 --x3-extent 150000

# Change resolution
python generate_config.py ann-arbor --nx1 200 --nx2 300 --nx3 300

# Multiple days
python generate_config.py ann-arbor --start-date 2025-11-01 --end-date 2025-11-03
```

For all options, see:
```bash
cd ..
python generate_config.py --help
```

## Documentation

For detailed documentation, see:
- `../README_UNIFIED_SYSTEM.md` - Complete unified system guide
- `../ecmwf/README_ECMWF.md` - Complete ECMWF API documentation
- `../ecmwf/STEP1_README.md` - Data fetching details
- `../ecmwf/STEP2_README.md` - Density calculation
- `../ecmwf/STEP3_README.md` - Regridding
- `../ecmwf/STEP4_README.md` - Hydrostatic pressure

## Support

For issues with:
- **This pipeline**: Open an issue in the planets repository
- **ECMWF CDS API**: Contact ECMWF support at [https://support.ecmwf.int/](https://support.ecmwf.int/)
- **ERA5 data**: See [ERA5 documentation](https://confluence.ecmwf.int/display/CKB/ERA5)

## References

- University of Michigan: [https://umich.edu/](https://umich.edu/)
- Ann Arbor Climate Data: [https://www.weather.gov/dtx/](https://www.weather.gov/dtx/)
- ERA5 Reanalysis: [https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5)
- ECMWF CDS: [https://cds.climate.copernicus.eu/](https://cds.climate.copernicus.eu/)

## Test Case Details

This test case is designed to simulate weather in Ann Arbor on November 1, 2025. The configuration creates initial conditions suitable for:

- **Mesoscale weather modeling**: 125 km domain captures local features
- **Lake-effect studies**: Near Great Lakes for lake-effect precipitation
- **Fall weather systems**: November conditions include frontal passages
- **Urban-rural interactions**: Ann Arbor and Detroit metro area influences
- **Numerical weather prediction**: High-resolution grid for detailed forecasting

The domain size and resolution are appropriate for studying:
- Convective systems and precipitation
- Boundary layer dynamics
- Lake-land interactions
- Synoptic-scale weather patterns
- Local circulation features
