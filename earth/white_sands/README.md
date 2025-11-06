# White Sands Weather Prediction Pipeline

This directory contains the weather data curation and processing pipeline for the White Sands Missile Range test area in New Mexico.

## Overview

The White Sands weather prediction pipeline downloads and processes ERA5 reanalysis data from ECMWF for atmospheric modeling and simulation. The pipeline covers:

- **Test Area**: Longitude 106.7°W - 106.2°W, Latitude 32.6°N - 33.6°N
- **With Buffer**: Longitude 107.2°W - 105.7°W, Latitude 32.1°N - 34.1°N (0.5° buffer on each side)
- **Time Window**: October 1-2, 2025 (as specified in requirements)
- **Vertical Extent**: Surface to 15 km altitude
- **Domain Size**: ~223 km (N-S) × ~140 km (E-W)

**Note on Data Availability**: ERA5 data is typically released with a 5-day delay. If October 2025 data is not yet available, you can modify the dates in `white_sands.yaml` to use a different time period while keeping the same spatial configuration.

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

### Automated Pipeline (Recommended)

Run the complete automated pipeline that executes all 4 steps:

```bash
python download_white_sands_data.py
```

This will automatically:
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
python download_white_sands_data.py --stop-after 2

# Use custom timeout (2 hours per step)
python download_white_sands_data.py --timeout 7200

# Download to specific directory
python download_white_sands_data.py --output-base ./data
```

## Pipeline Steps

The data curation pipeline consists of 4 steps that are now automated:

### Step 1: Fetch ERA5 Data

Downloads ERA5 reanalysis data from ECMWF Climate Data Store.

**Automated**: Yes (via `download_white_sands_data.py`)

**Manual option**:
```bash
python ../ecmwf/fetch_era5_pipeline.py white_sands.yaml
```

**Output**: Directory containing:
- `era5_hourly_dynamics_YYYYMMDD.nc` (one file per day)
- `era5_hourly_densities_YYYYMMDD.nc` (one file per day)

**Variables downloaded**:
- Dynamics: temperature, wind components, vertical velocity, geopotential, etc.
- Densities: humidity, cloud water content, etc.

### Step 2: Calculate Air Density

Computes total air density from dynamics and density variables.

**Automated**: Yes (via `download_white_sands_data.py`)

**Manual option**:
```bash
python ../ecmwf/calculate_density.py \
    --input-dir <output_dir> \
    --output-dir <output_dir>
```

**Output**: 
- `era5_density_YYYYMMDD.nc` (one file per day)

**Variables**: Total density (ρ), dry air density (ρ_d), water vapor density (ρ_v), cloud density (ρ_c)

### Step 3: Regrid to Cartesian Coordinates

Transforms pressure-level data to height-based Cartesian grid suitable for finite-volume methods.

**Automated**: Yes (via `download_white_sands_data.py`)

**Manual option**:
```bash
python ../ecmwf/regrid_era5_to_cartesian.py \
    white_sands.yaml <output_dir> \
    --output regridded_white_sands.nc
```

**Output**: 
- `regridded_white_sands.nc` (Cartesian grid with ghost zones)

### Step 4: Compute Hydrostatic Pressure

Ensures hydrostatic balance in the regridded data.

**Automated**: Yes (via `download_white_sands_data.py`)

**Manual option**:
```bash
python ../ecmwf/compute_hydrostatic_pressure.py \
    white_sands.yaml regridded_white_sands.nc
```

**Output**: Augmented NetCDF file with balanced pressure field

## Configuration File

The `white_sands.yaml` file defines:

- **Geometry**: Cartesian domain with 150×400×300 interior cells plus 3 ghost cells on each side
- **Bounds**: 15 km vertical, ~223 km north-south, ~140 km east-west
- **Center**: 33.1°N, 106.45°W
- **Time**: October 1-2, 2025
- **Grid**: High-resolution atmospheric grid for mesoscale modeling

Edit this file to customize:
- Domain size and resolution
- Time period
- Geographic center
- Grid cell counts

## Output Data Structure

After running the complete pipeline, you will have:

```
<output_dir>/
├── era5_hourly_dynamics_20251001.nc     # Raw dynamics data
├── era5_hourly_dynamics_20251002.nc
├── era5_hourly_densities_20251001.nc    # Raw density variables
├── era5_hourly_densities_20251002.nc
├── era5_density_20251001.nc             # Computed densities
├── era5_density_20251002.nc
└── regridded_white_sands.nc             # Final Cartesian grid data
```

## White Sands Characteristics

### Geography
- **Location**: South-central New Mexico, USA
- **Elevation**: 1,200-1,300 meters above sea level
- **Terrain**: Chihuahuan Desert, flat to gently rolling
- **Notable Feature**: World's largest gypsum dune field

### Climate
- **October Weather**: 
  - Generally dry conditions
  - Mild to warm temperatures (15-25°C typical)
  - Occasional monsoon moisture remnants
  - Light winds, strong heating during day
- **Surface Pressure**: ~870-880 hPa (due to elevation)
- **Boundary Layer**: Well-developed daytime convective boundary layer

### Atmospheric Modeling Considerations
- Dry desert environment with minimal cloud cover
- Strong surface heating and thermal circulation
- Occasional dust events
- Clear-sky radiation important
- Mesoscale mountain-valley flows from surrounding ranges

## Usage Examples

### Automated Pipeline (Recommended)
```bash
# Run complete pipeline with all 4 steps
python download_white_sands_data.py
```

### Partial Pipeline Execution
```bash
# Run only Step 1 (download data)
python download_white_sands_data.py --stop-after 1

# Run Steps 1-2 (download and calculate density)
python download_white_sands_data.py --stop-after 2

# Run with custom timeout (2 hours per step)
python download_white_sands_data.py --timeout 7200
```

### Custom Output Directory
```bash
python download_white_sands_data.py \
    --config white_sands.yaml \
    --output-base ./october_2025_data
```

### Manual Pipeline Execution (Advanced)

If you need to run steps manually (e.g., for debugging):

```bash
# Step 1: Download
python ../ecmwf/fetch_era5_pipeline.py white_sands.yaml --output-base ./data

# Step 2: Calculate density (assuming output dir is 32.10N_34.10N_107.20W_105.70W)
python ../ecmwf/calculate_density.py \
    --input-dir ./data/32.10N_34.10N_107.20W_105.70W \
    --output-dir ./data/32.10N_34.10N_107.20W_105.70W

# Step 3: Regrid
python ../ecmwf/regrid_era5_to_cartesian.py \
    white_sands.yaml ./data/32.10N_34.10N_107.20W_105.70W \
    --output ./data/32.10N_34.10N_107.20W_105.70W/regridded_white_sands.nc

# Step 4: Compute pressure
python ../ecmwf/compute_hydrostatic_pressure.py \
    white_sands.yaml ./data/32.10N_34.10N_107.20W_105.70W/regridded_white_sands.nc
```

### Process Existing Data
If you already have downloaded ERA5 files:

```bash
# Skip fetching, run only processing steps
cd /path/to/existing/data

python ../ecmwf/calculate_density.py \
    --input-dir . \
    --output-dir .

python ../ecmwf/regrid_era5_to_cartesian.py \
    ../white_sands/white_sands.yaml . \
    --output regridded.nc

python ../ecmwf/compute_hydrostatic_pressure.py \
    ../white_sands/white_sands.yaml regridded.nc
```

## Monitoring Downloads

Monitor your CDS data requests at:
[https://cds.climate.copernicus.eu/requests](https://cds.climate.copernicus.eu/requests)

Large requests may take several minutes to hours depending on:
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
- Reduce time period or spatial extent
- Try again later when CDS load is lower

### Missing Dependencies
```
ModuleNotFoundError: No module named 'cdsapi'
```
**Solution**: Install requirements
```bash
pip install -r ../ecmwf/requirements.txt
```

## Advanced Configuration

### Customize Domain
Edit `white_sands.yaml` to change:
- Domain size: Modify `bounds` (x1max, x2max, x3max)
- Resolution: Modify `cells` (nx1, nx2, nx3)
- Center location: Modify `center_latitude`, `center_longitude`
- Time period: Modify `start-date`, `end-date`

## Documentation

For detailed documentation on the ECMWF data pipeline, see:
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

- White Sands Missile Range: [https://www.wsmr.army.mil/](https://www.wsmr.army.mil/)
- ERA5 Reanalysis: [https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5)
- ECMWF CDS: [https://cds.climate.copernicus.eu/](https://cds.climate.copernicus.eu/)
