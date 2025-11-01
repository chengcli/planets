# ECMWF Weather Data API

This directory contains an API for fetching and processing ECMWF ERA5 weather data from the Climate Data Store (CDS).

## Overview

The ECMWF Weather API provides a Python interface to:
- Query and download ERA5 reanalysis weather data
- Specify geographical bounds (latitude/longitude)
- Select atmospheric variables (temperature, wind, humidity, etc.)
- Define time windows and pressure levels
- Process and load the downloaded data

## Installation

### Requirements

Install the required Python packages:

```bash
pip install -r requirements.txt
```

The API requires:
- `cdsapi` - Climate Data Store API client
- `xarray` - For loading and processing NetCDF data
- `netCDF4` - NetCDF file format support
- `numpy` - Numerical operations (optional, for data processing)

### CDS API Key Setup

To use this API, you need a free CDS API key:

1. **Register**: Create an account at [https://cds.climate.copernicus.eu/](https://cds.climate.copernicus.eu/)

2. **Following instructions**: Follow the instructions to set up your API key at [https://cds.climate.copernicus.eu/how-to-api](https://cds.climate.copernicus.eu/how-to-api)

3. **Configure Authentication** (choose one method):

   **Option A: Environment Variable**
   ```bash
   export CDSAPI_KEY="your-uid:your-api-key"
   ```

   **Option B: Configuration File**
   
   Create `~/.cdsapirc` with:
   ```
   url: https://cds.climate.copernicus.eu/api
   key: your-uid:your-api-key
   ```

## Quick Start

### ECMWF/CDS Licenses

ECMWF/CDS requires users to agree to their terms of use for each data product. If some
API requests fail, an error message will prompt you to log in to the CDS website and accept the license.
Look for the URL in the error message to find the relevant license page and accept the terms.

### Basic Usage

```python
from ecmwf_weather_api import ECMWFWeatherAPI

# Initialize the API (uses credentials from environment or ~/.cdsapirc)
api = ECMWFWeatherAPI()

# Fetch weather data
output_file = api.fetch_weather_data(
    latmin=32.0,       # Minimum latitude
    latmax=33.5,       # Maximum latitude
    lonmin=-106.8,     # Minimum longitude
    lonmax=-105.8,     # Maximum longitude
    start_date="2024-01-01",
    end_date="2024-01-02",
    variables=["temperature", "u_component_of_wind", "v_component_of_wind"],
    pressure_levels=[1000, 925, 850, 700, 500]  # hPa
)

print(f"Data saved to: {output_file}")

# Load the downloaded data
data = api.load_data(output_file)
print(f"Variables: {data['variables'].keys()}")
print(f"Coordinates: {data['coordinates'].keys()}")
```

### Alternative: Explicit API Key

```python
# Provide API key directly
api = ECMWFWeatherAPI(api_key="your-uid:your-api-key")
```

### Monitor Download Progress

You can go to the CDS webportal to monitor the status of your data requests:

[https://cds.climate.copernicus.eu/requests](https://cds.climate.copernicus.eu/requests)

## API Reference

### ECMWFWeatherAPI Class

#### Initialization

```python
api = ECMWFWeatherAPI(api_key=None, api_url=None)
```

**Parameters:**
- `api_key` (str, optional): CDS API key. If not provided, reads from `CDSAPI_KEY` environment variable or `~/.cdsapirc`
- `api_url` (str, optional): CDS API URL. Default: `https://cds.climate.copernicus.eu/api/v2`

#### fetch_weather_data()

Fetch ECMWF ERA5 weather data from CDS.

```python
output_file = api.fetch_weather_data(
    latmin, latmax, lonmin, lonmax,
    start_date, end_date,
    variables,
    pressure_levels=None,
    output_file=None,
    times=None,
    product_type='reanalysis',
    format='netcdf'
)
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `latmin` | float | Yes | Minimum latitude (-90 to 90) |
| `latmax` | float | Yes | Maximum latitude (-90 to 90) |
| `lonmin` | float | Yes | Minimum longitude (-180 to 180) |
| `lonmax` | float | Yes | Maximum longitude (-180 to 180) |
| `start_date` | str | Yes | Start date in format 'YYYY-MM-DD' |
| `end_date` | str | Yes | End date in format 'YYYY-MM-DD' |
| `variables` | list[str] | Yes | List of atmospheric variables to fetch |
| `pressure_levels` | list[int] | No | Pressure levels in hPa. Default: [1000, 925, 850, 700, 500, 300, 200] |
| `output_file` | str | No | Output file path. If None, creates temporary file |
| `times` | list[str] | No | Times in format 'HH:MM'. Default: ['00:00', '06:00', '12:00', '18:00'] |
| `product_type` | str | No | 'reanalysis' or 'ensemble_members'. Default: 'reanalysis' |
| `format` | str | No | 'netcdf' or 'grib'. Default: 'netcdf' |

**Returns:** Path to downloaded file (str)

**Raises:**
- `ValueError`: If input parameters are invalid
- `RuntimeError`: If data retrieval fails

#### load_data()

Load downloaded weather data from file.

```python
data = api.load_data(file_path)
```

**Parameters:**
- `file_path` (str): Path to NetCDF or GRIB file

**Returns:** Dictionary containing:
```python
{
    'variables': {
        'temperature': np.ndarray,  # (time, level, lat, lon)
        'u_component_of_wind': np.ndarray,
        # ... other variables
    },
    'coordinates': {
        'time': np.ndarray,
        'level': np.ndarray,  # pressure levels
        'latitude': np.ndarray,
        'longitude': np.ndarray
    },
    'metadata': dict  # Dataset attributes
}
```

## Available Variables

### Common Variable Names

The API accepts both ECMWF names and common aliases:

| Alias | ECMWF Name | Description |
|-------|------------|-------------|
| `temperature`, `temp` | `temperature` | Air temperature (K) |
| `u`, `u_wind` | `u_component_of_wind` | U-component of wind (m/s) |
| `v`, `v_wind` | `v_component_of_wind` | V-component of wind (m/s) |
| `humidity`, `relative_humidity` | `relative_humidity` | Relative humidity (%) |
| `specific_humidity` | `specific_humidity` | Specific humidity (kg/kg) |
| `geopotential` | `geopotential` | Geopotential (m²/s²) |

For a complete list of available variables, see:
[ERA5 Pressure Levels Documentation](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels)

### Standard Pressure Levels (hPa)

Available pressure levels: 1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000

## Examples

### Example 1: Basic Weather Data

```python
from ecmwf_weather_api import ECMWFWeatherAPI

api = ECMWFWeatherAPI()

# Fetch data for White Sands, New Mexico
data_file = api.fetch_weather_data(
    latmin=32.0, latmax=33.5,
    lonmin=-106.8, lonmax=-105.8,
    start_date="2024-01-01",
    end_date="2024-01-02",
    variables=["temperature", "u_wind", "v_wind"],
    pressure_levels=[1000, 850, 500]
)

# Load and inspect
data = api.load_data(data_file)
print(f"Temperature shape: {data['variables']['temperature'].shape}")
```

### Example 2: Specific Times

```python
# Fetch data only at specific times of day
data_file = api.fetch_weather_data(
    latmin=30.0, latmax=35.0,
    lonmin=-110.0, lonmax=-105.0,
    start_date="2024-01-01",
    end_date="2024-01-01",
    variables=["temperature"],
    pressure_levels=[850],
    times=["00:00", "12:00"]  # Only midnight and noon UTC
)
```

### Example 3: High-Resolution Vertical Profile

```python
# Fetch detailed vertical profile
data_file = api.fetch_weather_data(
    latmin=32.0, latmax=33.0,
    lonmin=-106.0, lonmax=-105.0,
    start_date="2024-01-01",
    end_date="2024-01-01",
    variables=["temperature", "geopotential"],
    pressure_levels=[1000, 975, 950, 925, 900, 850, 800, 750, 700, 
                     650, 600, 550, 500, 450, 400, 300, 250, 200]
)
```

### Example 4: Multiple Variables

```python
# Fetch comprehensive atmospheric state
data_file = api.fetch_weather_data(
    latmin=30.0, latmax=40.0,
    lonmin=-110.0, lonmax=-100.0,
    start_date="2024-01-01",
    end_date="2024-01-01",
    variables=[
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "geopotential",
        "relative_humidity",
        "specific_humidity"
    ],
    pressure_levels=[1000, 850, 700, 500, 300]
)
```

## Complete Example Script

See `example_ecmwf_usage.py` for a complete working example with multiple use cases.

Run it with:
```bash
python example_ecmwf_usage.py
```

## Testing

Run the test suite:

```bash
python test_ecmwf_weather_api.py
```

The tests include:
- API initialization with various credential configurations
- Input validation (bounds, dates, pressure levels)
- Variable name normalization
- Mock data fetching (no actual API calls)
- Data loading functionality

## Error Handling

The API includes comprehensive error handling:

```python
try:
    data_file = api.fetch_weather_data(...)
except ValueError as e:
    # Invalid input parameters
    print(f"Invalid input: {e}")
except RuntimeError as e:
    # API request failed
    print(f"Request failed: {e}")
except ImportError as e:
    # Missing required packages
    print(f"Missing dependency: {e}")
```

## Notes

### Download Times

- Data retrieval from CDS can take several minutes depending on:
  - Size of the requested region
  - Number of variables and pressure levels
  - Time range
  - CDS server load
  
- For large requests, consider:
  - Breaking into smaller time windows
  - Fetching fewer variables per request
  - Using coarser pressure level spacing

### Data Format

- Default output format is NetCDF4 (`.nc`)
- Also supports GRIB format (set `format='grib'`)
- NetCDF is recommended for easier data processing with xarray/numpy

### Coordinate System

- Latitudes: -90 (South Pole) to +90 (North Pole)
- Longitudes: -180 (West) to +180 (East)
- Pressure levels: Surface (1000 hPa) to stratosphere (1 hPa)
- Times: UTC time zone

## Integration with Existing Code

This API is designed to work with the existing weather data processing pipeline in this repository:

```python
from ecmwf_weather_api import ECMWFWeatherAPI
from get_us_weather import interpolate_to_grid
import torch

# 1. Fetch raw ERA5 data
api = ECMWFWeatherAPI()
data_file = api.fetch_weather_data(...)

# 2. Load the data
raw_data = api.load_data(data_file)

# 3. Process with existing pipeline
# (Convert to format expected by interpolate_to_grid)
# ... additional processing code ...

# 4. Use with simulation
# ... integration with white_sand_crm.py or other simulations ...
```

## License

This API interfaces with ECMWF's Copernicus Climate Data Store. Users must comply with the [CDS Terms of Use](https://cds.climate.copernicus.eu/api/v2/terms/static/licence-to-use-copernicus-products.pdf).

## References

- [ERA5 Documentation](https://confluence.ecmwf.int/display/CKB/ERA5)
- [CDS API Documentation](https://cds.climate.copernicus.eu/api-how-to)
- [ERA5 Pressure Levels Dataset](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels)

## Support

For issues with:
- **This API**: Open an issue in this repository
- **CDS Service**: Contact ECMWF support at [https://support.ecmwf.int/](https://support.ecmwf.int/)
- **ERA5 Data**: See [ERA5 documentation](https://confluence.ecmwf.int/display/CKB/ERA5)
