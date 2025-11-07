# Earth's weather and climate API

This directory contains an API for accessing and analyzing Earth's weather and climate data. The API provides several endpoints for retrieving current weather conditions, historical climate data, and forecasts.

## Location Databases

The Earth directory includes several location databases with polygon boundaries:

- **locations.csv**: Original location database (Ann Arbor, White Sands)
- **us_states.csv**: US state boundaries with actual polygon data (see `README_US_STATES.md`)
- **us_cities.csv**: Major US city boundaries with polygon data (see `README_US_CITIES.md`)

Each location is identified by a unique ID and includes polygon vertices for precise geographic boundaries.

## Installation

For accessing the ECMWF Climate Data Store (CDS), read the instructions at `ecmwf/README_ECMWF.md`.

## Regional Weather Prediction Pipelines

### White Sands Missile Range
Complete weather data pipeline for the White Sands test area in New Mexico (October 2025).
- **Location**: `white_sands/`
- **Documentation**: See `white_sands/README.md`
- **Configuration**: `white_sands/white_sands.yaml`
- **Quick start**: `cd white_sands && python download_white_sands_data.py`

### Ann Arbor, Michigan
Complete weather data pipeline for Ann Arbor, Michigan (November 2025).
- **Location**: `ann_arbor/`
- **Documentation**: See `ann_arbor/README.md`
- **Configuration**: `ann_arbor/ann_arbor.yaml`
- **Quick start**: `cd ann_arbor && python download_ann_arbor_data.py`

### ECMWF Data Curation
General-purpose ECMWF ERA5 data access and processing tools.
- **Location**: `ecmwf/`
- **Documentation**: See `ecmwf/README_ECMWF.md`
- **Complete 4-step pipeline**: Fetch, Calculate Density, Regrid, Compute Pressure

## Configuration files

The following configuration files might be created:
- `$HOME/.cdsapirc`: Configuration file for accessing CDS API. Follow instructions at [https://cds.climate.copernicus.eu/api-how-to](https://cds.climate.copernicus.eu/api-how-to) to set it up.
