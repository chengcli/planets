#! /bin/bash


# Fetch white sand weather data from google drive
#
# era5_by_pressure_modules_2025_Jan_01_A.pt (89K)
URL="https://drive.google.com/uc?id=1HGe34OVEGIxE6VTf891ImKCbLAj0SHWm"

echo "Downloading from Google Drive..."
gdown "$URL"
