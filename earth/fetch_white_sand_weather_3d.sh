#! /bin/bash


# Fetch white sand weather data from google drive
#
# era5_by_pressure_modules_2025_Jan_01_A.pt (89K)
URL="https://drive.google.com/uc?id=1HJnAxqHT8fiPkTtEmat9lPWIjpVtvVcy"

echo "Downloading from Google Drive..."
gdown "$URL"
