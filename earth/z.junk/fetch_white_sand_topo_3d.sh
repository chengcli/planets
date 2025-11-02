#! /bin/bash

# Fetch white sand topography data from google drive
#
# topo_24kmx12km_32.7719_32.9881_-106.5098_-106.3802_adjusted.pt (57 M)
#URL="https://drive.google.com/uc?id=1GXpDXq9oMa-nZVQaCcBBqaePgwntcAcj"
#
URL="https://drive.google.com/uc?id=1n-0od5ytAzW5M0_4rC6kMBn3aPv8NsEm"

echo "Downloading from Google Drive..."
gdown "$URL"
