#! /bin/bash

# Fetch white sand topography data from google drive
# Folder ULR = https://drive.google.com/drive/folders/1UarL-r8H1fGL-v0rP7Ze8ezCZEbCiU0-
URL="https://drive.google.com/uc?id=1GXpDXq9oMa-nZVQaCcBBqaePgwntcAcj"

echo "Downloading from Google Drive..."
gdown "$URL"
