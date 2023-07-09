#!/bin/bash

# Download the file using curl
curl -o golve-files.zip https://storage.googleapis.com/tfy-public/downloads/golve-files.zip

# Create the destination directory if it doesn't exist
mkdir -p notebooks/downloads/

# Unzip the downloaded file
unzip golve-files.zip -d notebooks/downloads/

# Remove the downloaded zip file
rm golve-files.zip
