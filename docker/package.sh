#!/bin/bash

# Clone a fresh repository anonymously
git clone https://github.com/D3-AI/Orion.git

# Put the orion-jupyter image inside it
mv orion-jupyter.tar Orion

# Download the demo data inside the new repository
cd Orion
wget https://d3-ai-orion.s3.amazonaws.com/B-1.csv

mkdir -p notebooks/data
cd notebooks/data
wget https://d3-ai-orion.s3.amazonaws.com/S-1.csv
wget https://d3-ai-orion.s3.amazonaws.com/S-2.csv
wget https://d3-ai-orion.s3.amazonaws.com/E-1.csv
wget https://d3-ai-orion.s3.amazonaws.com/P-1.csv
cd ../..

mkdir -p orion/data
cp notebooks/data/* orion/data
cd orion/data
wget https://d3-ai-orion.s3.amazonaws.com/anomalies.csv

cd ../../../

# Make the zip file
ZIP_FILE=Orion-$(date +%Y-%m-%d).zip
zip -r $ZIP_FILE Orion
rm -rf Orion
