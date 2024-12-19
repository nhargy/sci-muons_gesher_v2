#! /bin/bash


# CONFIG:
DATA_PATH = /home/hargy/Science/DataBox/


# Create local python virtual environment
python3 -m venv sci-venv

# Install dependencies from requirements.txt file
pip3 install -r requirements.txt

# Build local file structure
mkdir lcd plt out



