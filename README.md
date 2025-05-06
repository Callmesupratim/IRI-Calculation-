# IRI-Calculation

This repository contains tools and scripts for computing the **International Roughness Index (IRI)** from deflection data using a quarter-car model. It's designed to support road surface evaluation from field data, and includes modules for data cleaning, filtering, visualization, and classification per IRC standards.

## Features

- Deflection data pre-processing (filtering, clipping, interpolation)
- Quarter-car simulation model for IRI estimation
- IRI classification as per Indian Roads Congress (IRC) standards
- Geospatial mapping of results using Folium
- Optional estimation of Mean Profile Depth (MPD) and Skid Resistance Index (SRI)

## Requirements

- Python 3.8+
- NumPy, Pandas, SciPy
- Matplotlib, Folium
- tqdm, seaborn

Install dependencies with:

```bash
pip install -r requirements.txt
