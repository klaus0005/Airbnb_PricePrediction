# Data Directory

This directory contains the Airbnb data used for price prediction analysis.

## Structure

```
data/
├── raw/                    # Original data files
│   ├── listings.gz         # Main listings data (3.1MB)
│   ├── calendar.gz         # Availability calendar (5.9MB) 
│   ├── reviews.gz          # Reviews data (22MB)
│   ├── neighbourhoodsJSON.geojson  # Geographic boundaries (53KB)
│   └── neighbourhoodsCSV.csv       # Neighborhood info (511B)
├── processed/              # Generated/processed data
│   ├── listings_clean.csv  # Cleaned listings data
│   ├── calendar_clean.csv  # Cleaned calendar data
│   └── features/           # Feature engineering outputs
└── README.md              # This file
```

## Data Sources

The raw data files are from the [Inside Airbnb](http://insideairbnb.com/) project, which provides publicly available Airbnb data for various cities.

### File Descriptions

- **listings.gz**: Contains property listings with details like price, location, amenities, host info
- **calendar.gz**: Contains availability and pricing data for each listing
- **reviews.gz**: Contains review data (not used in current analysis)
- **neighbourhoodsJSON.geojson**: Geographic boundaries for neighborhoods
- **neighbourhoodsCSV.csv**: Neighborhood names and metadata

## Getting the Data

If you don't have the raw data files, you can download them from:
- Inside Airbnb: http://insideairbnb.com/get-the-data.html
- Select Munich, Germany and download the files

## Data Processing

The processed files are generated automatically by running the notebooks in order:
1. `01_data_exploration.ipynb` - Initial data loading and exploration
2. `02_feature_engineering.ipynb` - Basic feature engineering
3. `04_categorical_encoding.ipynb` - Categorical feature encoding
4. `05_spatial_features.ipynb` - Spatial feature generation

## Notes

- Raw data files (.gz) are not included in Git due to size
- Processed files are generated automatically and not included in Git
- Only small metadata files (neighbourhoods) are included in Git 