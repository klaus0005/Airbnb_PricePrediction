# src/config.py

# This file contains configuration variables for data paths and download URLs.
# RAW_DIR: Directory where raw data files are stored.
# RAW_URLS: Dictionary mapping data types to their download URLs.
import os
RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/raw"))

RAW_URLS = {
    "listings": "https://data.insideairbnb.com/germany/bv/munich/2025-03-19/data/listings.csv.gz",
    "calendar": "https://data.insideairbnb.com/germany/bv/munich/2025-03-19/data/calendar.csv.gz",
    "reviews":  "https://data.insideairbnb.com/germany/bv/munich/2025-03-19/data/reviews.csv.gz",
    "neighbourhoodsCSV":  "https://data.insideairbnb.com/germany/bv/munich/2025-03-19/visualisations/neighbourhoods.csv",
    "neighbourhoodsJSON":  "https://data.insideairbnb.com/germany/bv/munich/2025-03-19/visualisations/neighbourhoods.geojson"
}
