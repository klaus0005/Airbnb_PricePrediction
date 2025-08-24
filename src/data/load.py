# src/data/load.py
import sys
import os
from urllib.request import urlretrieve
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.config import RAW_DIR, RAW_URLS

# This function downloads a file from a given URL if it does not already exist locally.
# The file is saved in the specified directory with the same extension as in the URL.
def download_if_missing(name: str, url: str, dest_dir: str = RAW_DIR) -> str:
    """
    Download file from `url` into `dest_dir/name.ext` if not already present.
    Returns the local filepath.
    """
    os.makedirs(dest_dir, exist_ok=True)  # Ensure the destination directory exists
    # Name the file after the key, keep the original extension
    filename = os.path.join(dest_dir, f"{name}{os.path.splitext(url)[1]}")
    if not os.path.exists(filename):
        print(f"→ Downloading {name} from {url} …")
        urlretrieve(url, filename)  # Download the file
    else:
        print(f"→ Found existing file for {name}, skipping download.")
    return filename

# This function loads a CSV (optionally gzipped) into a pandas DataFrame.
def load_dataframe(path: str, **read_kwargs) -> pd.DataFrame:
    """
    Loads a CSV (optionally .gz) into a DataFrame.
    """
    # If the file is gzipped, tell pandas to decompress it
    if path.endswith(".gz"):
        return pd.read_csv(path, compression="gzip", low_memory=False, **read_kwargs)
    return pd.read_csv(path, low_memory=False, **read_kwargs)

# This function downloads all raw data files defined in RAW_URLS and returns a dict of name→filepath.
def download_all_raw() -> dict[str,str]:
    """
    Downloads all URLs from RAW_URLS and returns a dict name→filepath.
    """
    return {
        name: download_if_missing(name, url)
        for name, url in RAW_URLS.items()
    }

# This function combines downloading and loading all data into DataFrames.
# Returns a dict: { 'listings': df_listings, 'calendar': df_calendar, ... }
def load_all() -> dict[str,pd.DataFrame]:
    """
    Combines download and loading:
      { 'listings': df_listings, 'calendar': df_calendar, ... }
    """
    paths = download_all_raw()  # Download all files if missing
    return {
        name: load_dataframe(path)
        for name, path in paths.items()
    }

# Optional: individual loader functions for each data type
def load_listings() -> pd.DataFrame:
    """Load only the listings data as a DataFrame."""
    return load_dataframe(download_if_missing("listings", RAW_URLS["listings"]))

def load_calendar() -> pd.DataFrame:
    """Load only the calendar data as a DataFrame."""
    return load_dataframe(download_if_missing("calendar", RAW_URLS["calendar"]))

def load_reviews() -> pd.DataFrame:
    """Load only the reviews data as a DataFrame."""
    return load_dataframe(download_if_missing("reviews", RAW_URLS["reviews"]))