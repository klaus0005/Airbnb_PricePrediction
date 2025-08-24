# Datei: scripts/run_preprocessing.py

import os, sys
# Add the project root to sys.path so that 'src' can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import pandas as pd
import numpy as np
from src.data.load import load_all
from src.data.preprocess import impute_missing_values, add_derived_features, normalize_features

# This function cleans the listings DataFrame by handling missing values, cleaning price, and more.
def clean_listings(df: pd.DataFrame) -> pd.DataFrame:
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # 1) Drop columns with more than 50% missing values
    thresh = len(df) * 0.5
    df = df.dropna(axis=1, thresh=thresh)
    
    # 2) Clean the price column (remove currency symbols, convert to float)
    df.loc[:, "price"] = (
        df["price"]
        .str.replace(r"[^\d\.]", "", regex=True)
        .astype(float)
    )
    
    # 3) Convert date columns to datetime
    if "last_review" in df.columns:
        df.loc[:, "last_review"] = pd.to_datetime(df["last_review"])
    
    # 4) Fill missing values in numeric columns with the median
    num_feats = ["beds", "bathrooms", "minimum_nights", "price"]
    for col in num_feats:
        if col in df.columns:
            median = df[col].median()
            df.loc[:, col] = df[col].fillna(median)
    
    # 5) Fill missing values in categorical columns with the most frequent value
    cat_feats = ["neighbourhood", "room_type", "property_type"]
    for col in cat_feats:
        if col in df.columns:
            mode = df[col].mode(dropna=True)
            if len(mode) > 0:
                df.loc[:, col] = df[col].fillna(mode[0])
    
    # 6) Remove duplicate listings based on the 'id' column
    df = df.drop_duplicates(subset="id")
    
    # Impute missing values (numeric/ categorical)
    df = impute_missing_values(df)

    # Add derived features (e.g. price_per_person, description_length)
    df = add_derived_features(df)

    # Normalize numeric features (exclude id and count columns to preserve interpretability)
    exclude_cols = [
        "id",  # preserve for joins
        "host_id",  # preserve for joins
        "host_listings_count",  # count variable
        "host_total_listings_count",  # count variable
        "number_of_reviews",  # count variable
        "number_of_reviews_ltm",  # count variable
        "number_of_reviews_l30d",  # count variable
        "number_of_reviews_ly",  # count variable
        "calculated_host_listings_count",  # count variable
        "calculated_host_listings_count_entire_homes",  # count variable
        "calculated_host_listings_count_private_rooms",  # count variable
        "calculated_host_listings_count_shared_rooms",  # count variable
    ]
    df = normalize_features(df, method="standard", exclude_cols=exclude_cols)

    return df

# This function cleans the calendar DataFrame by converting dates and mapping availability to 0/1.
def clean_calendar(df: pd.DataFrame) -> pd.DataFrame:
    # Convert the 'date' column to datetime
    df["date"] = pd.to_datetime(df["date"])
    # Create a boolean flag for availability (1 if available, 0 if not)
    df["available_flag"] = df["available"].map({"t":1, "f":0})
    return df

# Main function to run the preprocessing pipeline
def main():
    # Load raw data using the loader function
    data = load_all()
    df_listings = data["listings"]
    df_calendar = data["calendar"]
    df_reviews = data["reviews"]

    # Clean the listings and calendar data
    df_listings_clean = clean_listings(df_listings)
    df_calendar_clean = clean_calendar(df_calendar)
    # (Reviews can be cleaned later during feature engineering)

    # Save the cleaned data to the processed data folder
    os.makedirs("data/processed", exist_ok=True)
    df_listings_clean.to_csv("data/processed/listings_clean.csv", index=False)
    df_calendar_clean.to_csv("data/processed/calendar_clean.csv", index=False)

    print("✅ Preprocessing complete. Clean files in data/processed/")

if __name__ == "__main__":
    main()
