# This file is a template for preprocessing functions. Add your data cleaning and transformation functions here.

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Impute missing values for numeric and categorical columns
def impute_missing_values(df: pd.DataFrame, num_cols=None, cat_cols=None) -> pd.DataFrame:
    """
    Fill missing values: numeric columns with median, categorical with mode.
    """
    df = df.copy()
    if num_cols is None:
        num_cols = df.select_dtypes(include=[np.number]).columns
    if cat_cols is None:
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in num_cols:
        if col in df.columns:
            median = df[col].median()
            df[col] = df[col].fillna(median)
    for col in cat_cols:
        if col in df.columns:
            mode = df[col].mode(dropna=True)
            if not mode.empty:
                df[col] = df[col].fillna(mode[0])
    return df

# Add derived features such as price per person and description length
def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add new columns: price_per_person, description_length, etc.
    """
    df = df.copy()
    # Price per person (if guests column exists and >0)
    if "price" in df.columns and "accommodates" in df.columns:
        df["price_per_person"] = df["price"] / df["accommodates"].replace(0, np.nan)
    # Length of description (if description column exists)
    if "description" in df.columns:
        df["description_length"] = df["description"].fillna("").apply(len)
    return df

# Normalize numeric features using StandardScaler or MinMaxScaler
def normalize_features(df: pd.DataFrame, cols=None, method="standard", exclude_cols=None) -> pd.DataFrame:
    """
    Normalize numeric columns using standardization or min-max scaling.
    
    Args:
        df: DataFrame to normalize
        cols: Columns to normalize (if None, all numeric columns)
        method: 'standard' or 'minmax'
        exclude_cols: Columns to exclude from normalization (e.g., ['id'])
    """
    df = df.copy()
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns
    
    # Exclude specified columns from normalization
    if exclude_cols:
        cols = [col for col in cols if col not in exclude_cols]
    
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("method must be 'standard' or 'minmax'")
    
    df[cols] = scaler.fit_transform(df[cols])
    return df
