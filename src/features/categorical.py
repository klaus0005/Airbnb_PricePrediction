"""
Categorical feature encoding for Airbnb price prediction.
This module provides different encoding strategies for categorical variables.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
import ast
import re

def encode_low_cardinality_categorical(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Encode low cardinality categorical variables using one-hot encoding.
    
    Args:
        df: DataFrame with categorical variables
        columns: List of columns to encode (if None, auto-detect low cardinality)
    
    Returns:
        DataFrame with encoded categorical variables
    """
    df_encoded = df.copy()
    
    if columns is None:
        # Auto-detect low cardinality categorical columns (≤20 unique values)
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        columns = []
        for col in cat_cols:
            if df[col].nunique() <= 20:
                columns.append(col)
    
    print(f"Encoding {len(columns)} low cardinality categorical variables...")
    
    for col in columns:
        if col in df.columns:
            # Handle missing values
            df_encoded[col] = df_encoded[col].fillna('Unknown')
            
            # Create dummy variables
            dummies = pd.get_dummies(df_encoded[col], prefix=col, dummy_na=False)
            
            # Drop the original column and add dummies
            df_encoded = df_encoded.drop(columns=[col])
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            
            print(f"  ✅ {col}: {dummies.shape[1]} dummy variables created")
    
    return df_encoded

def encode_high_cardinality_categorical(df: pd.DataFrame, columns: list = None, 
                                      method: str = 'label') -> pd.DataFrame:
    """
    Encode high cardinality categorical variables using label encoding or frequency encoding.
    
    Args:
        df: DataFrame with categorical variables
        columns: List of columns to encode (if None, auto-detect high cardinality)
        method: 'label' for label encoding, 'frequency' for frequency encoding
    
    Returns:
        DataFrame with encoded categorical variables
    """
    df_encoded = df.copy()
    
    if columns is None:
        # Auto-detect high cardinality categorical columns (>20 unique values)
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        columns = []
        for col in cat_cols:
            if df[col].nunique() > 20:
                columns.append(col)
    
    print(f"Encoding {len(columns)} high cardinality categorical variables using {method} encoding...")
    
    for col in columns:
        if col in df.columns:
            if method == 'label':
                # Label encoding
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].fillna('Unknown'))
                print(f"  ✅ {col}: label encoded")
                
            elif method == 'frequency':
                # Frequency encoding
                freq_map = df_encoded[col].value_counts(normalize=True)
                df_encoded[col] = df_encoded[col].map(freq_map).fillna(0)
                print(f"  ✅ {col}: frequency encoded")
    
    return df_encoded

def extract_amenities_features(df: pd.DataFrame, amenities_col: str = 'amenities') -> pd.DataFrame:
    """
    Extract features from amenities list.
    
    Args:
        df: DataFrame with amenities column
        amenities_col: Name of the amenities column
    
    Returns:
        DataFrame with extracted amenities features
    """
    df_features = df.copy()
    
    if amenities_col not in df.columns:
        print(f"⚠️  {amenities_col} column not found")
        return df_features
    
    print(f"Extracting features from {amenities_col}...")
    
    # Common amenities to look for
    common_amenities = [
        'Wifi', 'Kitchen', 'Air conditioning', 'Heating', 'Washer', 'Dryer',
        'Free parking', 'Pool', 'Gym', 'Breakfast', 'Pets allowed', 'Balcony',
        'Elevator', 'Dishwasher', 'Microwave', 'TV', 'Iron', 'Hair dryer',
        'Shampoo', 'Essentials', 'Cooking basics', 'Coffee maker', 'Refrigerator'
    ]
    
    # Check if amenities column is already numeric (count)
    if df_features[amenities_col].dtype in ['int64', 'float64']:
        print(f"  ⚠️  {amenities_col} is already numeric (count), skipping amenity extraction")
        # Rename to amenities_count if it's not already named that
        if amenities_col != 'amenities_count':
            df_features['amenities_count'] = df_features[amenities_col]
            df_features = df_features.drop(columns=[amenities_col])
        return df_features
    
    # Create binary features for each common amenity
    for amenity in common_amenities:
        feature_name = f'amenity_{amenity.lower().replace(" ", "_")}'
        df_features[feature_name] = df_features[amenities_col].str.contains(
            amenity, case=False, na=False
        ).astype(int)
    
    # Count total amenities
    def count_amenities(amenities_str):
        try:
            if pd.isna(amenities_str):
                return 0
            amenities_list = ast.literal_eval(amenities_str)
            return len(amenities_list)
        except:
            return 0
    
    df_features['amenities_count'] = df_features[amenities_col].apply(count_amenities)
    
    print(f"  ✅ Created {len(common_amenities)} amenity features + amenities_count")
    
    return df_features

def encode_neighborhood_features(df: pd.DataFrame, neighborhood_col: str = 'neighbourhood_cleansed') -> pd.DataFrame:
    """
    Encode neighborhood features using one-hot encoding.
    
    Args:
        df: DataFrame with neighborhood column
        neighborhood_col: Name of the neighborhood column
    
    Returns:
        DataFrame with encoded neighborhood features
    """
    df_encoded = df.copy()
    
    if neighborhood_col not in df.columns:
        print(f"⚠️  {neighborhood_col} column not found")
        return df_encoded
    
    print(f"Encoding {neighborhood_col}...")
    
    # Create dummy variables for neighborhoods
    dummies = pd.get_dummies(df_encoded[neighborhood_col], prefix='neighborhood')
    
    # Drop the original column and add dummies
    df_encoded = df_encoded.drop(columns=[neighborhood_col])
    df_encoded = pd.concat([df_encoded, dummies], axis=1)
    
    print(f"  ✅ Created {dummies.shape[1]} neighborhood dummy variables")
    
    return df_encoded

def encode_property_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode property type and room type features.
    
    Args:
        df: DataFrame with property_type and room_type columns
    
    Returns:
        DataFrame with encoded property features
    """
    df_encoded = df.copy()
    
    # Encode room_type
    if 'room_type' in df.columns:
        room_dummies = pd.get_dummies(df_encoded['room_type'], prefix='room_type')
        df_encoded = df_encoded.drop(columns=['room_type'])
        df_encoded = pd.concat([df_encoded, room_dummies], axis=1)
        print(f"  ✅ room_type: {room_dummies.shape[1]} dummy variables")
    
    # Encode property_type (top categories only to avoid too many features)
    if 'property_type' in df.columns:
        # Get top property types (keep top 10, group others as 'Other')
        top_properties = df_encoded['property_type'].value_counts().head(10).index
        df_encoded['property_type_grouped'] = df_encoded['property_type'].apply(
            lambda x: x if x in top_properties else 'Other'
        )
        
        property_dummies = pd.get_dummies(df_encoded['property_type_grouped'], prefix='property_type')
        df_encoded = df_encoded.drop(columns=['property_type', 'property_type_grouped'])
        df_encoded = pd.concat([df_encoded, property_dummies], axis=1)
        print(f"  ✅ property_type: {property_dummies.shape[1]} dummy variables (top 10 + Other)")
    
    return df_encoded

def encode_host_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode host-related categorical features.
    
    Args:
        df: DataFrame with host-related columns
    
    Returns:
        DataFrame with encoded host features
    """
    df_encoded = df.copy()
    
    # Binary features (already numeric, just ensure they're 0/1)
    binary_cols = ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified', 'instant_bookable']
    
    for col in binary_cols:
        if col in df.columns:
            # Handle both string and numeric binary values
            if df_encoded[col].dtype == 'object':
                # Convert string values to binary
                true_values = ['t', 'true', 'yes', '1', 1, True]
                df_encoded[col] = df_encoded[col].astype(str).str.lower().isin(true_values).astype(int)
            else:
                # Ensure numeric values are 0 or 1
                df_encoded[col] = (df_encoded[col] > 0).astype(int)
            print(f"  ✅ {col}: ensured binary values (0/1)")
    
    # Encode host_response_time (if not already encoded)
    if 'host_response_time' in df.columns and 'host_response_time_encoded' not in df.columns:
        response_time_map = {
            'within an hour': 1,
            'within a few hours': 2,
            'within a day': 3,
            'a few days or more': 4
        }
        df_encoded['host_response_time_encoded'] = df_encoded['host_response_time'].map(response_time_map)
        df_encoded = df_encoded.drop(columns=['host_response_time'])
        print(f"  ✅ host_response_time: ordinal encoded")
    elif 'host_response_time_encoded' in df.columns:
        print(f"  ✅ host_response_time: already encoded")
    
    # Encode host_verifications (if not already encoded)
    if 'host_verifications' in df.columns and 'host_verifications_count' not in df.columns:
        def count_verifications(verifications_str):
            try:
                if pd.isna(verifications_str):
                    return 0
                verifications_list = ast.literal_eval(verifications_str)
                return len(verifications_list)
            except:
                return 0
        
        df_encoded['host_verifications_count'] = df_encoded['host_verifications'].apply(count_verifications)
        df_encoded = df_encoded.drop(columns=['host_verifications'])
        print(f"  ✅ host_verifications: converted to count")
    elif 'host_verifications_count' in df.columns:
        print(f"  ✅ host_verifications: already encoded")
    
    # Encode host_response_rate and host_acceptance_rate (ensure they're numeric and in [0,1])
    for col in ['host_response_rate', 'host_acceptance_rate']:
        if col in df.columns:
            # Convert to numeric, handling string values like "100%" or "95%"
            df_encoded[col] = pd.to_numeric(
                df_encoded[col].astype(str).str.replace('%', '').str.replace(',', ''),
                errors='coerce'
            )
            # Ensure values are in [0,1] range (assuming percentages are 0-100)
            df_encoded[col] = (df_encoded[col] / 100).clip(0, 1)
            print(f"  ✅ {col}: ensured values in [0,1] range")
    
    return df_encoded

def encode_all_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode all categorical features using appropriate strategies.
    
    Args:
        df: DataFrame with categorical variables
    
    Returns:
        DataFrame with all categorical variables encoded
    """
    print("🚀 Starting categorical feature encoding...")
    
    df_encoded = df.copy()
    
    # 1. Encode host features
    print("\n1. Encoding host features...")
    df_encoded = encode_host_features(df_encoded)
    
    # 2. Encode property features
    print("\n2. Encoding property features...")
    df_encoded = encode_property_features(df_encoded)
    
    # 3. Encode neighborhood features
    print("\n3. Encoding neighborhood features...")
    df_encoded = encode_neighborhood_features(df_encoded)
    
    # 4. Extract amenities features
    print("\n4. Extracting amenities features...")
    df_encoded = extract_amenities_features(df_encoded)
    
    # 5. Encode low cardinality categorical variables
    print("\n5. Encoding low cardinality categorical variables...")
    df_encoded = encode_low_cardinality_categorical(df_encoded)
    
    # 6. Encode high cardinality categorical variables
    print("\n6. Encoding high cardinality categorical variables...")
    df_encoded = encode_high_cardinality_categorical(df_encoded, method='label')
    
    # 7. Drop remaining high cardinality text columns that are not useful
    text_cols_to_drop = [
        'listing_url', 'name', 'description', 'picture_url', 'host_url', 
        'host_name', 'host_thumbnail_url', 'host_picture_url', 'bathrooms_text',
        'first_review', 'last_review', 'last_scraped', 'source', 'calendar_last_scraped'
    ]
    
    cols_to_drop = [col for col in text_cols_to_drop if col in df_encoded.columns]
    df_encoded = df_encoded.drop(columns=cols_to_drop)
    print(f"\n7. Dropped {len(cols_to_drop)} high cardinality text columns")
    
    print(f"\n✅ Categorical encoding complete!")
    print(f"   Original shape: {df.shape}")
    print(f"   Encoded shape: {df_encoded.shape}")
    print(f"   New features: {df_encoded.shape[1] - df.shape[1]}")
    
    return df_encoded 