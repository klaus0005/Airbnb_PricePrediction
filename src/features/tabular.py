import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from typing import List, Dict, Optional, Tuple
import warnings

def create_interaction_features(df: pd.DataFrame, feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
    """
    Create interaction features between pairs of numerical features.
    
    Args:
        df: DataFrame with features
        feature_pairs: List of tuples with feature names to create interactions
        
    Returns:
        DataFrame with interaction features added
    """
    result_df = df.copy()
    
    for feat1, feat2 in feature_pairs:
        if feat1 in result_df.columns and feat2 in result_df.columns:
            # Multiplication interaction
            result_df[f'{feat1}_x_{feat2}'] = result_df[feat1] * result_df[feat2]
            
            # Division interaction (avoid division by zero)
            result_df[f'{feat1}_div_{feat2}'] = result_df[feat1] / result_df[feat2].replace(0, 1)
            
            # Difference
            result_df[f'{feat1}_minus_{feat2}'] = result_df[feat1] - result_df[feat2]
    
    return result_df

def create_polynomial_features(df: pd.DataFrame, features: List[str], degree: int = 2) -> pd.DataFrame:
    """
    Create polynomial features for specified numerical columns.
    
    Args:
        df: DataFrame with features
        features: List of feature names to create polynomials for
        degree: Maximum polynomial degree (default 2)
        
    Returns:
        DataFrame with polynomial features added
    """
    result_df = df.copy()
    
    for feature in features:
        if feature in result_df.columns:
            for d in range(2, degree + 1):
                result_df[f'{feature}_pow_{d}'] = result_df[feature] ** d
    
    return result_df

def create_ratio_features(df: pd.DataFrame, numerator_features: List[str], 
                         denominator_features: List[str]) -> pd.DataFrame:
    """
    Create ratio features between numerical features.
    
    Args:
        df: DataFrame with features
        numerator_features: List of features to use as numerators
        denominator_features: List of features to use as denominators
        
    Returns:
        DataFrame with ratio features added
    """
    result_df = df.copy()
    
    for num_feat in numerator_features:
        for den_feat in denominator_features:
            if num_feat in result_df.columns and den_feat in result_df.columns:
                # Avoid division by zero
                result_df[f'{num_feat}_div_{den_feat}'] = (
                    result_df[num_feat] / result_df[den_feat].replace(0, 1)
                )
    
    return result_df

def create_aggregation_features(df: pd.DataFrame, group_by: str, 
                               agg_features: List[str]) -> pd.DataFrame:
    """
    Create aggregation features based on grouping.
    
    Args:
        df: DataFrame with features
        group_by: Column name to group by
        agg_features: List of features to aggregate
        
    Returns:
        DataFrame with aggregation features added
    """
    result_df = df.copy()
    
    # Calculate aggregations
    agg_stats = result_df.groupby(group_by)[agg_features].agg([
        'mean', 'std', 'min', 'max', 'count'
    ]).reset_index()
    
    # Flatten column names
    agg_stats.columns = [group_by] + [
        f'{col[0]}_{col[1]}' for col in agg_stats.columns[1:]
    ]
    
    # Merge back
    result_df = result_df.merge(agg_stats, on=group_by, how='left')
    
    return result_df

def scale_numerical_features(df: pd.DataFrame, features: List[str], 
                           scaler_type: str = 'standard') -> Tuple[pd.DataFrame, object]:
    """
    Scale numerical features using specified scaler.
    
    Args:
        df: DataFrame with features
        features: List of numerical features to scale
        scaler_type: Type of scaler ('standard', 'minmax', 'robust')
        
    Returns:
        Tuple of (scaled DataFrame, fitted scaler)
    """
    result_df = df.copy()
    
    # Select scaler
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    # Scale features
    available_features = [f for f in features if f in result_df.columns]
    if available_features:
        result_df[available_features] = scaler.fit_transform(result_df[available_features])
    
    return result_df, scaler

def select_best_features(df: pd.DataFrame, target: str, k: int = 10) -> List[str]:
    """
    Select the best k features using f_regression.
    
    Args:
        df: DataFrame with features
        target: Target variable name
        k: Number of features to select
        
    Returns:
        List of selected feature names
    """
    # Prepare data
    X = df.drop(columns=[target])
    y = df[target]
    
    # Remove non-numerical columns
    numerical_cols = X.select_dtypes(include=[np.number]).columns
    X_numerical = X[numerical_cols]
    
    # Select best features
    selector = SelectKBest(score_func=f_regression, k=min(k, len(numerical_cols)))
    selector.fit(X_numerical, y)
    
    # Get selected feature names
    selected_features = X_numerical.columns[selector.get_support()].tolist()
    
    return selected_features

def handle_outliers(df: pd.DataFrame, features: List[str], 
                   method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
    """
    Handle outliers in numerical features.
    
    Args:
        df: DataFrame with features
        features: List of features to handle outliers for
        method: Method to detect outliers ('iqr', 'zscore')
        threshold: Threshold for outlier detection
        
    Returns:
        DataFrame with outliers handled
    """
    result_df = df.copy()
    
    for feature in features:
        if feature not in result_df.columns:
            continue
            
        if method == 'iqr':
            Q1 = result_df[feature].quantile(0.25)
            Q3 = result_df[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # Cap outliers
            result_df[feature] = result_df[feature].clip(lower=lower_bound, upper=upper_bound)
            
        elif method == 'zscore':
            mean_val = result_df[feature].mean()
            std_val = result_df[feature].std()
            lower_bound = mean_val - threshold * std_val
            upper_bound = mean_val + threshold * std_val
            
            # Cap outliers
            result_df[feature] = result_df[feature].clip(lower=lower_bound, upper=upper_bound)
    
    return result_df

def create_time_features(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """
    Create time-based features from date columns.
    
    Args:
        df: DataFrame with date column
        date_column: Name of the date column
        
    Returns:
        DataFrame with time features added
    """
    result_df = df.copy()
    
    if date_column not in result_df.columns:
        return result_df
    
    # Convert to datetime
    result_df[date_column] = pd.to_datetime(result_df[date_column], errors='coerce')
    
    # Extract time features
    result_df[f'{date_column}_year'] = result_df[date_column].dt.year
    result_df[f'{date_column}_month'] = result_df[date_column].dt.month
    result_df[f'{date_column}_day'] = result_df[date_column].dt.day
    result_df[f'{date_column}_dayofweek'] = result_df[date_column].dt.dayofweek
    result_df[f'{date_column}_quarter'] = result_df[date_column].dt.quarter
    result_df[f'{date_column}_is_weekend'] = result_df[date_column].dt.dayofweek.isin([5, 6]).astype(int)
    
    return result_df
