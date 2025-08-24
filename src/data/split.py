"""
Data splitting utilities for Airbnb price prediction.

This module provides functions for splitting data into training, validation, and test sets
with various strategies including random splits, stratified splits, and time-based splits.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import logging


def random_split(X: Union[pd.DataFrame, np.ndarray], 
                y: Union[pd.Series, np.ndarray],
                test_size: float = 0.2,
                val_size: float = 0.2,
                random_state: int = 42) -> Tuple:
    """
    Split data randomly into train, validation, and test sets.
    
    Args:
        X: Features
        y: Targets
        test_size: Proportion of data for test set
        val_size: Proportion of remaining data for validation set
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Second split: separate validation set from remaining data
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def stratified_split(X: Union[pd.DataFrame, np.ndarray], 
                    y: Union[pd.Series, np.ndarray],
                    test_size: float = 0.2,
                    val_size: float = 0.2,
                    n_bins: int = 10,
                    random_state: int = 42) -> Tuple:
    """
    Split data using stratified sampling based on price bins.
    
    Args:
        X: Features
        y: Targets (prices)
        test_size: Proportion of data for test set
        val_size: Proportion of remaining data for validation set
        n_bins: Number of bins for stratifying
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Create price bins for stratification
    y_binned = pd.cut(y, bins=n_bins, labels=False)
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test, y_binned_temp, y_binned_test = train_test_split(
        X, y, y_binned, test_size=test_size, 
        stratify=y_binned, random_state=random_state
    )
    
    # Second split: separate validation set from remaining data
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, 
        stratify=y_binned_temp, random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def time_based_split(df: pd.DataFrame,
                    date_column: str,
                    test_size: float = 0.2,
                    val_size: float = 0.2) -> Tuple:
    """
    Split data based on time order (assuming newer data is better for testing).
    
    Args:
        df: DataFrame with features and target
        date_column: Name of the date column
        test_size: Proportion of data for test set
        val_size: Proportion of remaining data for validation set
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Sort by date
    df_sorted = df.sort_values(date_column)
    
    # Calculate split indices
    n_samples = len(df_sorted)
    test_idx = int(n_samples * (1 - test_size))
    val_idx = int(test_idx * (1 - val_size))
    
    # Split the data
    train_df = df_sorted.iloc[:val_idx]
    val_df = df_sorted.iloc[val_idx:test_idx]
    test_df = df_sorted.iloc[test_idx:]
    
    # Separate features and target
    X_train = train_df.drop(['price'], axis=1, errors='ignore')
    y_train = train_df['price']
    
    X_val = val_df.drop(['price'], axis=1, errors='ignore')
    y_val = val_df['price']
    
    X_test = test_df.drop(['price'], axis=1, errors='ignore')
    y_test = test_df['price']
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def k_fold_split(X: Union[pd.DataFrame, np.ndarray], 
                y: Union[pd.Series, np.ndarray],
                n_splits: int = 5,
                random_state: int = 42) -> list:
    """
    Create k-fold cross-validation splits.
    
    Args:
        X: Features
        y: Targets
        n_splits: Number of folds
        random_state: Random seed for reproducibility
        
    Returns:
        List of tuples (X_train, X_val, y_train, y_val) for each fold
    """
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = []
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx] if isinstance(X, pd.DataFrame) else X[train_idx], \
                        X.iloc[val_idx] if isinstance(X, pd.DataFrame) else X[val_idx]
        y_train, y_val = y.iloc[train_idx] if isinstance(y, pd.Series) else y[train_idx], \
                        y.iloc[val_idx] if isinstance(y, pd.Series) else y[val_idx]
        
        splits.append((X_train, X_val, y_train, y_val))
    
    return splits


def split_with_validation(X: Union[pd.DataFrame, np.ndarray], 
                         y: Union[pd.Series, np.ndarray],
                         method: str = 'random',
                         test_size: float = 0.2,
                         val_size: float = 0.2,
                         random_state: int = 42,
                         **kwargs) -> dict:
    """
    Split data with validation set using specified method.
    
    Args:
        X: Features
        y: Targets
        method: Split method ('random', 'stratified', 'time')
        test_size: Proportion of data for test set
        val_size: Proportion of remaining data for validation set
        random_state: Random seed for reproducibility
        **kwargs: Additional arguments for specific methods
        
    Returns:
        Dictionary with split data and metadata
    """
    logger = logging.getLogger(__name__)
    
    if method == 'random':
        X_train, X_val, X_test, y_train, y_val, y_test = random_split(
            X, y, test_size, val_size, random_state
        )
    elif method == 'stratified':
        n_bins = kwargs.get('n_bins', 10)
        X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(
            X, y, test_size, val_size, n_bins, random_state
        )
    elif method == 'time':
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Time-based split requires DataFrame with date column")
        date_column = kwargs.get('date_column', 'date')
        X_train, X_val, X_test, y_train, y_val, y_test = time_based_split(
            X, y, date_column, test_size, val_size
        )
    else:
        raise ValueError(f"Unknown split method: {method}")
    
    # Calculate split statistics
    total_samples = len(X)
    train_samples = len(X_train)
    val_samples = len(X_val)
    test_samples = len(X_test)
    
    logger.info(f"Data split completed:")
    logger.info(f"  Total samples: {total_samples}")
    logger.info(f"  Train samples: {train_samples} ({train_samples/total_samples:.1%})")
    logger.info(f"  Validation samples: {val_samples} ({val_samples/total_samples:.1%})")
    logger.info(f"  Test samples: {test_samples} ({test_samples/total_samples:.1%})")
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'metadata': {
            'method': method,
            'test_size': test_size,
            'val_size': val_size,
            'random_state': random_state,
            'total_samples': total_samples,
            'train_samples': train_samples,
            'val_samples': val_samples,
            'test_samples': test_samples
        }
    }


def create_cv_splits(X: Union[pd.DataFrame, np.ndarray], 
                    y: Union[pd.Series, np.ndarray],
                    cv_method: str = 'kfold',
                    n_splits: int = 5,
                    random_state: int = 42) -> list:
    """
    Create cross-validation splits.
    
    Args:
        X: Features
        y: Targets
        cv_method: Cross-validation method ('kfold', 'stratified_kfold')
        n_splits: Number of splits
        random_state: Random seed for reproducibility
        
    Returns:
        List of split indices or data
    """
    if cv_method == 'kfold':
        return k_fold_split(X, y, n_splits, random_state)
    elif cv_method == 'stratified_kfold':
        from sklearn.model_selection import StratifiedKFold
        
        # Create price bins for stratification
        y_binned = pd.cut(y, bins=10, labels=False)
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = []
        
        for train_idx, val_idx in skf.split(X, y_binned):
            X_train, X_val = X.iloc[train_idx] if isinstance(X, pd.DataFrame) else X[train_idx], \
                            X.iloc[val_idx] if isinstance(X, pd.DataFrame) else X[val_idx]
            y_train, y_val = y.iloc[train_idx] if isinstance(y, pd.Series) else y[train_idx], \
                            y.iloc[val_idx] if isinstance(y, pd.Series) else y[val_idx]
            
            splits.append((X_train, X_val, y_train, y_val))
        
        return splits
    else:
        raise ValueError(f"Unknown CV method: {cv_method}")


def get_split_summary(split_data: dict) -> pd.DataFrame:
    """
    Create a summary of the data split.
    
    Args:
        split_data: Dictionary returned by split_with_validation
        
    Returns:
        DataFrame with split summary
    """
    y_train = split_data['y_train']
    y_val = split_data['y_val']
    y_test = split_data['y_test']
    
    summary_data = []
    
    for split_name, y_data in [('Train', y_train), ('Validation', y_val), ('Test', y_test)]:
        summary_data.append({
            'Split': split_name,
            'Samples': len(y_data),
            'Mean': y_data.mean(),
            'Std': y_data.std(),
            'Min': y_data.min(),
            'Max': y_data.max(),
            'Q25': y_data.quantile(0.25),
            'Q50': y_data.quantile(0.50),
            'Q75': y_data.quantile(0.75)
        })
    
    return pd.DataFrame(summary_data)


def save_split_data(split_data: dict, output_dir: str, prefix: str = "split") -> None:
    """
    Save split data to files.
    
    Args:
        split_data: Dictionary returned by split_with_validation
        output_dir: Directory to save the data
        prefix: Prefix for file names
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save data
    split_data['X_train'].to_csv(os.path.join(output_dir, f"{prefix}_X_train.csv"), index=False)
    split_data['X_val'].to_csv(os.path.join(output_dir, f"{prefix}_X_val.csv"), index=False)
    split_data['X_test'].to_csv(os.path.join(output_dir, f"{prefix}_X_test.csv"), index=False)
    
    split_data['y_train'].to_csv(os.path.join(output_dir, f"{prefix}_y_train.csv"), index=False)
    split_data['y_val'].to_csv(os.path.join(output_dir, f"{prefix}_y_val.csv"), index=False)
    split_data['y_test'].to_csv(os.path.join(output_dir, f"{prefix}_y_test.csv"), index=False)
    
    # Save metadata
    import json
    with open(os.path.join(output_dir, f"{prefix}_metadata.json"), 'w') as f:
        json.dump(split_data['metadata'], f, indent=2, default=str)
    
    print(f"✅ Split data saved to {output_dir}")


def load_split_data(input_dir: str, prefix: str = "split") -> dict:
    """
    Load split data from files.
    
    Args:
        input_dir: Directory containing the split data
        prefix: Prefix for file names
        
    Returns:
        Dictionary with loaded split data
    """
    import os
    import json
    
    # Load data
    X_train = pd.read_csv(os.path.join(input_dir, f"{prefix}_X_train.csv"))
    X_val = pd.read_csv(os.path.join(input_dir, f"{prefix}_X_val.csv"))
    X_test = pd.read_csv(os.path.join(input_dir, f"{prefix}_X_test.csv"))
    
    y_train = pd.read_csv(os.path.join(input_dir, f"{prefix}_y_train.csv"))
    y_val = pd.read_csv(os.path.join(input_dir, f"{prefix}_y_val.csv"))
    y_test = pd.read_csv(os.path.join(input_dir, f"{prefix}_y_test.csv"))
    
    # Load metadata
    with open(os.path.join(input_dir, f"{prefix}_metadata.json"), 'r') as f:
        metadata = json.load(f)
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train.iloc[:, 0],  # Remove index column
        'y_val': y_val.iloc[:, 0],
        'y_test': y_test.iloc[:, 0],
        'metadata': metadata
    }
