#!/usr/bin/env python3
"""
Training script for Airbnb price prediction models.

This script provides a command-line interface for training various models
and saving them for later use.
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from models.linear import create_all_models, compare_models
from models.multimodal import MultimodalModel
from models.nn import train_mlp_model
from data.split import split_with_validation, save_split_data
from data.load import load_all


def setup_logging(log_level: str = 'INFO') -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_and_prepare_data(data_path: str, target_col: str = 'price') -> tuple:
    """
    Load and prepare data for training.
    
    Args:
        data_path: Path to the data file
        target_col: Name of the target column
        
    Returns:
        Tuple of (X, y) where X is features and y is target
    """
    logger = logging.getLogger(__name__)
    
    # Load data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Remove missing values
    df_clean = df.dropna()
    logger.info(f"Data shape after cleaning: {df_clean.shape}")
    
    # Separate features and target
    if target_col not in df_clean.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")
    
    X = df_clean.drop([target_col], axis=1)
    y = df_clean[target_col]
    
    # Keep only numerical columns
    numerical_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numerical_cols]
    
    logger.info(f"Final data shape: X={X.shape}, y={y.shape}")
    
    return X, y


def train_traditional_models(X: pd.DataFrame, y: pd.Series, 
                           test_size: float = 0.2, 
                           val_size: float = 0.2,
                           random_state: int = 42) -> Dict[str, Any]:
    """
    Train traditional linear and tree-based models.
    
    Args:
        X: Features
        y: Targets
        test_size: Proportion for test set
        val_size: Proportion for validation set
        random_state: Random seed
        
    Returns:
        Dictionary with training results
    """
    logger = logging.getLogger(__name__)
    
    # Split data
    logger.info("Splitting data into train/validation/test sets")
    split_data = split_with_validation(
        X, y, method='random', 
        test_size=test_size, val_size=val_size, 
        random_state=random_state
    )
    
    # Create models
    logger.info("Creating traditional models")
    models = create_all_models()
    
    # Train all models
    results = {}
    for name, model in models.items():
        logger.info(f"Training {name}...")
        
        try:
            # Train model
            model.fit(split_data['X_train'], split_data['y_train'])
            
            # Evaluate on validation set
            val_metrics = model.evaluate(split_data['X_val'], split_data['y_val'])
            
            # Evaluate on test set
            test_metrics = model.evaluate(split_data['X_test'], split_data['y_test'])
            
            results[name] = {
                'model': model,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'training_history': model.training_history
            }
            
            logger.info(f"{name} - Val RMSE: {val_metrics['rmse']:.2f}, Test RMSE: {test_metrics['rmse']:.2f}")
            
        except Exception as e:
            logger.error(f"Error training {name}: {str(e)}")
            continue
    
    return results


def train_multimodal_model(data_path: str, 
                          test_size: float = 0.2,
                          val_size: float = 0.2,
                          random_state: int = 42) -> Dict[str, Any]:
    """
    Train multimodal model.
    
    Args:
        data_path: Path to the data file
        test_size: Proportion for test set
        val_size: Proportion for validation set
        random_state: Random seed
        
    Returns:
        Dictionary with training results
    """
    logger = logging.getLogger(__name__)
    
    # Load data
    df = pd.read_csv(data_path)
    df_clean = df.dropna()
    
    logger.info("Training multimodal model...")
    
    # Create and train multimodal model
    multimodal_model = MultimodalModel(
        use_images=True,
        use_cnn=False,  # Faster without CNN
        use_spatial=True,
        use_text=True
    )
    
    results = multimodal_model.train(
        df_clean,
        target_col='price',
        test_size=test_size,
        val_size=val_size,
        train_neural=True
    )
    
    return results


def train_neural_network(X: pd.DataFrame, y: pd.Series,
                        test_size: float = 0.2,
                        val_size: float = 0.2,
                        random_state: int = 42) -> Dict[str, Any]:
    """
    Train neural network model.
    
    Args:
        X: Features
        y: Targets
        test_size: Proportion for test set
        val_size: Proportion for validation set
        random_state: Random seed
        
    Returns:
        Dictionary with training results
    """
    logger = logging.getLogger(__name__)
    
    # Split data
    split_data = split_with_validation(
        X, y, method='random',
        test_size=test_size, val_size=val_size,
        random_state=random_state
    )
    
    logger.info("Training neural network...")
    
    # Train neural network
    model, scaler = train_mlp_model(
        split_data['X_train'], split_data['y_train'],
        hidden_sizes=[256, 128, 64],
        dropout_rate=0.3,
        epochs=50,
        batch_size=32,
        lr=0.001
    )
    
    if model is not None:
        # Evaluate on validation set
        from models.nn import evaluate_neural_network
        val_metrics = evaluate_neural_network(model, scaler, split_data['X_val'], split_data['y_val'])
        test_metrics = evaluate_neural_network(model, scaler, split_data['X_test'], split_data['y_test'])
        
        return {
            'model': model,
            'scaler': scaler,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics
        }
    else:
        logger.warning("Neural network training failed (PyTorch not available)")
        return {}


def save_models(results: Dict[str, Any], output_dir: str) -> None:
    """
    Save trained models to disk.
    
    Args:
        results: Dictionary with training results
        output_dir: Directory to save models
    """
    logger = logging.getLogger(__name__)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each model
    for name, result in results.items():
        if 'model' in result:
            model_path = os.path.join(output_dir, f"{name}.pkl")
            try:
                result['model'].save(model_path)
                logger.info(f"Saved {name} to {model_path}")
            except Exception as e:
                logger.error(f"Error saving {name}: {str(e)}")
    
    # Save training summary
    summary = {}
    for name, result in results.items():
        if 'test_metrics' in result:
            summary[name] = {
                'rmse': result['test_metrics']['rmse'],
                'mae': result['test_metrics']['mae'],
                'r2': result['test_metrics']['r2']
            }
    
    summary_path = os.path.join(output_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Training summary saved to {summary_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Airbnb price prediction models')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to the data file')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Directory to save trained models')
    parser.add_argument('--model-type', type=str, default='all',
                       choices=['traditional', 'multimodal', 'neural', 'all'],
                       help='Type of models to train')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proportion of data for test set')
    parser.add_argument('--val-size', type=float, default=0.2,
                       help='Proportion of data for validation set')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"training_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Starting training session: {timestamp}")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Model type: {args.model_type}")
    
    all_results = {}
    
    try:
        # Train traditional models
        if args.model_type in ['traditional', 'all']:
            logger.info("=" * 50)
            logger.info("TRAINING TRADITIONAL MODELS")
            logger.info("=" * 50)
            
            X, y = load_and_prepare_data(args.data_path)
            traditional_results = train_traditional_models(
                X, y, args.test_size, args.val_size, args.random_state
            )
            all_results.update(traditional_results)
        
        # Train multimodal model
        if args.model_type in ['multimodal', 'all']:
            logger.info("=" * 50)
            logger.info("TRAINING MULTIMODAL MODEL")
            logger.info("=" * 50)
            
            multimodal_results = train_multimodal_model(
                args.data_path, args.test_size, args.val_size, args.random_state
            )
            all_results['multimodal'] = multimodal_results
        
        # Train neural network
        if args.model_type in ['neural', 'all']:
            logger.info("=" * 50)
            logger.info("TRAINING NEURAL NETWORK")
            logger.info("=" * 50)
            
            X, y = load_and_prepare_data(args.data_path)
            neural_results = train_neural_network(
                X, y, args.test_size, args.val_size, args.random_state
            )
            if neural_results:
                all_results['neural_network'] = neural_results
        
        # Save models
        logger.info("=" * 50)
        logger.info("SAVING MODELS")
        logger.info("=" * 50)
        
        save_models(all_results, output_dir)
        
        # Print final summary
        logger.info("=" * 50)
        logger.info("TRAINING COMPLETED")
        logger.info("=" * 50)
        
        for name, result in all_results.items():
            if 'test_metrics' in result:
                metrics = result['test_metrics']
                logger.info(f"{name}: RMSE={metrics['rmse']:.2f}, R²={metrics['r2']:.3f}")
        
        logger.info(f"All models saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
