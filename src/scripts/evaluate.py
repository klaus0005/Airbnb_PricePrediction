#!/usr/bin/env python3
"""
Evaluation script for Airbnb price prediction models.

This script provides comprehensive evaluation of trained models including
metrics calculation, visualization, and report generation.
"""

import os
import sys
import argparse
import logging
import json
import pickle
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score

from models.base_model import BaseModel
from models.linear import compare_models, plot_model_comparison
from data.split import split_with_validation, get_split_summary


def setup_logging(log_level: str = 'INFO') -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/evaluation.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_models(models_dir: str) -> Dict[str, BaseModel]:
    """
    Load trained models from directory.
    
    Args:
        models_dir: Directory containing model files
        
    Returns:
        Dictionary of loaded models
    """
    logger = logging.getLogger(__name__)
    models = {}
    
    for file_path in Path(models_dir).glob("*.pkl"):
        try:
            model_name = file_path.stem
            model = BaseModel.load(str(file_path))
            models[model_name] = model
            logger.info(f"Loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
    
    return models


def evaluate_single_model(model: BaseModel, X_test: pd.DataFrame, 
                         y_test: pd.Series, save_plots: bool = True,
                         output_dir: str = "evaluation_results") -> Dict[str, Any]:
    """
    Evaluate a single model comprehensively.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        save_plots: Whether to save plots
        output_dir: Directory to save results
        
    Returns:
        Dictionary with evaluation results
    """
    logger = logging.getLogger(__name__)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    }
    
    # Create plots if requested
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
        
        # Predictions plot
        plot_path = os.path.join(output_dir, f"{model.name}_predictions.png")
        model.plot_predictions(X_test, y_test, save_path=plot_path)
        
        # Feature importance plot (if available)
        importance_path = os.path.join(output_dir, f"{model.name}_importance.png")
        model.plot_feature_importance(save_path=importance_path)
    
    logger.info(f"{model.name} evaluation completed - RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.3f}")
    
    return {
        'model_name': model.name,
        'metrics': metrics,
        'predictions': y_pred.tolist(),
        'model_summary': model.get_summary()
    }


def compare_all_models(models: Dict[str, BaseModel], X: pd.DataFrame, 
                      y: pd.Series, cv: int = 5, save_plots: bool = True,
                      output_dir: str = "evaluation_results") -> pd.DataFrame:
    """
    Compare all models using cross-validation.
    
    Args:
        models: Dictionary of trained models
        X: Features
        y: Targets
        cv: Number of cross-validation folds
        save_plots: Whether to save comparison plots
        output_dir: Directory to save results
        
    Returns:
        DataFrame with comparison results
    """
    logger = logging.getLogger(__name__)
    
    logger.info("Comparing all models using cross-validation...")
    
    # Perform cross-validation comparison
    comparison_df = compare_models(models, X, y, cv=cv)
    
    # Create comparison plots
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, "model_comparison.png")
        plot_model_comparison(comparison_df, save_path=plot_path)
    
    return comparison_df


def generate_evaluation_report(results: Dict[str, Any], comparison_df: pd.DataFrame,
                             output_dir: str = "evaluation_results") -> str:
    """
    Generate a comprehensive evaluation report.
    
    Args:
        results: Dictionary with individual model results
        comparison_df: DataFrame with model comparison
        output_dir: Directory to save report
        
    Returns:
        Path to the generated report
    """
    logger = logging.getLogger(__name__)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Airbnb Price Prediction - Model Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1, h2, h3 {{ color: #333; }}
            .metric {{ background-color: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }}
            .best {{ background-color: #d4edda; border: 1px solid #c3e6cb; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .plot {{ text-align: center; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>Airbnb Price Prediction - Model Evaluation Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <h2>Executive Summary</h2>
        <p>This report presents the evaluation results for {len(results)} trained models.</p>
        
        <h2>Model Performance Comparison</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>RMSE (€)</th>
                <th>MAE (€)</th>
                <th>R²</th>
                <th>MAPE (%)</th>
            </tr>
    """
    
    # Add model results to table
    best_rmse = float('inf')
    best_model = None
    
    for model_name, result in results.items():
        metrics = result['metrics']
        rmse = metrics['rmse']
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model_name
        
        html_content += f"""
            <tr>
                <td>{model_name}</td>
                <td>{metrics['rmse']:.2f}</td>
                <td>{metrics['mae']:.2f}</td>
                <td>{metrics['r2']:.3f}</td>
                <td>{metrics['mape']:.1f}</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <h2>Best Performing Model</h2>
        <div class="metric best">
            <h3>🏆 Best Model: """ + best_model + """</h3>
            <p><strong>RMSE:</strong> """ + f"{best_rmse:.2f}€" + """</p>
        </div>
        
        <h2>Cross-Validation Results</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>RMSE Mean</th>
                <th>RMSE Std</th>
                <th>R² Mean</th>
                <th>R² Std</th>
            </tr>
    """
    
    # Add cross-validation results
    for _, row in comparison_df.iterrows():
        html_content += f"""
            <tr>
                <td>{row['Model']}</td>
                <td>{row['RMSE_mean']:.2f}</td>
                <td>{row['RMSE_std']:.2f}</td>
                <td>{row['R2_mean']:.3f}</td>
                <td>{row['R2_std']:.3f}</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <h2>Model Details</h2>
    """
    
    # Add individual model details
    for model_name, result in results.items():
        metrics = result['metrics']
        summary = result['model_summary']
        
        html_content += f"""
        <div class="metric">
            <h3>{model_name}</h3>
            <p><strong>Model Type:</strong> {summary.get('type', 'Unknown')}</p>
            <p><strong>Features:</strong> {summary.get('feature_count', 'Unknown')}</p>
            <p><strong>Training Status:</strong> {'Trained' if summary.get('is_trained', False) else 'Not Trained'}</p>
            <p><strong>RMSE:</strong> {metrics['rmse']:.2f}€</p>
            <p><strong>R²:</strong> {metrics['r2']:.3f}</p>
            <p><strong>MAE:</strong> {metrics['mae']:.2f}€</p>
            <p><strong>MAPE:</strong> {metrics['mape']:.1f}%</p>
        </div>
        """
    
    html_content += """
        <h2>Visualizations</h2>
        <p>Prediction plots and feature importance charts have been saved as PNG files in the output directory.</p>
        
        <h2>Recommendations</h2>
        <ul>
            <li>Use the best performing model for production deployment</li>
            <li>Consider ensemble methods to improve performance</li>
            <li>Monitor model performance over time</li>
            <li>Retrain models periodically with new data</li>
        </ul>
        
        <footer>
            <p><em>Report generated by Airbnb Price Prediction Evaluation System</em></p>
        </footer>
    </body>
    </html>
    """
    
    # Save HTML report
    report_path = os.path.join(output_dir, "evaluation_report.html")
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Evaluation report saved to: {report_path}")
    return report_path


def create_summary_statistics(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create summary statistics for all models.
    
    Args:
        results: Dictionary with model results
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'total_models': len(results),
        'best_model': None,
        'best_rmse': float('inf'),
        'worst_model': None,
        'worst_rmse': 0,
        'average_rmse': 0,
        'average_r2': 0,
        'model_types': set(),
        'performance_ranking': []
    }
    
    rmse_values = []
    r2_values = []
    
    for model_name, result in results.items():
        metrics = result['metrics']
        rmse = metrics['rmse']
        r2 = metrics['r2']
        
        rmse_values.append(rmse)
        r2_values.append(r2)
        
        # Track best and worst models
        if rmse < summary['best_rmse']:
            summary['best_rmse'] = rmse
            summary['best_model'] = model_name
        
        if rmse > summary['worst_rmse']:
            summary['worst_rmse'] = rmse
            summary['worst_model'] = model_name
        
        # Track model types
        model_type = result['model_summary'].get('type', 'Unknown')
        summary['model_types'].add(model_type)
    
    # Calculate averages
    summary['average_rmse'] = np.mean(rmse_values)
    summary['average_r2'] = np.mean(r2_values)
    
    # Create performance ranking
    performance_data = [(name, results[name]['metrics']['rmse']) for name in results.keys()]
    performance_data.sort(key=lambda x: x[1])
    summary['performance_ranking'] = [name for name, _ in performance_data]
    
    return summary


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate Airbnb price prediction models')
    parser.add_argument('--models-dir', type=str, required=True,
                       help='Directory containing trained models')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to the test data file')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--save-plots', action='store_true', default=True,
                       help='Save evaluation plots')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"evaluation_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Starting evaluation session: {timestamp}")
    logger.info(f"Models directory: {args.models_dir}")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Load models
        logger.info("Loading trained models...")
        models = load_models(args.models_dir)
        
        if not models:
            logger.error("No models found in the specified directory")
            sys.exit(1)
        
        logger.info(f"Loaded {len(models)} models")
        
        # Load test data
        logger.info("Loading test data...")
        df = pd.read_csv(args.data_path)
        df_clean = df.dropna()
        
        # Prepare features and target
        X = df_clean.drop(['price'], axis=1, errors='ignore')
        y = df_clean['price']
        
        # Keep only numerical columns
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numerical_cols]
        
        logger.info(f"Test data shape: X={X.shape}, y={y.shape}")
        
        # Split data for evaluation
        split_data = split_with_validation(X, y, method='random', 
                                         test_size=0.3, val_size=0.2, 
                                         random_state=42)
        
        # Evaluate individual models
        logger.info("Evaluating individual models...")
        results = {}
        for name, model in models.items():
            try:
                result = evaluate_single_model(
                    model, split_data['X_test'], split_data['y_test'],
                    save_plots=args.save_plots, output_dir=output_dir
                )
                results[name] = result
            except Exception as e:
                logger.error(f"Error evaluating {name}: {str(e)}")
                continue
        
        # Compare models using cross-validation
        logger.info("Performing cross-validation comparison...")
        comparison_df = compare_all_models(
            models, X, y, cv=args.cv_folds,
            save_plots=args.save_plots, output_dir=output_dir
        )
        
        # Generate summary statistics
        logger.info("Generating summary statistics...")
        summary_stats = create_summary_statistics(results)
        
        # Generate evaluation report
        logger.info("Generating evaluation report...")
        report_path = generate_evaluation_report(results, comparison_df, output_dir)
        
        # Save results to JSON
        results_path = os.path.join(output_dir, "evaluation_results.json")
        with open(results_path, 'w') as f:
            json.dump({
                'results': results,
                'comparison': comparison_df.to_dict('records'),
                'summary': summary_stats,
                'metadata': {
                    'timestamp': timestamp,
                    'models_evaluated': len(results),
                    'test_samples': len(split_data['X_test'])
                }
            }, f, indent=2, default=str)
        
        # Print final summary
        logger.info("=" * 50)
        logger.info("EVALUATION COMPLETED")
        logger.info("=" * 50)
        logger.info(f"Best model: {summary_stats['best_model']} (RMSE: {summary_stats['best_rmse']:.2f}€)")
        logger.info(f"Average RMSE: {summary_stats['average_rmse']:.2f}€")
        logger.info(f"Average R²: {summary_stats['average_r2']:.3f}")
        logger.info(f"Results saved to: {output_dir}")
        logger.info(f"Report available at: {report_path}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
