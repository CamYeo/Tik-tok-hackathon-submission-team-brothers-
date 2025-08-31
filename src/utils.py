"""
Utilities Module

This module contains utility functions and helper classes used across the project.
"""

import os
import json
import pickle
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Manages configuration files and settings.
    """
    
    def __init__(self, config_path: str = "configs/config.json"):
        """
        Initialize the ConfigManager.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config = {}
        self.load_config()
    
    def load_config(self):
        """Load configuration from file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
                logger.info(f"Configuration loaded from {self.config_path}")
            else:
                logger.warning(f"Configuration file not found: {self.config_path}")
                self.config = self.get_default_config()
                self.save_config()
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self.config = self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "data": {
                "input_file": "review-Wyoming_10.json",
                "output_dir": "preprocessed_data",
                "sample_size": 50000,
                "chunk_size": 32
            },
            "feature_engineering": {
                "num_topics_lda": 5,
                "num_keywords": 5,
                "sentiment_model": "cardiffnlp/twitter-roberta-base-sentiment-latest"
            },
            "model": {
                "model_name": "distilbert-base-uncased",
                "max_length": 512,
                "num_labels": 4,
                "batch_size": 16,
                "num_epochs": 3,
                "learning_rate": 2e-5
            },
            "active_learning": {
                "n_iterations": 5,
                "samples_per_iteration": 100,
                "strategy": "uncertainty_entropy"
            },
            "policy_enforcement": {
                "use_ml": True,
                "combine_methods": True,
                "risk_thresholds": {
                    "approve": 0.3,
                    "review": 0.7,
                    "reject": 0.8
                }
            }
        }
    
    def save_config(self):
        """Save configuration to file."""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation, e.g., 'model.batch_size')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """
        Set configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        self.save_config()

class DataManager:
    """
    Manages data loading, saving, and transformations.
    """
    
    @staticmethod
    def load_data(file_path: str, file_type: str = "auto") -> pd.DataFrame:
        """
        Load data from various file formats.
        
        Args:
            file_path: Path to the data file
            file_type: File type ('csv', 'json', 'pickle', 'auto')
            
        Returns:
            Loaded DataFrame
        """
        if file_type == "auto":
            file_type = Path(file_path).suffix.lower().lstrip('.')
        
        try:
            if file_type == "csv":
                return pd.read_csv(file_path)
            elif file_type == "json":
                return pd.read_json(file_path, lines=True)
            elif file_type in ["pkl", "pickle"]:
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            raise
    
    @staticmethod
    def save_data(df: pd.DataFrame, file_path: str, file_type: str = "auto"):
        """
        Save DataFrame to various file formats.
        
        Args:
            df: DataFrame to save
            file_path: Path to save the file
            file_type: File type ('csv', 'json', 'pickle', 'auto')
        """
        if file_type == "auto":
            file_type = Path(file_path).suffix.lower().lstrip('.')
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        try:
            if file_type == "csv":
                df.to_csv(file_path, index=False)
            elif file_type == "json":
                df.to_json(file_path, orient='records', lines=True)
            elif file_type in ["pkl", "pickle"]:
                with open(file_path, 'wb') as f:
                    pickle.dump(df, f)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            logger.info(f"Data saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving data to {file_path}: {e}")
            raise
    
    @staticmethod
    def split_data(df: pd.DataFrame, train_ratio: float = 0.7, 
                  val_ratio: float = 0.15, test_ratio: float = 0.15,
                  stratify_column: Optional[str] = None, 
                  random_state: int = 42) -> Dict[str, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            df: Input DataFrame
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            stratify_column: Column to stratify on
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary with 'train', 'val', 'test' DataFrames
        """
        from sklearn.model_selection import train_test_split
        
        # Validate ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")
        
        stratify = df[stratify_column] if stratify_column else None
        
        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            df, test_size=(val_ratio + test_ratio), 
            random_state=random_state, stratify=stratify
        )
        
        # Second split: val vs test
        if val_ratio > 0 and test_ratio > 0:
            val_test_ratio = test_ratio / (val_ratio + test_ratio)
            temp_stratify = temp_df[stratify_column] if stratify_column else None
            
            val_df, test_df = train_test_split(
                temp_df, test_size=val_test_ratio,
                random_state=random_state, stratify=temp_stratify
            )
        elif val_ratio > 0:
            val_df = temp_df
            test_df = pd.DataFrame()
        else:
            val_df = pd.DataFrame()
            test_df = temp_df
        
        return {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }

class MetricsCalculator:
    """
    Calculates various evaluation metrics.
    """
    
    @staticmethod
    def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                       labels: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Calculate classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Label names
            
        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import (
            accuracy_score, precision_recall_fscore_support,
            classification_report, confusion_matrix
        )
        
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': int(np.sum(support))
        }
        
        # Per-class metrics
        if labels:
            precision_per_class, recall_per_class, f1_per_class, support_per_class = \
                precision_recall_fscore_support(y_true, y_pred, average=None)
            
            for i, label in enumerate(labels):
                metrics[f'{label}_precision'] = precision_per_class[i]
                metrics[f'{label}_recall'] = recall_per_class[i]
                metrics[f'{label}_f1'] = f1_per_class[i]
                metrics[f'{label}_support'] = int(support_per_class[i])
        
        return metrics
    
    @staticmethod
    def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }

class Visualizer:
    """
    Creates various visualizations for data analysis and model evaluation.
    """
    
    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                            labels: Optional[List[str]] = None,
                            title: str = "Confusion Matrix",
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Label names
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        return plt.gcf()
    
    @staticmethod
    def plot_learning_curve(train_scores: List[float], val_scores: List[float],
                          title: str = "Learning Curve",
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot learning curve.
        
        Args:
            train_scores: Training scores
            val_scores: Validation scores
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_scores) + 1)
        
        plt.plot(epochs, train_scores, 'b-', label='Training Score')
        plt.plot(epochs, val_scores, 'r-', label='Validation Score')
        
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Learning curve saved to {save_path}")
        
        return plt.gcf()
    
    @staticmethod
    def plot_feature_importance(feature_names: List[str], importance_scores: List[float],
                              title: str = "Feature Importance",
                              top_n: int = 20,
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot feature importance.
        
        Args:
            feature_names: Names of features
            importance_scores: Importance scores
            title: Plot title
            top_n: Number of top features to show
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Sort features by importance
        feature_importance = list(zip(feature_names, importance_scores))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        # Take top N features
        top_features = feature_importance[:top_n]
        names, scores = zip(*top_features)
        
        plt.figure(figsize=(10, 8))
        y_pos = np.arange(len(names))
        
        plt.barh(y_pos, scores)
        plt.yticks(y_pos, names)
        plt.xlabel('Importance Score')
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        return plt.gcf()

class Logger:
    """
    Enhanced logging utilities.
    """
    
    @staticmethod
    def setup_logger(name: str, log_file: Optional[str] = None, 
                    level: int = logging.INFO) -> logging.Logger:
        """
        Set up a logger with file and console handlers.
        
        Args:
            name: Logger name
            log_file: Path to log file (optional)
            level: Logging level
            
        Returns:
            Configured logger
        """
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler (if specified)
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger

def main():
    """
    Example usage of utility classes.
    """
    # Test ConfigManager
    config_manager = ConfigManager("test_config.json")
    print(f"Model name: {config_manager.get('model.model_name')}")
    
    # Test DataManager
    sample_data = pd.DataFrame({
        'text': ['Sample text 1', 'Sample text 2'],
        'label': [0, 1]
    })
    
    DataManager.save_data(sample_data, "test_data.csv")
    loaded_data = DataManager.load_data("test_data.csv")
    print(f"Loaded data shape: {loaded_data.shape}")
    
    # Test MetricsCalculator
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    metrics = MetricsCalculator.calculate_classification_metrics(y_true, y_pred)
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    
    # Clean up test files
    for file in ["test_config.json", "test_data.csv"]:
        if os.path.exists(file):
            os.remove(file)
    
    print("Utility classes tested successfully!")

if __name__ == "__main__":
    main()

