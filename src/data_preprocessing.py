"""
Data Preprocessing Module

This module handles the initial data collection and preprocessing for Google location reviews.
Based on the Copyofdata_preprocessing.ipynb notebook.
"""

import json
import pandas as pd
import os
import warnings
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Handles data collection and initial preprocessing for Google location reviews.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the DataPreprocessor.
        
        Args:
            config: Configuration dictionary containing file paths and parameters
        """
        self.config = config
        self.required_columns = ['text', 'rating']
        
    def load_json_data(self, file_path: str) -> pd.DataFrame:
        """
        Loads data from a JSON file where each line is a JSON object.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            DataFrame containing the loaded data
        """
        data = []
        try:
            with open(file_path, "r", encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Error decoding JSON from line {line_num}: {e}")
        except FileNotFoundError:
            logger.error(f"Error: The file at {file_path} was not found.")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        logger.info(f"Loaded {len(df)} records from {file_path}")
        return df
    
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate the loaded data and check for required columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Validated DataFrame
            
        Raises:
            ValueError: If required columns are missing
        """
        if df.empty:
            raise ValueError("No data loaded. Please check the file and try again.")
        
        # Check for required columns
        missing_columns = [col for col in self.required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        logger.info(f"Data validation passed. Dataset shape: {df.shape}")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform initial data cleaning.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        initial_count = len(df)
        
        # Handle missing values
        df = df.dropna(subset=self.required_columns)
        dropped_count = initial_count - len(df)
        if dropped_count > 0:
            logger.info(f"Dropped {dropped_count} rows with missing text or rating")
        
        # Convert 'time' to datetime
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], unit='ms', errors='coerce')
        else:
            df['time'] = pd.Timestamp.now()  # Default timestamp if missing
            logger.warning("Time column missing, using current timestamp")
        
        # Basic text cleaning
        df['cleaned_text'] = (df['text']
                              .astype(str)
                              .str.lower()
                              .str.strip())
        
        logger.info("Data cleaning completed")
        return df
    
    def preprocess(self, file_path: str) -> pd.DataFrame:
        """
        Complete preprocessing pipeline.
        
        Args:
            file_path: Path to the input JSON file
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Starting data preprocessing pipeline")
        
        # Load data
        df = self.load_json_data(file_path)
        
        # Validate data
        df = self.validate_data(df)
        
        # Clean data
        df = self.clean_data(df)
        
        logger.info("Data preprocessing pipeline completed")
        return df

def main():
    """
    Example usage of the DataPreprocessor class.
    """
    config = {
        'input_file': 'review-Wyoming_10.json',
        'output_dir': 'preprocessed_data'
    }
    
    preprocessor = DataPreprocessor(config)
    
    # Example preprocessing
    try:
        df = preprocessor.preprocess(config['input_file'])
        print(f"Preprocessed {len(df)} reviews")
        print(f"Columns: {list(df.columns)}")
        print(f"Sample data:\n{df.head()}")
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")

if __name__ == "__main__":
    main()

