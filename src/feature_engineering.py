"""
Feature Engineering Module

This module handles comprehensive feature engineering for Google location reviews.
Based on the feature_engineering.ipynb notebook.
"""

import pandas as pd
import numpy as np
import re
import warnings
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Scikit-learn imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Transformers and Torch imports
import torch
from transformers import pipeline
from tqdm.auto import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Handles comprehensive feature engineering for review quality assessment.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the FeatureEngineer.
        
        Args:
            config: Configuration dictionary containing parameters
        """
        self.config = config
        self.device = self._get_device()
        self.sentiment_pipeline = None
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        
        # Download required NLTK data
        self._download_nltk_data()
        
    def _download_nltk_data(self):
        """Download required NLTK data."""
        try:
            nltk.download("stopwords", quiet=True)
            nltk.download("wordnet", quiet=True)
            nltk.download("punkt", quiet=True)
            nltk.download('punkt_tab', quiet=True)
        except Exception as e:
            logger.warning(f"Error downloading NLTK data: {e}")
    
    def _get_device(self):
        """Get the best available device for model inference."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name()}")
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            logger.info("Using Apple Silicon GPU (MPS)")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU")
        return device
    
    def extract_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract sentiment features using a pre-trained model.
        
        Args:
            df: Input DataFrame with 'cleaned_text' column
            
        Returns:
            DataFrame with sentiment features added
        """
        logger.info("Extracting sentiment features...")
        
        try:
            # Initialize sentiment pipeline
            if self.sentiment_pipeline is None:
                model_name = self.config.get('sentiment_model', 'cardiffnlp/twitter-roberta-base-sentiment-latest')
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model=model_name,
                    device=0 if self.device.type == 'cuda' else -1,
                    padding=True,
                    truncation=True,
                    max_length=512
                )
            
            # Process texts in batches
            texts_to_analyze = df['cleaned_text'].tolist()
            batch_size = self.config.get('batch_size', 32)
            batch_results = self.sentiment_pipeline(texts_to_analyze, batch_size=batch_size)
            
            # Process results
            sent_labels = [res['label'] for res in batch_results]
            sent_scores = [res['score'] for res in batch_results]
            
            # Map labels to scores
            score_map = {
                'LABEL_0': -1, 'LABEL_1': 0, 'LABEL_2': 1,
                'NEGATIVE': -1, 'NEUTRAL': 0, 'POSITIVE': 1
            }
            df['sentiment_score'] = [
                score_map.get(label, 0) * score for label, score in zip(sent_labels, sent_scores)
            ]
            
            logger.info("Sentiment analysis complete")
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            df['sentiment_score'] = 0.0
            
        return df
    
    def extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract time-based features.
        
        Args:
            df: Input DataFrame with 'time' column
            
        Returns:
            DataFrame with time features added
        """
        logger.info("Extracting time-based features...")
        
        # Ensure time column is properly formatted
        if 'time' not in df.columns or df['time'].isna().all():
            df['time'] = pd.Timestamp.now()
            logger.warning("Using default timestamp for missing time data")
        
        df["review_day_of_week"] = df["time"].dt.dayofweek
        df["review_hour_of_day"] = df["time"].dt.hour
        
        logger.info("Time-based features complete")
        return df
    
    def preprocess_text_for_topic_modeling(self, text: str) -> str:
        """
        Preprocess text for topic modeling.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        try:
            tokens = word_tokenize(str(text))
            tokens = [self.lemmatizer.lemmatize(word.lower()) for word in tokens
                     if word.isalpha() and word.lower() not in self.stop_words and len(word) > 2]
            return " ".join(tokens)
        except Exception:
            return ""
    
    def extract_topic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract topic modeling features using LDA.
        
        Args:
            df: Input DataFrame with 'cleaned_text' column
            
        Returns:
            DataFrame with topic features added
        """
        logger.info("Extracting topic modeling features...")
        
        # Preprocess text for topic modeling
        df["processed_text_for_topic"] = df["cleaned_text"].apply(
            self.preprocess_text_for_topic_modeling
        )
        
        # Remove empty processed texts
        non_empty_mask = df["processed_text_for_topic"].str.len() > 0
        processed_texts = df.loc[non_empty_mask, "processed_text_for_topic"]
        
        num_topics = self.config.get('num_topics_lda', 5)
        
        if len(processed_texts) > num_topics and len(processed_texts) > 10:
            try:
                vectorizer = TfidfVectorizer(
                    max_df=0.95,
                    min_df=2,
                    stop_words="english",
                    max_features=1000
                )
                dtm = vectorizer.fit_transform(processed_texts)
                
                lda = LatentDirichletAllocation(
                    n_components=num_topics,
                    random_state=42,
                    max_iter=10  # Reduce iterations for faster processing
                )
                lda.fit(dtm)
                
                # Get topic assignments for all texts
                all_dtm = vectorizer.transform(df["processed_text_for_topic"])
                df["dominant_topic"] = lda.transform(all_dtm).argmax(axis=1)
                
                logger.info(f"Topic modeling complete with {num_topics} topics")
                
            except Exception as e:
                logger.error(f"Error in topic modeling: {e}")
                df["dominant_topic"] = 0
        else:
            df["dominant_topic"] = 0
            logger.warning("Insufficient data for topic modeling, using default topic")
            
        return df
    
    def extract_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract various text-based features.
        
        Args:
            df: Input DataFrame with 'text' column
            
        Returns:
            DataFrame with text features added
        """
        logger.info("Extracting text features...")
        
        # Basic text statistics
        df['review_length_words'] = df['text'].str.split().str.len()
        df['review_length_chars'] = df['text'].str.len()
        
        # URL and contact information
        df['num_urls'] = df['text'].str.count(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        df['num_emails'] = df['text'].str.count(r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b')
        df['num_phone_numbers'] = df['text'].str.count(r'\\b\\d{3}[-.]?\\d{3}[-.]?\\d{4}\\b')
        
        # Social media patterns
        df['num_mentions'] = df['text'].str.count(r'@\\w+')
        df['num_hashtags'] = df['text'].str.count(r'#\\w+')
        
        # Punctuation and formatting
        df['num_exclamations'] = df['text'].str.count('!')
        df['num_questions'] = df['text'].str.count('\\?')
        df['num_ellipsis'] = df['text'].str.count(r'\\.\\.\\.')
        df['all_caps_word_count'] = df['text'].str.count(r'\\b[A-Z]{2,}\\b')
        
        # Ratios
        df['caps_ratio'] = df['text'].str.count(r'[A-Z]') / df['review_length_chars'].replace(0, 1)
        df['digit_ratio'] = df['text'].str.count(r'\\d') / df['review_length_chars'].replace(0, 1)
        df['punctuation_ratio'] = df['text'].str.count(r'[^\\w\\s]') / df['review_length_chars'].replace(0, 1)
        
        # Fill NaN values with 0
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        logger.info("Text features extraction complete")
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete feature engineering pipeline.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with all engineered features
        """
        logger.info("Starting feature engineering pipeline...")
        
        # Sample data if specified
        sample_size = self.config.get('sample_size', -1)
        if sample_size > 0 and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            logger.info(f"Using sample of {len(df)} reviews")
        
        # Extract different types of features
        df = self.extract_sentiment_features(df)
        df = self.extract_time_features(df)
        df = self.extract_topic_features(df)
        df = self.extract_text_features(df)
        
        logger.info("Feature engineering pipeline completed")
        return df

def main():
    """
    Example usage of the FeatureEngineer class.
    """
    config = {
        'sentiment_model': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
        'batch_size': 32,
        'sample_size': 1000,
        'num_topics_lda': 5
    }
    
    # Create sample data
    sample_data = {
        'text': [
            "Great restaurant with excellent food!",
            "Terrible service, would not recommend.",
            "Average place, nothing special.",
            "Amazing experience, will definitely come back!"
        ],
        'rating': [5, 1, 3, 5],
        'time': pd.date_range('2023-01-01', periods=4, freq='D')
    }
    df = pd.DataFrame(sample_data)
    df['cleaned_text'] = df['text'].str.lower()
    
    engineer = FeatureEngineer(config)
    
    try:
        df_engineered = engineer.engineer_features(df)
        print(f"Engineered features for {len(df_engineered)} reviews")
        print(f"Feature columns: {list(df_engineered.columns)}")
        print(f"Sample data:\n{df_engineered.head()}")
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")

if __name__ == "__main__":
    main()

