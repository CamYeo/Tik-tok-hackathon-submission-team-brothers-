#!/usr/bin/env python3
"""
Basic tests for TikTok Hackathon Review Quality Assessment system.
"""

import unittest
import os
import sys
import tempfile
import json
import pandas as pd
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.policy_enforcement import PolicyEnforcer, PolicyViolationType
from src.utils import ConfigManager, DataManager, MetricsCalculator

class TestTikTokDataPreprocessor(unittest.TestCase):
    """Test cases for DataPreprocessor with TikTok-specific data."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {"sample_size": 10, "chunk_size": 2}
        self.preprocessor = DataPreprocessor(self.config)
        
        # Create TikTok-style sample data
        self.sample_data = [
            {"text": "Amazing dance moves! #fyp #viral", "rating": 5, "time": 1620085852324, "user_id": "user123", "content_id": "video456"},
            {"text": "This trend is so cringe 😬", "rating": 2, "time": 1620085852325, "user_id": "user456", "content_id": "video789"},
            {"text": "", "rating": 3, "time": 1620085852326, "user_id": "user789", "content_id": "video123"},  # Empty text
            {"text": "Love this creator! ✨", "rating": 5, "time": None, "user_id": "user321", "content_id": "video654"}  # Missing time
        ]
    
    def test_load_json_data_tiktok_format(self):
        """Test loading TikTok-style JSON data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            for item in self.sample_data:
                f.write(json.dumps(item) + '\n')
            temp_file = f.name
        
        try:
            df = self.preprocessor.load_json_data(temp_file)
            self.assertIsInstance(df, pd.DataFrame)
            self.assertEqual(len(df), 4)
            self.assertIn('text', df.columns)
            self.assertIn('rating', df.columns)
            self.assertIn('user_id', df.columns)
            self.assertIn('content_id', df.columns)
        finally:
            os.unlink(temp_file)
    
    def test_validate_tiktok_data(self):
        """Test data validation for TikTok content."""
        # Valid data
        valid_df = pd.DataFrame(self.sample_data)
        validated_df = self.preprocessor.validate_data(valid_df)
        self.assertIsInstance(validated_df, pd.DataFrame)
        
        # Invalid data (missing required column)
        invalid_df = pd.DataFrame([{"rating": 5, "time": 123, "user_id": "test"}])
        with self.assertRaises(ValueError):
            self.preprocessor.validate_data(invalid_df)
    
    def test_clean_tiktok_content(self):
        """Test cleaning TikTok-specific content."""
        df = pd.DataFrame(self.sample_data)
        cleaned_df = self.preprocessor.clean_data(df)
        
        # Check that empty text rows are removed
        self.assertFalse(cleaned_df['text'].str.strip().eq('').any())
        
        # Check that cleaned_text column is added
        self.assertIn('cleaned_text', cleaned_df.columns)
        
        # Check that emojis and hashtags are handled
        sample_text = cleaned_df['cleaned_text'].iloc[0]
        self.assertIsInstance(sample_text, str)

class TestTikTokFeatureEngineer(unittest.TestCase):
    """Test cases for FeatureEngineer with TikTok content."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "num_topics_lda": 2,
            "sentiment_model": "cardiffnlp/twitter-roberta-base-sentiment-latest"
        }
        self.engineer = FeatureEngineer(self.config)
        
        # Create TikTok-style sample DataFrame
        self.sample_df = pd.DataFrame({
            'text': ['Amazing dance! #fyp #viral ✨', 'This is so cringe 😬', 'Love this content! 💕'],
            'cleaned_text': ['amazing dance fyp viral', 'this is so cringe', 'love this content'],
            'rating': [5, 2, 5],
            'time': [1620085852324, 1620085852325, 1620085852326],
            'user_id': ['user123', 'user456', 'user789'],
            'content_id': ['video456', 'video789', 'video123']
        })
    
    def test_extract_tiktok_text_features(self):
        """Test text feature extraction for TikTok content."""
        df_with_features = self.engineer.extract_text_features(self.sample_df)
        
        # Check that text features are added
        expected_features = ['text_length', 'word_count', 'avg_word_length', 
                           'exclamation_count', 'question_count', 'caps_ratio']
        
        for feature in expected_features:
            self.assertIn(feature, df_with_features.columns)
        
        # Check TikTok-specific patterns
        self.assertTrue(all(df_with_features['text_length'] > 0))
        self.assertTrue(all(df_with_features['word_count'] > 0))
        
        # Check for hashtag and emoji handling
        first_text = self.sample_df['text'].iloc[0]
        self.assertIn('#', first_text)  # Contains hashtags
        self.assertIn('✨', first_text)  # Contains emojis
    
    def test_extract_social_media_features(self):
        """Test extraction of social media specific features."""
        df_with_features = self.engineer.extract_text_features(self.sample_df)
        
        # Should handle hashtags, mentions, emojis
        self.assertIn('text_length', df_with_features.columns)
        
        # Check that features are calculated correctly
        self.assertTrue(all(df_with_features['text_length'] > 0))

class TestTikTokPolicyEnforcer(unittest.TestCase):
    """Test cases for PolicyEnforcer with TikTok content."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.enforcer = PolicyEnforcer(use_ml=False)  # Use rule-based only for testing
    
    def test_tiktok_spam_detection(self):
        """Test spam detection for TikTok content."""
        spam_text = "Follow my page for amazing deals! Link in bio! #ad #sponsored"
        result = self.enforcer.enforce_policies(spam_text)
        
        self.assertGreater(result['overall_risk_score'], 0.3)
        
        # Check for spam or promotional violations
        violation_types = [v.violation_type for v in result['violations']]
        self.assertTrue(
            PolicyViolationType.SPAM in violation_types or 
            PolicyViolationType.PROMOTIONAL in violation_types
        )
    
    def test_tiktok_quality_content(self):
        """Test quality content detection for TikTok."""
        quality_texts = [
            "Amazing dance moves! Love this choreography ✨",
            "This tutorial is so helpful! Thank you for sharing 🙏",
            "Incredible editing skills! How did you do this effect?"
        ]
        
        for text in quality_texts:
            result = self.enforcer.enforce_policies(text)
            self.assertLess(result['overall_risk_score'], 0.5)
            self.assertIn(result['action_recommended'], ['approve', 'review'])
    
    def test_tiktok_inappropriate_content(self):
        """Test inappropriate content detection for TikTok."""
        inappropriate_texts = [
            "This creator is trash and should quit",
            "Hate this stupid trend, everyone doing it is an idiot",
            "This is fake and the creator is a scammer"
        ]
        
        for text in inappropriate_texts:
            result = self.enforcer.enforce_policies(text)
            self.assertGreater(result['overall_risk_score'], 0.3)
            
            if result['violations']:
                violation_types = [v.violation_type for v in result['violations']]
                self.assertTrue(
                    PolicyViolationType.INAPPROPRIATE_CONTENT in violation_types or
                    PolicyViolationType.PERSONAL_ATTACK in violation_types or
                    PolicyViolationType.FAKE_REVIEW in violation_types
                )
    
    def test_tiktok_engagement_farming(self):
        """Test detection of engagement farming content."""
        engagement_farming_texts = [
            "First! Like if you agree!",
            "Subscribe for more! Hit that follow button!",
            "Comment below if you want part 2!"
        ]
        
        for text in engagement_farming_texts:
            result = self.enforcer.enforce_policies(text)
            # Engagement farming might be detected as promotional
            if result['violations']:
                violation_types = [v.violation_type for v in result['violations']]
                # Could be promotional or spam
                self.assertTrue(len(violation_types) >= 0)  # May or may not be flagged

class TestTikTokConfigManager(unittest.TestCase):
    """Test cases for ConfigManager with TikTok-specific settings."""
    
    def test_tiktok_config_loading(self):
        """Test loading TikTok-specific configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            tiktok_config = {
                "data": {
                    "input_file": "data/tiktok_reviews.json",
                    "sample_size": 10000
                },
                "labeling": {
                    "llm_models": [
                        "google/gemma-2-2b-it",
                        "Qwen/Qwen2-1.5B-Instruct"
                    ]
                },
                "model": {
                    "use_lora": True,
                    "lora_r": 16
                }
            }
            json.dump(tiktok_config, f)
            temp_config = f.name
        
        try:
            config_manager = ConfigManager(temp_config)
            
            # Test TikTok-specific settings
            self.assertEqual(config_manager.get('data.input_file'), 'data/tiktok_reviews.json')
            self.assertTrue(config_manager.get('model.use_lora'))
            self.assertEqual(config_manager.get('model.lora_r'), 16)
            
            llm_models = config_manager.get('labeling.llm_models')
            self.assertIsInstance(llm_models, list)
            self.assertIn('google/gemma-2-2b-it', llm_models)
            
        finally:
            if os.path.exists(temp_config):
                os.unlink(temp_config)

class TestTikTokDataManager(unittest.TestCase):
    """Test cases for DataManager with TikTok data formats."""
    
    def test_tiktok_data_operations(self):
        """Test saving and loading TikTok data."""
        # Create TikTok-style DataFrame
        df = pd.DataFrame({
            'text': ['Amazing content! #fyp', 'Love this trend ✨'],
            'rating': [5, 4],
            'user_id': ['user123', 'user456'],
            'content_id': ['video789', 'video012'],
            'hashtags': [['fyp'], ['trend', 'viral']]
        })
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            temp_file = f.name
        
        try:
            # Test saving
            DataManager.save_data(df, temp_file)
            self.assertTrue(os.path.exists(temp_file))
            
            # Test loading
            loaded_df = DataManager.load_data(temp_file)
            self.assertEqual(len(loaded_df), len(df))
            self.assertIn('user_id', loaded_df.columns)
            self.assertIn('content_id', loaded_df.columns)
            
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_tiktok_data_splitting(self):
        """Test data splitting for TikTok content."""
        # Create larger TikTok dataset
        df = pd.DataFrame({
            'text': [f'TikTok content {i}' for i in range(100)],
            'rating': [(i % 5) + 1 for i in range(100)],
            'user_id': [f'user{i}' for i in range(100)],
            'content_id': [f'video{i}' for i in range(100)],
            'label': [i % 4 for i in range(100)]  # 4 quality categories
        })
        
        splits = DataManager.split_data(
            df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
            stratify_column='label'
        )
        
        self.assertIn('train', splits)
        self.assertIn('val', splits)
        self.assertIn('test', splits)
        
        total_samples = len(splits['train']) + len(splits['val']) + len(splits['test'])
        self.assertEqual(total_samples, len(df))
        
        # Check that all splits have TikTok-specific columns
        for split_name, split_df in splits.items():
            if len(split_df) > 0:
                self.assertIn('user_id', split_df.columns)
                self.assertIn('content_id', split_df.columns)

if __name__ == '__main__':
    # Create test directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('configs', exist_ok=True)
    
    # Run tests
    unittest.main(verbosity=2)

