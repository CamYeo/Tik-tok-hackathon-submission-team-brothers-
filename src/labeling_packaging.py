"""
Labeling and Packaging Module

This module handles automated labeling using multiple LLMs and consensus-based
quality assessment. Based on the labeling_&_packaging.ipynb notebook.
"""

import pandas as pd
import numpy as np
import json
import time
from typing import Dict, List, Tuple, Optional, Any
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import cohen_kappa_score
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMLabeler:
    """
    Base class for LLM-based labeling.
    """
    
    def __init__(self, model_name: str, api_config: Dict):
        """
        Initialize the LLM labeler.
        
        Args:
            model_name: Name of the LLM model
            api_config: API configuration
        """
        self.model_name = model_name
        self.api_config = api_config
        self.label_mapping = {
            "advertisement": 0,
            "irrelevant": 1, 
            "rant_without_visit": 2,
            "relevant_and_quality": 3
        }
        
    def create_prompt(self, review_text: str) -> str:
        """
        Create a prompt for review classification.
        
        Args:
            review_text: The review text to classify
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""
Please classify the following Google location review into one of these categories:

1. advertisement - Reviews that are clearly promotional or advertising content
2. irrelevant - Reviews that are not related to the business or location
3. rant_without_visit - Complaints or rants from users who likely never visited the location
4. relevant_and_quality - Genuine, helpful reviews from actual customers

Review text: "{review_text}"

Please respond with only one of these four labels: advertisement, irrelevant, rant_without_visit, or relevant_and_quality
"""
        return prompt
    
    def label_review(self, review_text: str) -> Tuple[str, float]:
        """
        Label a single review.
        
        Args:
            review_text: The review text to classify
            
        Returns:
            Tuple of (predicted_label, confidence_score)
        """
        # This is a mock implementation
        # In practice, you would call the actual LLM API
        
        # Simulate API call delay
        time.sleep(0.1)
        
        # Mock prediction based on simple heuristics
        text_lower = review_text.lower()
        
        if any(word in text_lower for word in ['buy', 'sale', 'discount', 'promotion', 'deal']):
            return "advertisement", 0.8
        elif any(word in text_lower for word in ['terrible', 'worst', 'horrible', 'never', 'avoid']):
            return "rant_without_visit", 0.7
        elif any(word in text_lower for word in ['great', 'excellent', 'amazing', 'love', 'recommend']):
            return "relevant_and_quality", 0.9
        else:
            return "relevant_and_quality", 0.6
    
    def batch_label(self, review_texts: List[str]) -> List[Tuple[str, float]]:
        """
        Label a batch of reviews.
        
        Args:
            review_texts: List of review texts to classify
            
        Returns:
            List of (predicted_label, confidence_score) tuples
        """
        results = []
        for text in review_texts:
            label, confidence = self.label_review(text)
            results.append((label, confidence))
        return results

class MultiLLMLabeler:
    """
    Handles labeling using multiple LLMs with consensus-based quality assessment.
    """
    
    def __init__(self, llm_configs: List[Dict]):
        """
        Initialize the multi-LLM labeler.
        
        Args:
            llm_configs: List of LLM configurations
        """
        self.llm_labelers = []
        for config in llm_configs:
            labeler = LLMLabeler(config['model_name'], config['api_config'])
            self.llm_labelers.append(labeler)
        
        self.label_mapping = {
            "advertisement": 0,
            "irrelevant": 1,
            "rant_without_visit": 2, 
            "relevant_and_quality": 3
        }
        
        self.reverse_label_mapping = {v: k for k, v in self.label_mapping.items()}
    
    def get_consensus_label(self, predictions: List[Tuple[str, float]], 
                          method: str = "confidence_weighted") -> Tuple[str, float, bool]:
        """
        Get consensus label from multiple LLM predictions.
        
        Args:
            predictions: List of (label, confidence) tuples from different LLMs
            method: Consensus method ("majority_vote" or "confidence_weighted")
            
        Returns:
            Tuple of (consensus_label, consensus_confidence, is_high_quality)
        """
        if not predictions:
            return "relevant_and_quality", 0.0, False
        
        labels = [pred[0] for pred in predictions]
        confidences = [pred[1] for pred in predictions]
        
        if method == "majority_vote":
            # Simple majority vote
            from collections import Counter
            label_counts = Counter(labels)
            consensus_label = label_counts.most_common(1)[0][0]
            consensus_confidence = np.mean([conf for label, conf in predictions if label == consensus_label])
            
        elif method == "confidence_weighted":
            # Weight votes by confidence
            label_scores = {}
            for label, conf in predictions:
                if label not in label_scores:
                    label_scores[label] = 0
                label_scores[label] += conf
            
            consensus_label = max(label_scores.keys(), key=lambda x: label_scores[x])
            consensus_confidence = label_scores[consensus_label] / len(predictions)
        
        else:
            raise ValueError(f"Unknown consensus method: {method}")
        
        # Determine if this is high quality based on agreement and confidence
        agreement_score = labels.count(consensus_label) / len(labels)
        is_high_quality = (agreement_score >= 0.67 and consensus_confidence >= 0.6)
        
        return consensus_label, consensus_confidence, is_high_quality
    
    def calculate_inter_rater_agreement(self, all_predictions: List[List[str]]) -> Dict[str, float]:
        """
        Calculate inter-rater agreement metrics.
        
        Args:
            all_predictions: List of prediction lists from each LLM
            
        Returns:
            Dictionary containing agreement metrics
        """
        if len(all_predictions) < 2:
            return {"fleiss_kappa": 0.0, "pairwise_kappa_mean": 0.0}
        
        # Convert labels to numeric
        numeric_predictions = []
        for predictions in all_predictions:
            numeric_pred = [self.label_mapping.get(label, 3) for label in predictions]
            numeric_predictions.append(numeric_pred)
        
        # Calculate pairwise Cohen's kappa
        pairwise_kappas = []
        for i in range(len(numeric_predictions)):
            for j in range(i + 1, len(numeric_predictions)):
                try:
                    kappa = cohen_kappa_score(numeric_predictions[i], numeric_predictions[j])
                    pairwise_kappas.append(kappa)
                except:
                    continue
        
        pairwise_kappa_mean = np.mean(pairwise_kappas) if pairwise_kappas else 0.0
        
        # For Fleiss' kappa, we'll use a simplified approximation
        # In practice, you would use a proper implementation
        fleiss_kappa = pairwise_kappa_mean  # Simplified approximation
        
        return {
            "fleiss_kappa": fleiss_kappa,
            "pairwise_kappa_mean": pairwise_kappa_mean,
            "num_raters": len(all_predictions),
            "num_items": len(numeric_predictions[0]) if numeric_predictions else 0
        }
    
    def label_dataset(self, df: pd.DataFrame, text_column: str = 'text',
                     batch_size: int = 100, max_workers: int = 3) -> pd.DataFrame:
        """
        Label an entire dataset using multiple LLMs.
        
        Args:
            df: Input DataFrame
            text_column: Name of the text column
            batch_size: Batch size for processing
            max_workers: Number of parallel workers
            
        Returns:
            DataFrame with consensus labels and quality scores
        """
        logger.info(f"Starting multi-LLM labeling for {len(df)} samples")
        
        texts = df[text_column].astype(str).tolist()
        all_predictions = {f"llm_{i}": [] for i in range(len(self.llm_labelers))}
        all_confidences = {f"llm_{i}": [] for i in range(len(self.llm_labelers))}
        
        # Process in batches
        for start_idx in range(0, len(texts), batch_size):
            end_idx = min(start_idx + batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]
            
            logger.info(f"Processing batch {start_idx//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            # Get predictions from each LLM
            for i, labeler in enumerate(self.llm_labelers):
                batch_predictions = labeler.batch_label(batch_texts)
                
                batch_labels = [pred[0] for pred in batch_predictions]
                batch_confs = [pred[1] for pred in batch_predictions]
                
                all_predictions[f"llm_{i}"].extend(batch_labels)
                all_confidences[f"llm_{i}"].extend(batch_confs)
        
        # Calculate consensus labels
        consensus_labels = []
        consensus_confidences = []
        quality_flags = []
        
        for i in range(len(texts)):
            predictions = [(all_predictions[f"llm_{j}"][i], all_confidences[f"llm_{j}"][i]) 
                          for j in range(len(self.llm_labelers))]
            
            consensus_label, consensus_conf, is_high_quality = self.get_consensus_label(predictions)
            
            consensus_labels.append(consensus_label)
            consensus_confidences.append(consensus_conf)
            quality_flags.append(is_high_quality)
        
        # Add results to DataFrame
        result_df = df.copy()
        result_df['consensus_label'] = consensus_labels
        result_df['consensus_confidence'] = consensus_confidences
        result_df['is_high_quality'] = quality_flags
        
        # Add individual LLM predictions
        for i in range(len(self.llm_labelers)):
            result_df[f'llm_{i}_label'] = all_predictions[f"llm_{i}"]
            result_df[f'llm_{i}_confidence'] = all_confidences[f"llm_{i}"]
        
        # Calculate inter-rater agreement
        prediction_lists = [all_predictions[f"llm_{i}"] for i in range(len(self.llm_labelers))]
        ira_metrics = self.calculate_inter_rater_agreement(prediction_lists)
        
        logger.info(f"Labeling completed. Inter-rater agreement (Fleiss' κ): {ira_metrics['fleiss_kappa']:.3f}")
        logger.info(f"High quality samples: {sum(quality_flags)}/{len(quality_flags)} ({sum(quality_flags)/len(quality_flags)*100:.1f}%)")
        
        # Store metadata
        result_df.attrs['ira_metrics'] = ira_metrics
        result_df.attrs['labeling_metadata'] = {
            'num_llms': len(self.llm_labelers),
            'llm_models': [labeler.model_name for labeler in self.llm_labelers],
            'consensus_method': 'confidence_weighted',
            'high_quality_threshold': 0.6
        }
        
        return result_df
    
    def package_for_training(self, labeled_df: pd.DataFrame, 
                           output_path: str = "packaged_dataset.pkl",
                           quality_filter: bool = True) -> Dict[str, Any]:
        """
        Package the labeled dataset for model training.
        
        Args:
            labeled_df: DataFrame with consensus labels
            output_path: Path to save the packaged dataset
            quality_filter: Whether to filter for high-quality samples only
            
        Returns:
            Dictionary containing dataset statistics
        """
        logger.info("Packaging dataset for training...")
        
        # Filter for high-quality samples if requested
        if quality_filter and 'is_high_quality' in labeled_df.columns:
            filtered_df = labeled_df[labeled_df['is_high_quality']].copy()
            logger.info(f"Filtered to {len(filtered_df)} high-quality samples from {len(labeled_df)} total")
        else:
            filtered_df = labeled_df.copy()
        
        # Convert labels to numeric
        filtered_df['label_numeric'] = filtered_df['consensus_label'].map(self.label_mapping)
        
        # Create train/val/test splits
        from sklearn.model_selection import train_test_split
        
        train_df, temp_df = train_test_split(
            filtered_df, test_size=0.3, random_state=42, 
            stratify=filtered_df['label_numeric']
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, random_state=42,
            stratify=temp_df['label_numeric']
        )
        
        # Package the data
        packaged_data = {
            'train': train_df,
            'validation': val_df,
            'test': test_df,
            'label_mapping': self.label_mapping,
            'metadata': {
                'total_samples': len(labeled_df),
                'high_quality_samples': len(filtered_df),
                'train_samples': len(train_df),
                'val_samples': len(val_df),
                'test_samples': len(test_df),
                'ira_metrics': labeled_df.attrs.get('ira_metrics', {}),
                'labeling_metadata': labeled_df.attrs.get('labeling_metadata', {})
            }
        }
        
        # Save to file
        import pickle
        with open(output_path, 'wb') as f:
            pickle.dump(packaged_data, f)
        
        logger.info(f"Dataset packaged and saved to {output_path}")
        
        return packaged_data['metadata']

def main():
    """
    Example usage of the MultiLLMLabeler class.
    """
    # Configuration for multiple LLMs
    llm_configs = [
        {
            'model_name': 'google/gemma-2-2b-it',
            'api_config': {'api_key': 'mock_key', 'endpoint': 'mock_endpoint'}
        },
        {
            'model_name': 'Qwen/Qwen2-1.5B-Instruct', 
            'api_config': {'api_key': 'mock_key', 'endpoint': 'mock_endpoint'}
        },
        {
            'model_name': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
            'api_config': {'api_key': 'mock_key', 'endpoint': 'mock_endpoint'}
        }
    ]
    
    # Create sample data
    sample_data = {
        'text': [
            "Great restaurant with excellent food and service!",
            "Terrible experience, would not recommend to anyone.",
            "Buy our amazing products at 50% discount today!",
            "This review is completely unrelated to the restaurant.",
            "Average place, nothing special but decent food.",
            "Amazing atmosphere and friendly staff, will come back!"
        ]
    }
    df = pd.DataFrame(sample_data)
    
    # Initialize multi-LLM labeler
    labeler = MultiLLMLabeler(llm_configs)
    
    try:
        # Label the dataset
        labeled_df = labeler.label_dataset(df, batch_size=3)
        
        # Package for training
        metadata = labeler.package_for_training(labeled_df)
        
        print("Labeling and packaging completed!")
        print(f"Total samples: {metadata['total_samples']}")
        print(f"High quality samples: {metadata['high_quality_samples']}")
        print(f"Train/Val/Test: {metadata['train_samples']}/{metadata['val_samples']}/{metadata['test_samples']}")
        
        # Display sample results
        print("\nSample labeled data:")
        print(labeled_df[['text', 'consensus_label', 'consensus_confidence', 'is_high_quality']].head())
        
    except Exception as e:
        logger.error(f"Labeling failed: {e}")

if __name__ == "__main__":
    main()

