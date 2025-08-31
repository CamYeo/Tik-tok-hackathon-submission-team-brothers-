"""
Model Training Module

This module handles model training for Google location review quality assessment.
Based on the model_training-3.ipynb notebook.
"""

import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for model training."""
    model_name: str = "distilbert-base-uncased"
    max_length: int = 512
    num_labels: int = 4
    batch_size: int = 16
    num_epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    logging_steps: int = 50
    eval_steps: int = 500
    save_steps: int = 500
    output_dir: str = "./models"
    seed: int = 42

class ReviewDataset(Dataset):
    """
    Custom dataset for review classification.
    """
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        """
        Initialize the dataset.
        
        Args:
            texts: List of review texts
            labels: List of corresponding labels
            tokenizer: Tokenizer for text encoding
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class ModelTrainer:
    """
    Handles model training for review quality assessment.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the ModelTrainer.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.tokenizer = None
        self.model = None
        self.label_encoder = LabelEncoder()
        self.device = self._get_device()
        
        # Set random seeds for reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
    def _get_device(self):
        """Get the best available device."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU")
        return device
    
    def prepare_data(self, df: pd.DataFrame, text_column: str = 'text', 
                    label_column: str = 'label') -> Tuple[Dataset, Dataset, Dataset]:
        """
        Prepare training, validation, and test datasets.
        
        Args:
            df: Input DataFrame
            text_column: Name of the text column
            label_column: Name of the label column
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        logger.info("Preparing datasets...")
        
        # Extract texts and labels
        texts = df[text_column].astype(str).tolist()
        labels = df[label_column].tolist()
        
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            texts, encoded_labels, test_size=0.3, random_state=self.config.seed, stratify=encoded_labels
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=self.config.seed, stratify=y_temp
        )
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # Create datasets
        train_dataset = ReviewDataset(X_train, y_train, self.tokenizer, self.config.max_length)
        val_dataset = ReviewDataset(X_val, y_val, self.tokenizer, self.config.max_length)
        test_dataset = ReviewDataset(X_test, y_test, self.tokenizer, self.config.max_length)
        
        logger.info(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset
    
    def initialize_model(self):
        """Initialize the model for training."""
        logger.info(f"Initializing model: {self.config.model_name}")
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=self.config.num_labels
        )
        
        # Move model to device
        self.model.to(self.device)
        
    def compute_metrics(self, eval_pred):
        """
        Compute evaluation metrics.
        
        Args:
            eval_pred: Evaluation predictions
            
        Returns:
            Dictionary of metrics
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train(self, train_dataset: Dataset, val_dataset: Dataset) -> Dict:
        """
        Train the model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            
        Returns:
            Training history
        """
        logger.info("Starting model training...")
        
        # Initialize model if not already done
        if self.model is None:
            self.initialize_model()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            logging_dir=f"{self.config.output_dir}/logs",
            logging_steps=self.config.logging_steps,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            seed=self.config.seed,
            learning_rate=self.config.learning_rate,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train the model
        train_result = trainer.train()
        
        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        logger.info("Model training completed")
        
        return train_result
    
    def evaluate(self, test_dataset: Dataset) -> Dict:
        """
        Evaluate the model on test data.
        
        Args:
            test_dataset: Test dataset
            
        Returns:
            Evaluation metrics
        """
        logger.info("Evaluating model...")
        
        # Create trainer for evaluation
        trainer = Trainer(
            model=self.model,
            compute_metrics=self.compute_metrics
        )
        
        # Evaluate
        eval_result = trainer.evaluate(test_dataset)
        
        logger.info(f"Evaluation results: {eval_result}")
        
        return eval_result
    
    def save_training_metadata(self, train_result: Dict, eval_result: Dict, 
                             output_path: str = "training_metadata.json"):
        """
        Save training metadata to a JSON file.
        
        Args:
            train_result: Training results
            eval_result: Evaluation results
            output_path: Path to save metadata
        """
        metadata = {
            "config": {
                "model_name": self.config.model_name,
                "max_length": self.config.max_length,
                "num_labels": self.config.num_labels,
                "batch_size": self.config.batch_size,
                "num_epochs": self.config.num_epochs,
                "learning_rate": self.config.learning_rate,
            },
            "training_results": {
                "train_loss": train_result.training_loss,
                "train_runtime": train_result.metrics.get('train_runtime', 0),
                "train_samples_per_second": train_result.metrics.get('train_samples_per_second', 0),
            },
            "evaluation_results": eval_result,
            "label_classes": self.label_encoder.classes_.tolist()
        }
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Training metadata saved to {output_path}")

def main():
    """
    Example usage of the ModelTrainer class.
    """
    # Configuration
    config = ModelConfig(
        model_name="distilbert-base-uncased",
        num_epochs=2,
        batch_size=8,  # Smaller batch size for demo
        output_dir="./demo_model"
    )
    
    # Create sample data
    sample_data = {
        'text': [
            "Great restaurant with excellent food and service!",
            "Terrible experience, would not recommend to anyone.",
            "Average place, nothing special but okay.",
            "Amazing food and atmosphere, highly recommended!",
            "Poor quality food and rude staff.",
            "Decent restaurant with good value for money."
        ] * 50,  # Repeat to have enough data
        'label': ['relevant_and_quality', 'rant_without_visit', 'relevant_and_quality', 
                 'relevant_and_quality', 'rant_without_visit', 'relevant_and_quality'] * 50
    }
    df = pd.DataFrame(sample_data)
    
    # Initialize trainer
    trainer = ModelTrainer(config)
    
    try:
        # Prepare data
        train_dataset, val_dataset, test_dataset = trainer.prepare_data(df)
        
        # Train model
        train_result = trainer.train(train_dataset, val_dataset)
        
        # Evaluate model
        eval_result = trainer.evaluate(test_dataset)
        
        # Save metadata
        trainer.save_training_metadata(train_result, eval_result)
        
        print("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")

if __name__ == "__main__":
    main()

