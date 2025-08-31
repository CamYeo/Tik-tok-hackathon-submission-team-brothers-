"""
Model Evaluation Module

This module handles comprehensive model evaluation for Google location review quality assessment.
Based on the model_eval.ipynb notebook.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
from typing import Dict, List, Tuple, Optional, Any
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Comprehensive model evaluation for review quality assessment.
    """
    
    def __init__(self, model_path: str, tokenizer_path: Optional[str] = None):
        """
        Initialize the ModelEvaluator.
        
        Args:
            model_path: Path to the trained model
            tokenizer_path: Path to the tokenizer (defaults to model_path)
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.model = None
        self.tokenizer = None
        self.device = self._get_device()
        self.label_names = [
            "Spam/Advertisement",
            "Irrelevant Content", 
            "Rant/Complaint (without visit)",
            "Relevant and Quality"
        ]
        
    def _get_device(self):
        """Get the best available device."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU")
        return device
    
    def load_model(self):
        """Load the trained model and tokenizer."""
        logger.info(f"Loading model from {self.model_path}")
        
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def predict(self, texts: List[str], batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on a list of texts.
        
        Args:
            texts: List of texts to predict
            batch_size: Batch size for prediction
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.model is None:
            self.load_model()
        
        all_predictions = []
        all_probabilities = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_probabilities)
    
    def calculate_per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Calculate per-class precision, recall, F1, and support.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of per-class metrics
        """
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        metrics = {}
        for i, label_name in enumerate(self.label_names):
            metrics[label_name] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i])
            }
        
        return metrics
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.label_names,
            yticklabels=self.label_names
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        return plt.gcf()
    
    def plot_one_vs_rest_confusion_matrices(self, y_true: np.ndarray, y_pred: np.ndarray,
                                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot one-vs-rest confusion matrices for each class.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, label_name in enumerate(self.label_names):
            # Create binary labels (current class vs all others)
            y_true_binary = (y_true == i).astype(int)
            y_pred_binary = (y_pred == i).astype(int)
            
            cm = confusion_matrix(y_true_binary, y_pred_binary)
            
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=['Other', label_name],
                yticklabels=['Other', label_name],
                ax=axes[i]
            )
            axes[i].set_title(f'{label_name} vs Rest')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"One-vs-rest confusion matrices saved to {save_path}")
        
        return fig
    
    def plot_roc_curves(self, y_true: np.ndarray, y_prob: np.ndarray,
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot ROC curves for each class.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Binarize the labels for multiclass ROC
        y_true_bin = label_binarize(y_true, classes=range(len(self.label_names)))
        
        plt.figure(figsize=(10, 8))
        
        for i, label_name in enumerate(self.label_names):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            auc = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
            
            plt.plot(fpr, tpr, label=f'{label_name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - One vs Rest')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curves saved to {save_path}")
        
        return plt.gcf()
    
    def evaluate_model(self, test_texts: List[str], test_labels: List[int],
                      output_dir: str = "./evaluation_results") -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            test_texts: List of test texts
            test_labels: List of test labels
            output_dir: Directory to save evaluation results
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        logger.info("Starting comprehensive model evaluation...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Make predictions
        y_pred, y_prob = self.predict(test_texts)
        y_true = np.array(test_labels)
        
        # Calculate overall metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        # Calculate per-class metrics
        per_class_metrics = self.calculate_per_class_metrics(y_true, y_pred)
        
        # Generate classification report
        class_report = classification_report(
            y_true, y_pred, 
            target_names=self.label_names,
            output_dict=True
        )
        
        # Create visualizations
        cm_fig = self.plot_confusion_matrix(
            y_true, y_pred, 
            save_path=os.path.join(output_dir, "confusion_matrix.png")
        )
        
        ovr_cm_fig = self.plot_one_vs_rest_confusion_matrices(
            y_true, y_pred,
            save_path=os.path.join(output_dir, "one_vs_rest_confusion_matrices.png")
        )
        
        roc_fig = self.plot_roc_curves(
            y_true, y_prob,
            save_path=os.path.join(output_dir, "roc_curves.png")
        )
        
        # Compile results
        results = {
            "overall_metrics": {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1)
            },
            "per_class_metrics": per_class_metrics,
            "classification_report": class_report,
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "predictions": y_pred.tolist(),
            "probabilities": y_prob.tolist(),
            "true_labels": y_true.tolist()
        }
        
        # Save results to JSON
        results_path = os.path.join(output_dir, "evaluation_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation completed. Results saved to {output_dir}")
        logger.info(f"Overall Accuracy: {accuracy:.4f}")
        logger.info(f"Overall F1 Score: {f1:.4f}")
        
        return results

def main():
    """
    Example usage of the ModelEvaluator class.
    """
    # This would typically use a real trained model
    # For demo purposes, we'll create sample data
    
    sample_texts = [
        "Great restaurant with excellent food!",
        "Terrible service, would not recommend.",
        "Average place, nothing special.",
        "Amazing experience, will definitely come back!",
        "Poor quality food and rude staff.",
        "Decent restaurant with good value for money."
    ]
    sample_labels = [3, 2, 3, 3, 2, 3]  # Corresponding to label indices
    
    try:
        # Note: This would fail without a real model, but shows the usage pattern
        evaluator = ModelEvaluator("./demo_model")
        results = evaluator.evaluate_model(sample_texts, sample_labels)
        print("Evaluation completed successfully!")
        print(f"Accuracy: {results['overall_metrics']['accuracy']:.4f}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print("This demo requires a trained model. Please train a model first.")

if __name__ == "__main__":
    main()

