"""
Active Learning Module

This module implements active learning strategies for improving model performance
with fewer labeled examples. Based on the active_learning.ipynb notebook.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Callable
import logging
from sklearn.metrics import accuracy_score
from scipy.stats import entropy
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ActiveLearningStrategy:
    """
    Base class for active learning strategies.
    """
    
    def __init__(self, name: str):
        self.name = name
    
    def select_samples(self, model, unlabeled_data: List[str], 
                      n_samples: int, **kwargs) -> List[int]:
        """
        Select samples for labeling.
        
        Args:
            model: Trained model
            unlabeled_data: List of unlabeled texts
            n_samples: Number of samples to select
            **kwargs: Additional arguments
            
        Returns:
            List of indices of selected samples
        """
        raise NotImplementedError

class UncertaintyStrategy(ActiveLearningStrategy):
    """
    Uncertainty-based sampling strategy.
    """
    
    def __init__(self, method: str = "entropy"):
        super().__init__(f"uncertainty_{method}")
        self.method = method
    
    def select_samples(self, model, unlabeled_data: List[str], 
                      n_samples: int, **kwargs) -> List[int]:
        """
        Select samples based on prediction uncertainty.
        
        Args:
            model: Trained model
            unlabeled_data: List of unlabeled texts
            n_samples: Number of samples to select
            
        Returns:
            List of indices of selected samples
        """
        # Get predictions and probabilities
        predictions, probabilities = model.predict(unlabeled_data)
        
        if self.method == "entropy":
            # Calculate entropy for each prediction
            uncertainties = [entropy(prob) for prob in probabilities]
        elif self.method == "least_confident":
            # Use 1 - max probability as uncertainty
            uncertainties = [1 - np.max(prob) for prob in probabilities]
        elif self.method == "margin":
            # Use margin between top two predictions
            sorted_probs = np.sort(probabilities, axis=1)
            uncertainties = [sorted_probs[i, -1] - sorted_probs[i, -2] 
                           for i in range(len(sorted_probs))]
            uncertainties = [1 - margin for margin in uncertainties]  # Higher uncertainty = lower margin
        else:
            raise ValueError(f"Unknown uncertainty method: {self.method}")
        
        # Select samples with highest uncertainty
        selected_indices = np.argsort(uncertainties)[-n_samples:].tolist()
        
        return selected_indices

class DiversityStrategy(ActiveLearningStrategy):
    """
    Diversity-based sampling strategy.
    """
    
    def __init__(self, method: str = "kmeans"):
        super().__init__(f"diversity_{method}")
        self.method = method
    
    def select_samples(self, model, unlabeled_data: List[str], 
                      n_samples: int, **kwargs) -> List[int]:
        """
        Select diverse samples.
        
        Args:
            model: Trained model
            unlabeled_data: List of unlabeled texts
            n_samples: Number of samples to select
            
        Returns:
            List of indices of selected samples
        """
        # For simplicity, we'll use random sampling as a placeholder
        # In practice, you would extract features and use clustering
        indices = np.random.choice(len(unlabeled_data), n_samples, replace=False)
        return indices.tolist()

class HybridStrategy(ActiveLearningStrategy):
    """
    Hybrid strategy combining uncertainty and diversity.
    """
    
    def __init__(self, uncertainty_weight: float = 0.7):
        super().__init__("hybrid")
        self.uncertainty_weight = uncertainty_weight
        self.uncertainty_strategy = UncertaintyStrategy("entropy")
        self.diversity_strategy = DiversityStrategy("kmeans")
    
    def select_samples(self, model, unlabeled_data: List[str], 
                      n_samples: int, **kwargs) -> List[int]:
        """
        Select samples using hybrid approach.
        
        Args:
            model: Trained model
            unlabeled_data: List of unlabeled texts
            n_samples: Number of samples to select
            
        Returns:
            List of indices of selected samples
        """
        # Select more samples than needed from each strategy
        uncertainty_samples = int(n_samples * self.uncertainty_weight * 1.5)
        diversity_samples = int(n_samples * (1 - self.uncertainty_weight) * 1.5)
        
        uncertainty_indices = self.uncertainty_strategy.select_samples(
            model, unlabeled_data, uncertainty_samples
        )
        diversity_indices = self.diversity_strategy.select_samples(
            model, unlabeled_data, diversity_samples
        )
        
        # Combine and select final samples
        combined_indices = list(set(uncertainty_indices + diversity_indices))
        
        if len(combined_indices) > n_samples:
            # Prioritize uncertainty samples
            final_indices = uncertainty_indices[:n_samples//2] + diversity_indices[:n_samples//2]
            if len(final_indices) < n_samples:
                remaining = n_samples - len(final_indices)
                additional = [idx for idx in combined_indices if idx not in final_indices][:remaining]
                final_indices.extend(additional)
        else:
            final_indices = combined_indices
        
        return final_indices[:n_samples]

class ActiveLearner:
    """
    Main active learning class that orchestrates the learning process.
    """
    
    def __init__(self, model, strategy: ActiveLearningStrategy, 
                 retrain_func: Callable, evaluate_func: Callable):
        """
        Initialize the ActiveLearner.
        
        Args:
            model: Initial trained model
            strategy: Active learning strategy
            retrain_func: Function to retrain the model
            evaluate_func: Function to evaluate the model
        """
        self.model = model
        self.strategy = strategy
        self.retrain_func = retrain_func
        self.evaluate_func = evaluate_func
        self.learning_history = []
    
    def active_learning_loop(self, labeled_data: pd.DataFrame, 
                           unlabeled_data: pd.DataFrame,
                           test_data: pd.DataFrame,
                           n_iterations: int = 5,
                           samples_per_iteration: int = 100,
                           text_column: str = 'text',
                           label_column: str = 'label') -> Dict:
        """
        Run the active learning loop.
        
        Args:
            labeled_data: Initially labeled data
            unlabeled_data: Unlabeled data pool
            test_data: Test data for evaluation
            n_iterations: Number of active learning iterations
            samples_per_iteration: Number of samples to label per iteration
            text_column: Name of text column
            label_column: Name of label column
            
        Returns:
            Dictionary containing learning history and final results
        """
        logger.info(f"Starting active learning with {self.strategy.name} strategy")
        
        current_labeled = labeled_data.copy()
        current_unlabeled = unlabeled_data.copy()
        
        # Initial evaluation
        initial_score = self.evaluate_func(self.model, test_data)
        self.learning_history.append({
            'iteration': 0,
            'labeled_samples': len(current_labeled),
            'accuracy': initial_score,
            'strategy': self.strategy.name
        })
        
        logger.info(f"Initial accuracy: {initial_score:.4f} with {len(current_labeled)} labeled samples")
        
        for iteration in range(1, n_iterations + 1):
            logger.info(f"Active learning iteration {iteration}/{n_iterations}")
            
            # Check if we have enough unlabeled data
            if len(current_unlabeled) < samples_per_iteration:
                logger.warning(f"Not enough unlabeled data. Only {len(current_unlabeled)} samples remaining.")
                samples_per_iteration = len(current_unlabeled)
                if samples_per_iteration == 0:
                    break
            
            # Select samples using the strategy
            unlabeled_texts = current_unlabeled[text_column].tolist()
            selected_indices = self.strategy.select_samples(
                self.model, unlabeled_texts, samples_per_iteration
            )
            
            # Move selected samples from unlabeled to labeled
            selected_samples = current_unlabeled.iloc[selected_indices].copy()
            current_labeled = pd.concat([current_labeled, selected_samples], ignore_index=True)
            current_unlabeled = current_unlabeled.drop(current_unlabeled.index[selected_indices]).reset_index(drop=True)
            
            # Retrain the model
            logger.info(f"Retraining model with {len(current_labeled)} labeled samples")
            self.model = self.retrain_func(current_labeled)
            
            # Evaluate the model
            accuracy = self.evaluate_func(self.model, test_data)
            
            # Record progress
            self.learning_history.append({
                'iteration': iteration,
                'labeled_samples': len(current_labeled),
                'accuracy': accuracy,
                'strategy': self.strategy.name
            })
            
            logger.info(f"Iteration {iteration}: Accuracy = {accuracy:.4f}, Labeled samples = {len(current_labeled)}")
        
        # Compile final results
        results = {
            'strategy': self.strategy.name,
            'learning_history': self.learning_history,
            'final_accuracy': self.learning_history[-1]['accuracy'],
            'total_labeled_samples': self.learning_history[-1]['labeled_samples'],
            'improvement': self.learning_history[-1]['accuracy'] - self.learning_history[0]['accuracy']
        }
        
        logger.info(f"Active learning completed. Final accuracy: {results['final_accuracy']:.4f}")
        logger.info(f"Improvement: {results['improvement']:.4f}")
        
        return results
    
    def compare_strategies(self, strategies: List[ActiveLearningStrategy],
                          labeled_data: pd.DataFrame,
                          unlabeled_data: pd.DataFrame,
                          test_data: pd.DataFrame,
                          **kwargs) -> Dict:
        """
        Compare multiple active learning strategies.
        
        Args:
            strategies: List of strategies to compare
            labeled_data: Initially labeled data
            unlabeled_data: Unlabeled data pool
            test_data: Test data for evaluation
            **kwargs: Additional arguments for active_learning_loop
            
        Returns:
            Dictionary containing comparison results
        """
        logger.info(f"Comparing {len(strategies)} active learning strategies")
        
        comparison_results = {}
        
        for strategy in strategies:
            logger.info(f"Testing strategy: {strategy.name}")
            
            # Reset the learner with the new strategy
            self.strategy = strategy
            self.learning_history = []
            
            # Run active learning
            results = self.active_learning_loop(
                labeled_data.copy(), 
                unlabeled_data.copy(), 
                test_data.copy(),
                **kwargs
            )
            
            comparison_results[strategy.name] = results
        
        # Find best strategy
        best_strategy = max(comparison_results.keys(), 
                          key=lambda x: comparison_results[x]['final_accuracy'])
        
        logger.info(f"Best strategy: {best_strategy} with accuracy {comparison_results[best_strategy]['final_accuracy']:.4f}")
        
        return {
            'strategies': comparison_results,
            'best_strategy': best_strategy,
            'best_accuracy': comparison_results[best_strategy]['final_accuracy']
        }

def main():
    """
    Example usage of the ActiveLearner class.
    """
    # This is a simplified example - in practice you would use real data and models
    
    # Create sample data
    sample_labeled = pd.DataFrame({
        'text': ["Great food!", "Terrible service.", "Average place."],
        'label': [3, 2, 3]
    })
    
    sample_unlabeled = pd.DataFrame({
        'text': ["Amazing restaurant!", "Poor quality.", "Decent food.", "Excellent service!"],
        'label': [3, 2, 3, 3]  # In practice, these wouldn't be available
    })
    
    sample_test = pd.DataFrame({
        'text': ["Good experience.", "Bad food."],
        'label': [3, 2]
    })
    
    # Mock functions (in practice, these would be real implementations)
    def mock_retrain(data):
        logger.info(f"Mock retraining with {len(data)} samples")
        return "mock_model"
    
    def mock_evaluate(model, test_data):
        # Return a mock accuracy score
        return np.random.uniform(0.7, 0.9)
    
    # Create strategies
    strategies = [
        UncertaintyStrategy("entropy"),
        UncertaintyStrategy("least_confident"),
        DiversityStrategy("kmeans"),
        HybridStrategy(0.7)
    ]
    
    # Initialize active learner
    learner = ActiveLearner(
        model="mock_model",
        strategy=strategies[0],
        retrain_func=mock_retrain,
        evaluate_func=mock_evaluate
    )
    
    try:
        # Compare strategies
        results = learner.compare_strategies(
            strategies,
            sample_labeled,
            sample_unlabeled,
            sample_test,
            n_iterations=2,
            samples_per_iteration=1
        )
        
        print("Strategy comparison completed!")
        print(f"Best strategy: {results['best_strategy']}")
        print(f"Best accuracy: {results['best_accuracy']:.4f}")
        
    except Exception as e:
        logger.error(f"Active learning failed: {e}")

if __name__ == "__main__":
    main()

