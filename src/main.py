"""
Main Entry Point for Google Review Quality Assessment

This module provides the main entry point and command-line interface
for the Google Review Quality Assessment system.
"""

import argparse
import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer, ModelConfig
from src.model_evaluation import ModelEvaluator
from src.active_learning import ActiveLearner, UncertaintyStrategy
from src.labeling_packaging import MultiLLMLabeler
from src.policy_enforcement import PolicyEnforcer
from src.utils import ConfigManager, DataManager, Logger

# Configure logging
logger = Logger.setup_logger(__name__, "logs/main.log")

class ReviewQualityAssessment:
    """
    Main class for the Google Review Quality Assessment system.
    """
    
    def __init__(self, config_path: str = "configs/config.json"):
        """
        Initialize the system.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        
        # Initialize components
        self.data_preprocessor = None
        self.feature_engineer = None
        self.model_trainer = None
        self.model_evaluator = None
        self.policy_enforcer = None
        
        logger.info("Review Quality Assessment system initialized")
    
    def preprocess_data(self, input_file: str, output_file: str = None) -> str:
        """
        Preprocess raw review data.
        
        Args:
            input_file: Path to input data file
            output_file: Path to save preprocessed data
            
        Returns:
            Path to preprocessed data file
        """
        logger.info("Starting data preprocessing...")
        
        if not self.data_preprocessor:
            self.data_preprocessor = DataPreprocessor(self.config.get('data', {}))
        
        # Preprocess data
        df = self.data_preprocessor.preprocess(input_file)
        
        # Save preprocessed data
        if not output_file:
            output_file = os.path.join(
                self.config.get('data.output_dir', 'data'),
                'preprocessed_data.pkl'
            )
        
        DataManager.save_data(df, output_file)
        logger.info(f"Preprocessed data saved to {output_file}")
        
        return output_file
    
    def engineer_features(self, input_file: str, output_file: str = None) -> str:
        """
        Engineer features from preprocessed data.
        
        Args:
            input_file: Path to preprocessed data
            output_file: Path to save engineered features
            
        Returns:
            Path to engineered features file
        """
        logger.info("Starting feature engineering...")
        
        if not self.feature_engineer:
            self.feature_engineer = FeatureEngineer(self.config.get('feature_engineering', {}))
        
        # Load data
        df = DataManager.load_data(input_file)
        
        # Engineer features
        df_engineered = self.feature_engineer.engineer_features(df)
        
        # Save engineered features
        if not output_file:
            output_file = os.path.join(
                self.config.get('data.output_dir', 'data'),
                'engineered_features.pkl'
            )
        
        DataManager.save_data(df_engineered, output_file)
        logger.info(f"Engineered features saved to {output_file}")
        
        return output_file
    
    def train_model(self, input_file: str, output_dir: str = None) -> str:
        """
        Train the quality assessment model.
        
        Args:
            input_file: Path to engineered features
            output_dir: Directory to save trained model
            
        Returns:
            Path to trained model directory
        """
        logger.info("Starting model training...")
        
        # Load data
        df = DataManager.load_data(input_file)
        
        # Create model config
        model_config = ModelConfig(**self.config.get('model', {}))
        if output_dir:
            model_config.output_dir = output_dir
        
        # Initialize trainer
        if not self.model_trainer:
            self.model_trainer = ModelTrainer(model_config)
        
        # Prepare datasets
        train_dataset, val_dataset, test_dataset = self.model_trainer.prepare_data(
            df, text_column='text', label_column='consensus_label'
        )
        
        # Train model
        train_result = self.model_trainer.train(train_dataset, val_dataset)
        
        # Evaluate model
        eval_result = self.model_trainer.evaluate(test_dataset)
        
        # Save training metadata
        self.model_trainer.save_training_metadata(
            train_result, eval_result,
            os.path.join(model_config.output_dir, "training_metadata.json")
        )
        
        logger.info(f"Model training completed. Model saved to {model_config.output_dir}")
        
        return model_config.output_dir
    
    def evaluate_model(self, model_path: str, test_data_file: str, 
                      output_dir: str = "evaluation_results") -> Dict[str, Any]:
        """
        Evaluate a trained model.
        
        Args:
            model_path: Path to trained model
            test_data_file: Path to test data
            output_dir: Directory to save evaluation results
            
        Returns:
            Evaluation results dictionary
        """
        logger.info("Starting model evaluation...")
        
        # Initialize evaluator
        if not self.model_evaluator:
            self.model_evaluator = ModelEvaluator(model_path)
        
        # Load test data
        test_df = DataManager.load_data(test_data_file)
        test_texts = test_df['text'].tolist()
        test_labels = test_df['consensus_label'].map({
            'advertisement': 0,
            'irrelevant': 1,
            'rant_without_visit': 2,
            'relevant_and_quality': 3
        }).tolist()
        
        # Evaluate model
        results = self.model_evaluator.evaluate_model(test_texts, test_labels, output_dir)
        
        logger.info(f"Model evaluation completed. Results saved to {output_dir}")
        
        return results
    
    def enforce_policies(self, input_file: str, output_file: str = None) -> str:
        """
        Enforce policies on review data.
        
        Args:
            input_file: Path to review data
            output_file: Path to save policy enforcement results
            
        Returns:
            Path to policy enforcement results
        """
        logger.info("Starting policy enforcement...")
        
        # Initialize policy enforcer
        if not self.policy_enforcer:
            policy_config = self.config.get('policy_enforcement', {})
            self.policy_enforcer = PolicyEnforcer(
                use_ml=policy_config.get('use_ml', True)
            )
        
        # Load data
        df = DataManager.load_data(input_file)
        texts = df['text'].tolist()
        
        # Enforce policies
        results = self.policy_enforcer.batch_enforce_policies(texts)
        
        # Create results DataFrame
        results_df = df.copy()
        results_df['policy_violations'] = [r['violations'] for r in results]
        results_df['risk_score'] = [r['overall_risk_score'] for r in results]
        results_df['action_recommended'] = [r['action_recommended'] for r in results]
        
        # Save results
        if not output_file:
            output_file = os.path.join(
                self.config.get('data.output_dir', 'data'),
                'policy_enforcement_results.pkl'
            )
        
        DataManager.save_data(results_df, output_file)
        
        # Generate and save report
        report = self.policy_enforcer.generate_policy_report(results)
        report_file = output_file.replace('.pkl', '_report.json')
        with open(report_file, 'w') as f:
            import json
            json.dump(report, f, indent=2)
        
        logger.info(f"Policy enforcement completed. Results saved to {output_file}")
        
        return output_file
    
    def run_full_pipeline(self, input_file: str, output_dir: str = "output") -> Dict[str, str]:
        """
        Run the complete pipeline from data preprocessing to model evaluation.
        
        Args:
            input_file: Path to raw input data
            output_dir: Directory to save all outputs
            
        Returns:
            Dictionary with paths to all output files
        """
        logger.info("Starting full pipeline...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        outputs = {}
        
        try:
            # Step 1: Data preprocessing
            preprocessed_file = self.preprocess_data(
                input_file, 
                os.path.join(output_dir, "preprocessed_data.pkl")
            )
            outputs['preprocessed_data'] = preprocessed_file
            
            # Step 2: Feature engineering
            features_file = self.engineer_features(
                preprocessed_file,
                os.path.join(output_dir, "engineered_features.pkl")
            )
            outputs['engineered_features'] = features_file
            
            # Step 3: Model training (if labels are available)
            if 'consensus_label' in DataManager.load_data(features_file).columns:
                model_dir = self.train_model(
                    features_file,
                    os.path.join(output_dir, "trained_model")
                )
                outputs['trained_model'] = model_dir
                
                # Step 4: Model evaluation
                eval_results_dir = os.path.join(output_dir, "evaluation_results")
                eval_results = self.evaluate_model(model_dir, features_file, eval_results_dir)
                outputs['evaluation_results'] = eval_results_dir
            
            # Step 5: Policy enforcement
            policy_results = self.enforce_policies(
                features_file,
                os.path.join(output_dir, "policy_enforcement_results.pkl")
            )
            outputs['policy_enforcement'] = policy_results
            
            logger.info("Full pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
        
        return outputs

def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Google Review Quality Assessment System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python -m src.main --mode full --input data/reviews.json --output results/

  # Preprocess data only
  python -m src.main --mode preprocess --input data/reviews.json --output data/preprocessed.pkl

  # Train model
  python -m src.main --mode train --input data/features.pkl --output models/

  # Evaluate model
  python -m src.main --mode evaluate --model models/ --test-data data/test.pkl

  # Enforce policies
  python -m src.main --mode policy --input data/reviews.json --output results/policy.pkl
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["full", "preprocess", "features", "train", "evaluate", "policy"],
        required=True,
        help="Mode of operation"
    )
    
    parser.add_argument(
        "--input",
        required=True,
        help="Input file path"
    )
    
    parser.add_argument(
        "--output",
        help="Output file/directory path"
    )
    
    parser.add_argument(
        "--config",
        default="configs/config.json",
        help="Configuration file path"
    )
    
    parser.add_argument(
        "--model",
        help="Model path (for evaluation mode)"
    )
    
    parser.add_argument(
        "--test-data",
        help="Test data path (for evaluation mode)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser

def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize system
        system = ReviewQualityAssessment(args.config)
        
        # Execute based on mode
        if args.mode == "full":
            outputs = system.run_full_pipeline(args.input, args.output or "output")
            print("Pipeline completed successfully!")
            print("Output files:")
            for key, path in outputs.items():
                print(f"  {key}: {path}")
        
        elif args.mode == "preprocess":
            output_file = system.preprocess_data(args.input, args.output)
            print(f"Data preprocessing completed: {output_file}")
        
        elif args.mode == "features":
            output_file = system.engineer_features(args.input, args.output)
            print(f"Feature engineering completed: {output_file}")
        
        elif args.mode == "train":
            model_dir = system.train_model(args.input, args.output)
            print(f"Model training completed: {model_dir}")
        
        elif args.mode == "evaluate":
            if not args.model or not args.test_data:
                parser.error("--model and --test-data are required for evaluation mode")
            
            results = system.evaluate_model(args.model, args.test_data, args.output or "evaluation_results")
            print(f"Model evaluation completed: {args.output or 'evaluation_results'}")
            print(f"Accuracy: {results['overall_metrics']['accuracy']:.4f}")
        
        elif args.mode == "policy":
            output_file = system.enforce_policies(args.input, args.output)
            print(f"Policy enforcement completed: {output_file}")
        
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

