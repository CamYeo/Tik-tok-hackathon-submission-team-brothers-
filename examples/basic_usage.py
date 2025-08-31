#!/usr/bin/env python3
"""
Basic Usage Example for TikTok Hackathon Review Quality Assessment

This script demonstrates basic usage of the review quality assessment system
developed for the TikTok Hackathon.
"""

import os
import sys
import json
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.main import ReviewQualityAssessment
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.policy_enforcement import PolicyEnforcer

def create_sample_data():
    """Create sample review data for demonstration."""
    sample_reviews = [
        {
            "text": "Amazing content creator! Love their videos and creativity. Highly recommend following!",
            "rating": 5,
            "time": 1620085852324,
            "user_id": "user123",
            "content_id": "video456"
        },
        {
            "text": "Average content, nothing special. Could be better with more effort.",
            "rating": 3,
            "time": 1620085852325,
            "user_id": "user456",
            "content_id": "video789"
        },
        {
            "text": "Terrible content! This creator is overrated and boring. Waste of time.",
            "rating": 1,
            "time": 1620085852326,
            "user_id": "user789",
            "content_id": "video123"
        },
        {
            "text": "Incredible talent and amazing production quality! Keep up the great work!",
            "rating": 5,
            "time": 1620085852327,
            "user_id": "user321",
            "content_id": "video654"
        },
        {
            "text": "Buy our amazing products at 50% discount! Follow our page for more deals!",
            "rating": 5,
            "time": 1620085852328,
            "user_id": "spammer1",
            "content_id": "video987"
        },
        {
            "text": "This review is completely unrelated to the content. Just testing the system.",
            "rating": 3,
            "time": 1620085852329,
            "user_id": "tester1",
            "content_id": "video111"
        }
    ]
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Save sample data
    with open("data/sample_reviews.json", "w") as f:
        for review in sample_reviews:
            f.write(json.dumps(review) + "\n")
    
    print("Sample data created at data/sample_reviews.json")
    return "data/sample_reviews.json"

def example_1_full_pipeline():
    """Example 1: Run the complete pipeline."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Full Pipeline - TikTok Review Quality Assessment")
    print("="*60)
    
    # Create sample data
    input_file = create_sample_data()
    
    # Initialize system
    system = ReviewQualityAssessment("configs/config.json")
    
    try:
        # Run full pipeline
        outputs = system.run_full_pipeline(input_file, "results/example1")
        
        print("\nPipeline completed successfully!")
        print("Output files:")
        for component, path in outputs.items():
            print(f"  {component}: {path}")
            
    except Exception as e:
        print(f"Pipeline failed: {e}")

def example_2_individual_components():
    """Example 2: Use individual components for TikTok content analysis."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Individual Components - TikTok Content Analysis")
    print("="*60)
    
    # Create sample data
    input_file = create_sample_data()
    
    try:
        # Step 1: Data preprocessing
        print("\n1. Data Preprocessing...")
        preprocessor = DataPreprocessor({"sample_size": 10})
        df = preprocessor.preprocess(input_file)
        print(f"   Preprocessed {len(df)} reviews")
        print(f"   Columns: {list(df.columns)}")
        
        # Step 2: Feature engineering
        print("\n2. Feature Engineering...")
        engineer = FeatureEngineer({
            "num_topics_lda": 3,
            "sentiment_model": "cardiffnlp/twitter-roberta-base-sentiment-latest"
        })
        features_df = engineer.engineer_features(df)
        print(f"   Engineered {len(features_df.columns)} features")
        
        # Display some key features
        key_features = ['text_length', 'word_count', 'sentiment_score', 'rating']
        if all(col in features_df.columns for col in key_features):
            print("\n   Sample feature values:")
            for idx, row in features_df[key_features].head(3).iterrows():
                print(f"     Review {idx+1}: Length={row['text_length']}, Words={row['word_count']}, "
                      f"Sentiment={row['sentiment_score']:.3f}, Rating={row['rating']}")
        
        # Step 3: Policy enforcement
        print("\n3. Policy Enforcement...")
        enforcer = PolicyEnforcer(use_ml=False)  # Use rule-based only for demo
        
        sample_texts = df['text'].tolist()
        results = enforcer.batch_enforce_policies(sample_texts)
        
        print("   Policy enforcement results:")
        for i, result in enumerate(results):
            text_preview = result['text'][:50] + "..." if len(result['text']) > 50 else result['text']
            print(f"     Review {i+1}: {text_preview}")
            print(f"     Risk Score: {result['overall_risk_score']:.3f}")
            print(f"     Action: {result['action_recommended']}")
            if result['violations']:
                print(f"     Violations: {[v.violation_type.value for v in result['violations']]}")
            print()
        
        # Generate report
        report = enforcer.generate_policy_report(results)
        print("   Summary Report:")
        print(f"     Total reviews: {report['summary']['total_texts_analyzed']}")
        print(f"     Reviews with violations: {report['summary']['texts_with_violations']}")
        print(f"     Action distribution: {report['action_distribution']}")
        
    except Exception as e:
        print(f"Component processing failed: {e}")

def example_3_tiktok_specific_analysis():
    """Example 3: TikTok-specific content quality analysis."""
    print("\n" + "="*60)
    print("EXAMPLE 3: TikTok-Specific Content Quality Analysis")
    print("="*60)
    
    # TikTok-specific test content
    tiktok_content = [
        "This dance trend is so cool! #fyp #viral #dance",  # Trending content
        "Check out my new product! Link in bio! #ad #sponsored",  # Advertisement
        "This creator is trash and should quit making content",  # Hate/inappropriate
        "Love this song choice! Perfect for the vibe ✨",  # Positive engagement
        "First! Like if you agree! Subscribe for more!",  # Engagement farming
        "Amazing editing skills! How did you do this effect?",  # Genuine interest
        "This is fake and staged. Unfollow this account.",  # Negative/fake claim
        "Tutorial please! This looks so easy to follow 🙏"  # Educational request
    ]
    
    # Initialize policy enforcer
    enforcer = PolicyEnforcer(use_ml=False)  # Rule-based for clear demonstration
    
    print("\nAnalyzing TikTok content quality:")
    print("-" * 40)
    
    content_categories = {
        "high_quality": [],
        "moderate_quality": [],
        "low_quality": [],
        "policy_violations": []
    }
    
    for i, text in enumerate(tiktok_content, 1):
        print(f"\nContent {i}: {text}")
        result = enforcer.enforce_policies(text)
        
        risk_score = result['overall_risk_score']
        action = result['action_recommended']
        
        print(f"Risk Score: {risk_score:.3f}")
        print(f"Action: {action}")
        
        # Categorize content
        if action == 'approve' and risk_score < 0.2:
            content_categories["high_quality"].append(i)
        elif action == 'approve':
            content_categories["moderate_quality"].append(i)
        elif action == 'review':
            content_categories["low_quality"].append(i)
        else:  # reject
            content_categories["policy_violations"].append(i)
        
        if result['violations']:
            print("Violations detected:")
            for violation in result['violations']:
                print(f"  - {violation.violation_type.value}: {violation.severity} severity")
                print(f"    Confidence: {violation.confidence:.3f}")
        else:
            print("No violations detected")
    
    # Summary
    print("\n" + "="*40)
    print("CONTENT QUALITY SUMMARY")
    print("="*40)
    print(f"High Quality Content: {len(content_categories['high_quality'])} items")
    print(f"Moderate Quality Content: {len(content_categories['moderate_quality'])} items")
    print(f"Low Quality Content: {len(content_categories['low_quality'])} items")
    print(f"Policy Violations: {len(content_categories['policy_violations'])} items")

def example_4_model_performance():
    """Example 4: Demonstrate model performance with real data."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Model Performance Analysis")
    print("="*60)
    
    # Check if we have real data available
    real_data_path = "data/review-Wyoming_10.json"
    
    if os.path.exists(real_data_path):
        print(f"Using real data from: {real_data_path}")
        
        try:
            # Load a small sample of real data
            with open(real_data_path, 'r') as f:
                real_reviews = []
                for i, line in enumerate(f):
                    if i >= 5:  # Limit to 5 reviews for demo
                        break
                    real_reviews.append(json.loads(line))
            
            print(f"Loaded {len(real_reviews)} real reviews for analysis")
            
            # Analyze real reviews
            enforcer = PolicyEnforcer(use_ml=False)
            
            for i, review in enumerate(real_reviews, 1):
                text = review.get('text', '')[:100] + "..." if len(review.get('text', '')) > 100 else review.get('text', '')
                rating = review.get('rating', 'N/A')
                
                result = enforcer.enforce_policies(review.get('text', ''))
                
                print(f"\nReal Review {i}:")
                print(f"  Text: {text}")
                print(f"  Rating: {rating}")
                print(f"  Risk Score: {result['overall_risk_score']:.3f}")
                print(f"  Recommended Action: {result['action_recommended']}")
                
        except Exception as e:
            print(f"Error processing real data: {e}")
    else:
        print(f"Real data file not found at {real_data_path}")
        print("Using sample data instead...")
        example_1_full_pipeline()

def main():
    """Run all examples."""
    print("TikTok Hackathon Review Quality Assessment - Usage Examples")
    print("=" * 60)
    
    # Create output directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Run examples
    try:
        example_1_full_pipeline()
        example_2_individual_components()
        example_3_tiktok_specific_analysis()
        example_4_model_performance()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("Check the 'results/' directory for output files.")
        print("="*60)
        
    except Exception as e:
        print(f"\nExample execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

