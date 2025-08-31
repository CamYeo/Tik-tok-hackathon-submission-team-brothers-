"""
Policy Enforcement Module

This module implements policy enforcement for Google location review quality assessment.
It provides rule-based and ML-based policy checking capabilities.
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PolicyViolationType(Enum):
    """Enumeration of policy violation types."""
    SPAM = "spam"
    ADVERTISEMENT = "advertisement"
    IRRELEVANT = "irrelevant"
    INAPPROPRIATE_CONTENT = "inappropriate_content"
    FAKE_REVIEW = "fake_review"
    PERSONAL_ATTACK = "personal_attack"
    OFF_TOPIC = "off_topic"
    PROMOTIONAL = "promotional"

@dataclass
class PolicyViolation:
    """Represents a policy violation."""
    violation_type: PolicyViolationType
    severity: str  # "low", "medium", "high"
    confidence: float
    description: str
    evidence: List[str]

class RuleBasedPolicyChecker:
    """
    Rule-based policy checker using pattern matching and heuristics.
    """
    
    def __init__(self):
        """Initialize the rule-based policy checker."""
        self.spam_patterns = [
            r'\b(buy|sale|discount|promotion|deal|offer|free|cheap)\b',
            r'\b(click|visit|website|link|url)\b',
            r'\b(call|phone|contact|email)\b.*\d{3}[-.\s]?\d{3}[-.\s]?\d{4}',
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        ]
        
        self.advertisement_patterns = [
            r'\b(advertise|marketing|promote|sponsor|brand)\b',
            r'\b(special offer|limited time|act now|don\'t miss)\b',
            r'\b(best price|lowest price|guaranteed|money back)\b'
        ]
        
        self.inappropriate_patterns = [
            r'\b(hate|racist|sexist|offensive|inappropriate)\b',
            r'\b(stupid|idiot|moron|dumb)\b',
            r'\b(scam|fraud|cheat|steal)\b'
        ]
        
        self.fake_review_indicators = [
            r'\b(never been|never visited|never went)\b',
            r'\b(fake|false|made up|fabricated)\b',
            r'\b(competitor|rival|enemy)\b'
        ]
        
        self.promotional_patterns = [
            r'\b(check out|follow us|like us|subscribe)\b',
            r'\b(social media|facebook|instagram|twitter)\b',
            r'\b(coupon|voucher|promo code)\b'
        ]
    
    def check_spam_patterns(self, text: str) -> List[PolicyViolation]:
        """Check for spam patterns in text."""
        violations = []
        text_lower = text.lower()
        
        spam_matches = []
        for pattern in self.spam_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            spam_matches.extend(matches)
        
        if spam_matches:
            severity = "high" if len(spam_matches) > 2 else "medium"
            confidence = min(0.9, 0.3 + 0.2 * len(spam_matches))
            
            violation = PolicyViolation(
                violation_type=PolicyViolationType.SPAM,
                severity=severity,
                confidence=confidence,
                description=f"Contains {len(spam_matches)} spam indicators",
                evidence=spam_matches[:5]  # Limit evidence to first 5 matches
            )
            violations.append(violation)
        
        return violations
    
    def check_advertisement_patterns(self, text: str) -> List[PolicyViolation]:
        """Check for advertisement patterns in text."""
        violations = []
        text_lower = text.lower()
        
        ad_matches = []
        for pattern in self.advertisement_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            ad_matches.extend(matches)
        
        if ad_matches:
            severity = "high" if len(ad_matches) > 1 else "medium"
            confidence = min(0.8, 0.4 + 0.2 * len(ad_matches))
            
            violation = PolicyViolation(
                violation_type=PolicyViolationType.ADVERTISEMENT,
                severity=severity,
                confidence=confidence,
                description=f"Contains {len(ad_matches)} advertisement indicators",
                evidence=ad_matches[:5]
            )
            violations.append(violation)
        
        return violations
    
    def check_inappropriate_content(self, text: str) -> List[PolicyViolation]:
        """Check for inappropriate content patterns."""
        violations = []
        text_lower = text.lower()
        
        inappropriate_matches = []
        for pattern in self.inappropriate_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            inappropriate_matches.extend(matches)
        
        if inappropriate_matches:
            severity = "high"
            confidence = min(0.9, 0.5 + 0.1 * len(inappropriate_matches))
            
            violation = PolicyViolation(
                violation_type=PolicyViolationType.INAPPROPRIATE_CONTENT,
                severity=severity,
                confidence=confidence,
                description=f"Contains {len(inappropriate_matches)} inappropriate content indicators",
                evidence=inappropriate_matches[:5]
            )
            violations.append(violation)
        
        return violations
    
    def check_fake_review_indicators(self, text: str) -> List[PolicyViolation]:
        """Check for fake review indicators."""
        violations = []
        text_lower = text.lower()
        
        fake_matches = []
        for pattern in self.fake_review_indicators:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            fake_matches.extend(matches)
        
        if fake_matches:
            severity = "high"
            confidence = min(0.8, 0.4 + 0.2 * len(fake_matches))
            
            violation = PolicyViolation(
                violation_type=PolicyViolationType.FAKE_REVIEW,
                severity=severity,
                confidence=confidence,
                description=f"Contains {len(fake_matches)} fake review indicators",
                evidence=fake_matches[:5]
            )
            violations.append(violation)
        
        return violations
    
    def check_promotional_content(self, text: str) -> List[PolicyViolation]:
        """Check for promotional content patterns."""
        violations = []
        text_lower = text.lower()
        
        promo_matches = []
        for pattern in self.promotional_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            promo_matches.extend(matches)
        
        if promo_matches:
            severity = "medium"
            confidence = min(0.7, 0.3 + 0.2 * len(promo_matches))
            
            violation = PolicyViolation(
                violation_type=PolicyViolationType.PROMOTIONAL,
                severity=severity,
                confidence=confidence,
                description=f"Contains {len(promo_matches)} promotional indicators",
                evidence=promo_matches[:5]
            )
            violations.append(violation)
        
        return violations
    
    def check_all_policies(self, text: str) -> List[PolicyViolation]:
        """Check all policy violations for a given text."""
        all_violations = []
        
        all_violations.extend(self.check_spam_patterns(text))
        all_violations.extend(self.check_advertisement_patterns(text))
        all_violations.extend(self.check_inappropriate_content(text))
        all_violations.extend(self.check_fake_review_indicators(text))
        all_violations.extend(self.check_promotional_content(text))
        
        return all_violations

class MLBasedPolicyChecker:
    """
    ML-based policy checker using trained models.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the ML-based policy checker.
        
        Args:
            model_path: Path to the trained policy enforcement model
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        
        # Policy thresholds
        self.policy_thresholds = {
            PolicyViolationType.SPAM: 0.7,
            PolicyViolationType.ADVERTISEMENT: 0.6,
            PolicyViolationType.IRRELEVANT: 0.5,
            PolicyViolationType.INAPPROPRIATE_CONTENT: 0.8,
            PolicyViolationType.FAKE_REVIEW: 0.6
        }
    
    def load_model(self):
        """Load the trained policy enforcement model."""
        if self.model_path:
            try:
                # In practice, you would load your actual trained model here
                logger.info(f"Loading policy enforcement model from {self.model_path}")
                # self.model = load_model(self.model_path)
                # self.tokenizer = load_tokenizer(self.model_path)
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                raise
    
    def predict_policy_violations(self, text: str) -> Dict[PolicyViolationType, float]:
        """
        Predict policy violation probabilities for a given text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary mapping violation types to probabilities
        """
        # Mock implementation - in practice, you would use your trained model
        text_lower = text.lower()
        
        # Simple heuristic-based mock predictions
        predictions = {}
        
        # Spam detection
        spam_indicators = ['buy', 'sale', 'discount', 'promotion', 'deal', 'free', 'cheap']
        spam_score = sum(1 for word in spam_indicators if word in text_lower) / len(spam_indicators)
        predictions[PolicyViolationType.SPAM] = min(0.9, spam_score * 2)
        
        # Advertisement detection
        ad_indicators = ['advertise', 'marketing', 'promote', 'sponsor', 'brand']
        ad_score = sum(1 for word in ad_indicators if word in text_lower) / len(ad_indicators)
        predictions[PolicyViolationType.ADVERTISEMENT] = min(0.9, ad_score * 2)
        
        # Irrelevant content detection
        irrelevant_score = 0.1 if len(text.split()) < 5 else 0.05
        predictions[PolicyViolationType.IRRELEVANT] = irrelevant_score
        
        # Inappropriate content detection
        inappropriate_indicators = ['hate', 'racist', 'sexist', 'offensive', 'stupid', 'idiot']
        inappropriate_score = sum(1 for word in inappropriate_indicators if word in text_lower) / len(inappropriate_indicators)
        predictions[PolicyViolationType.INAPPROPRIATE_CONTENT] = min(0.9, inappropriate_score * 3)
        
        # Fake review detection
        fake_indicators = ['never been', 'never visited', 'fake', 'false', 'competitor']
        fake_score = sum(1 for phrase in fake_indicators if phrase in text_lower) / len(fake_indicators)
        predictions[PolicyViolationType.FAKE_REVIEW] = min(0.9, fake_score * 2)
        
        return predictions
    
    def check_policies(self, text: str) -> List[PolicyViolation]:
        """
        Check policy violations using ML model predictions.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of detected policy violations
        """
        predictions = self.predict_policy_violations(text)
        violations = []
        
        for violation_type, probability in predictions.items():
            threshold = self.policy_thresholds.get(violation_type, 0.5)
            
            if probability > threshold:
                # Determine severity based on probability
                if probability > 0.8:
                    severity = "high"
                elif probability > 0.6:
                    severity = "medium"
                else:
                    severity = "low"
                
                violation = PolicyViolation(
                    violation_type=violation_type,
                    severity=severity,
                    confidence=probability,
                    description=f"ML model detected {violation_type.value} with {probability:.2f} confidence",
                    evidence=[f"Model prediction: {probability:.3f}"]
                )
                violations.append(violation)
        
        return violations

class PolicyEnforcer:
    """
    Main policy enforcement class that combines rule-based and ML-based checking.
    """
    
    def __init__(self, use_ml: bool = True, model_path: Optional[str] = None):
        """
        Initialize the policy enforcer.
        
        Args:
            use_ml: Whether to use ML-based checking
            model_path: Path to the trained ML model
        """
        self.rule_checker = RuleBasedPolicyChecker()
        self.ml_checker = MLBasedPolicyChecker(model_path) if use_ml else None
        self.use_ml = use_ml
        
        if self.use_ml and self.ml_checker:
            try:
                self.ml_checker.load_model()
            except Exception as e:
                logger.warning(f"Failed to load ML model: {e}. Falling back to rule-based only.")
                self.use_ml = False
    
    def enforce_policies(self, text: str, combine_methods: bool = True) -> Dict[str, Any]:
        """
        Enforce policies on a given text.
        
        Args:
            text: Input text to analyze
            combine_methods: Whether to combine rule-based and ML-based results
            
        Returns:
            Dictionary containing policy enforcement results
        """
        results = {
            'text': text,
            'violations': [],
            'rule_based_violations': [],
            'ml_based_violations': [],
            'overall_risk_score': 0.0,
            'action_recommended': 'approve'
        }
        
        # Rule-based checking
        rule_violations = self.rule_checker.check_all_policies(text)
        results['rule_based_violations'] = rule_violations
        
        # ML-based checking
        ml_violations = []
        if self.use_ml and self.ml_checker:
            ml_violations = self.ml_checker.check_policies(text)
            results['ml_based_violations'] = ml_violations
        
        # Combine results
        if combine_methods:
            # Merge violations from both methods
            all_violations = rule_violations + ml_violations
            
            # Remove duplicates based on violation type
            unique_violations = {}
            for violation in all_violations:
                vtype = violation.violation_type
                if vtype not in unique_violations or violation.confidence > unique_violations[vtype].confidence:
                    unique_violations[vtype] = violation
            
            results['violations'] = list(unique_violations.values())
        else:
            # Use rule-based as primary, ML as secondary
            results['violations'] = rule_violations if rule_violations else ml_violations
        
        # Calculate overall risk score
        if results['violations']:
            risk_scores = []
            for violation in results['violations']:
                severity_weight = {'low': 0.3, 'medium': 0.6, 'high': 1.0}
                risk_score = violation.confidence * severity_weight.get(violation.severity, 0.5)
                risk_scores.append(risk_score)
            
            results['overall_risk_score'] = max(risk_scores)
        
        # Determine recommended action
        if results['overall_risk_score'] > 0.8:
            results['action_recommended'] = 'reject'
        elif results['overall_risk_score'] > 0.5:
            results['action_recommended'] = 'review'
        else:
            results['action_recommended'] = 'approve'
        
        return results
    
    def batch_enforce_policies(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Enforce policies on a batch of texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of policy enforcement results
        """
        results = []
        for text in texts:
            result = self.enforce_policies(text)
            results.append(result)
        
        return results
    
    def generate_policy_report(self, enforcement_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a summary report of policy enforcement results.
        
        Args:
            enforcement_results: List of enforcement results
            
        Returns:
            Summary report dictionary
        """
        total_texts = len(enforcement_results)
        
        # Count violations by type
        violation_counts = {}
        action_counts = {'approve': 0, 'review': 0, 'reject': 0}
        risk_scores = []
        
        for result in enforcement_results:
            # Count actions
            action_counts[result['action_recommended']] += 1
            risk_scores.append(result['overall_risk_score'])
            
            # Count violations
            for violation in result['violations']:
                vtype = violation.violation_type.value
                if vtype not in violation_counts:
                    violation_counts[vtype] = 0
                violation_counts[vtype] += 1
        
        report = {
            'summary': {
                'total_texts_analyzed': total_texts,
                'texts_with_violations': sum(1 for r in enforcement_results if r['violations']),
                'average_risk_score': np.mean(risk_scores) if risk_scores else 0.0,
                'max_risk_score': max(risk_scores) if risk_scores else 0.0
            },
            'action_distribution': action_counts,
            'violation_type_counts': violation_counts,
            'risk_score_distribution': {
                'low_risk': sum(1 for score in risk_scores if score < 0.3),
                'medium_risk': sum(1 for score in risk_scores if 0.3 <= score < 0.7),
                'high_risk': sum(1 for score in risk_scores if score >= 0.7)
            }
        }
        
        return report

def main():
    """
    Example usage of the PolicyEnforcer class.
    """
    # Sample texts for testing
    sample_texts = [
        "Great restaurant with excellent food and service!",
        "Buy our amazing products at 50% discount! Call 123-456-7890 now!",
        "This place is terrible and the staff are idiots.",
        "I never actually visited this place but I heard it's bad.",
        "Check out our website and follow us on social media!",
        "Average food, decent service, reasonable prices."
    ]
    
    # Initialize policy enforcer
    enforcer = PolicyEnforcer(use_ml=True)
    
    try:
        # Enforce policies on sample texts
        results = enforcer.batch_enforce_policies(sample_texts)
        
        # Generate report
        report = enforcer.generate_policy_report(results)
        
        print("Policy Enforcement Results:")
        print("=" * 50)
        
        for i, result in enumerate(results):
            print(f"\nText {i+1}: {result['text'][:50]}...")
            print(f"Risk Score: {result['overall_risk_score']:.3f}")
            print(f"Action: {result['action_recommended']}")
            
            if result['violations']:
                print("Violations:")
                for violation in result['violations']:
                    print(f"  - {violation.violation_type.value}: {violation.severity} severity, {violation.confidence:.3f} confidence")
            else:
                print("No violations detected")
        
        print("\n" + "=" * 50)
        print("Summary Report:")
        print(f"Total texts analyzed: {report['summary']['total_texts_analyzed']}")
        print(f"Texts with violations: {report['summary']['texts_with_violations']}")
        print(f"Average risk score: {report['summary']['average_risk_score']:.3f}")
        print(f"Action distribution: {report['action_distribution']}")
        print(f"Violation types: {report['violation_type_counts']}")
        
    except Exception as e:
        logger.error(f"Policy enforcement failed: {e}")

if __name__ == "__main__":
    main()

