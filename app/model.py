"""
Model Wrapper for Parkinson's UPDRS Prediction
==============================================

This module provides a high-level interface for loading the trained model
and making predictions with proper preprocessing and validation.
"""

import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Union, Tuple, Optional

from .preprocessing import UPDRSPreprocessor
from .validators import InputValidator, ValidationResult


class UPDRSPredictor:
    """
    High-level wrapper for UPDRS prediction.

    Combines model loading, preprocessing, validation, and prediction
    into a single easy-to-use interface.
    """

    def __init__(
        self,
        model_path: str = "models/random_forest_model.pkl",
        scaler_path: str = "models/scaler.pkl",
        config_path: str = "models/feature_config.json"
    ):
        """
        Initialize the predictor with model, scaler, and configuration.

        Args:
            model_path: Path to the trained model pickle file
            scaler_path: Path to the fitted scaler pickle file
            config_path: Path to the feature configuration JSON file
        """
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.config_path = Path(config_path)

        self.model = None
        self.config = None
        self.preprocessor = None
        self.validator = None

        self._load_model()
        self._load_config()
        self._initialize_preprocessor()
        self._initialize_validator()

    def _load_model(self):
        """Load the trained Random Forest model from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. "
                "Please run save_model.py first to generate model artifacts."
            )

        try:
            self.model = joblib.load(self.model_path)
        except Exception as e:
            raise RuntimeError(f"Error loading model: {str(e)}")

    def _load_config(self):
        """Load feature configuration from JSON."""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Config file not found at {self.config_path}. "
                "Please run save_model.py first to generate model artifacts."
            )

        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Error loading config: {str(e)}")

    def _initialize_preprocessor(self):
        """Initialize the preprocessor with scaler."""
        try:
            self.preprocessor = UPDRSPreprocessor(scaler_path=str(self.scaler_path))
        except Exception as e:
            raise RuntimeError(f"Error initializing preprocessor: {str(e)}")

    def _initialize_validator(self):
        """Initialize the input validator with config."""
        try:
            self.validator = InputValidator(config_path=str(self.config_path))
        except Exception as e:
            raise RuntimeError(f"Error initializing validator: {str(e)}")

    def predict(
        self,
        input_data: Union[Dict, pd.DataFrame],
        validate: bool = True
    ) -> Dict:
        """
        Make a UPDRS prediction with full preprocessing and validation.

        Args:
            input_data: Either a dictionary with feature values or a DataFrame
            validate: Whether to perform input validation (default: True)

        Returns:
            Dictionary with:
                - prediction: Predicted UPDRS score
                - validation: ValidationResult object (if validate=True)
                - severity: Severity interpretation (Mild/Moderate/Severe)
                - confidence: Contextual confidence information
        """
        # Validate input if requested
        validation_result = None
        if validate:
            # Convert DataFrame to dict for validation if needed
            if isinstance(input_data, pd.DataFrame):
                # Use first row if DataFrame
                input_dict = input_data.iloc[0].to_dict() if len(input_data) > 0 else {}
            else:
                input_dict = input_data

            validation_result = self.validator.validate(input_dict)

            # If there are hard errors, return early
            if not validation_result.is_valid:
                return {
                    'prediction': None,
                    'validation': validation_result,
                    'severity': None,
                    'confidence': None,
                    'error': "Validation failed. Please fix errors before predicting."
                }

        # Preprocess input
        try:
            scaled_features = self.preprocessor.preprocess(input_data)
        except Exception as e:
            return {
                'prediction': None,
                'validation': validation_result,
                'severity': None,
                'confidence': None,
                'error': f"Preprocessing error: {str(e)}"
            }

        # Make prediction
        try:
            prediction = float(self.model.predict(scaled_features)[0])
        except Exception as e:
            return {
                'prediction': None,
                'validation': validation_result,
                'severity': None,
                'confidence': None,
                'error': f"Prediction error: {str(e)}"
            }

        # Interpret severity
        severity = self._interpret_severity(prediction)

        # Generate confidence context
        confidence = self._generate_confidence_context(prediction)

        return {
            'prediction': round(prediction, 2),
            'validation': validation_result,
            'severity': severity,
            'confidence': confidence,
            'error': None
        }

    def predict_batch(
        self,
        input_df: pd.DataFrame,
        validate: bool = True
    ) -> pd.DataFrame:
        """
        Make predictions for multiple samples.

        Args:
            input_df: DataFrame where each row is a patient sample
            validate: Whether to perform input validation (default: True)

        Returns:
            DataFrame with original data plus prediction columns
        """
        results = []

        for idx, row in input_df.iterrows():
            result = self.predict(row.to_dict(), validate=validate)
            results.append({
                'prediction': result['prediction'],
                'severity': result['severity'],
                'has_errors': result['error'] is not None,
                'has_warnings': result['validation'].warnings if result['validation'] else []
            })

        results_df = pd.DataFrame(results)
        return pd.concat([input_df.reset_index(drop=True), results_df], axis=1)

    def _interpret_severity(self, updrs_score: float) -> str:
        """
        Interpret UPDRS score severity.

        Based on clinical guidelines:
        - Mild: 7-25 points
        - Moderate: 25-40 points
        - Severe: 40+ points

        Args:
            updrs_score: Predicted UPDRS score

        Returns:
            Severity level as string
        """
        if updrs_score < 25:
            return "Mild"
        elif updrs_score < 40:
            return "Moderate"
        else:
            return "Severe"

    def _generate_confidence_context(self, prediction: float) -> str:
        """
        Generate contextual confidence information.

        Args:
            prediction: Predicted UPDRS score

        Returns:
            String with confidence context
        """
        # Check if prediction is within training range
        if self.config:
            metrics = self.config.get('performance_metrics', {})
            mae = metrics.get('test_mae', 3.81)

            # Prediction is within typical range
            if 7 <= prediction <= 55:
                confidence = f"Prediction is within typical range. Expected error: ±{mae:.2f} points."
            else:
                confidence = f"Prediction ({prediction:.2f}) is outside typical UPDRS range (7-55). Use with caution."
        else:
            confidence = "Confidence information not available."

        return confidence

    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get feature importance from the Random Forest model.

        Args:
            top_n: Number of top features to return (default: 10)

        Returns:
            DataFrame with features and their importance scores
        """
        if self.config and 'feature_importance' in self.config:
            importance_dict = self.config['feature_importance']
            importance_df = pd.DataFrame({
                'feature': list(importance_dict.keys()),
                'importance': list(importance_dict.values())
            })
            importance_df = importance_df.sort_values('importance', ascending=False)
            return importance_df.head(top_n)
        else:
            return pd.DataFrame(columns=['feature', 'importance'])

    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        if self.config:
            return {
                'model_type': self.config.get('model_type', 'Unknown'),
                'parameters': self.config.get('model_parameters', {}),
                'performance': self.config.get('performance_metrics', {}),
                'feature_count': self.config.get('feature_count', 0)
            }
        else:
            return {'error': 'Config not loaded'}

    def get_required_features(self) -> list:
        """
        Get list of required input features.

        Returns:
            List of feature names that user must provide
        """
        if self.preprocessor:
            return self.preprocessor.get_feature_names()
        else:
            return []


def create_example_prediction():
    """
    Example usage of the UPDRSPredictor.
    """
    print("="*70)
    print("UPDRS Predictor Example Usage")
    print("="*70)

    try:
        # Initialize predictor
        predictor = UPDRSPredictor()

        # Example input (average patient)
        sample_input = {
            'age': 65,
            'sex': 0,
            'test_time': 92.0,
            'Jitter(%)': 0.0049,
            'Jitter(Abs)': 0.000035,
            'Shimmer': 0.0253,
            'Shimmer:APQ11': 0.0227,
            'NHR': 0.0184,
            'HNR': 21.92,
            'RPDE': 0.5422,
            'DFA': 0.6436,
            'PPE': 0.2055
        }

        print("\nSample Input:")
        for feature, value in sample_input.items():
            print(f"  {feature}: {value}")

        # Make prediction
        result = predictor.predict(sample_input)

        print("\nPrediction Result:")
        print(f"  UPDRS Score: {result['prediction']}")
        print(f"  Severity: {result['severity']}")
        print(f"  Confidence: {result['confidence']}")

        if result['validation']:
            print(f"\nValidation:")
            print(f"  Valid: {result['validation'].is_valid}")
            if result['validation'].warnings:
                print(f"  Warnings: {len(result['validation'].warnings)}")

        print("\n" + "="*70)

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please run save_model.py first to generate model artifacts.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")


if __name__ == "__main__":
    create_example_prediction()
