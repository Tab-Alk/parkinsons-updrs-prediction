"""
Preprocessing Module for Parkinson's UPDRS Prediction
=====================================================

This module handles all preprocessing steps for the UPDRS prediction model,
including feature engineering and scaling. It MUST match the exact logic
from the training notebook to ensure prediction accuracy.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Union


class UPDRSPreprocessor:
    """
    Preprocessor for UPDRS prediction inputs.

    Handles:
    1. Feature engineering (voice_quality_ratio, test_time_squared)
    2. Feature selection (keeping only the final 14 features)
    3. Feature scaling (StandardScaler transformation)
    """

    def __init__(self, scaler_path: str = "models/scaler.pkl"):
        """
        Initialize the preprocessor with a fitted scaler.

        Args:
            scaler_path: Path to the saved StandardScaler pickle file
        """
        self.scaler_path = Path(scaler_path)
        self.scaler = None

        # Define the required input features (before engineering)
        # These are the raw features that user must provide
        self.required_raw_features = [
            'age', 'sex', 'test_time',
            'Jitter(%)', 'Jitter(Abs)',
            'Shimmer', 'Shimmer:APQ11',
            'NHR', 'HNR',
            'RPDE', 'DFA', 'PPE'
        ]

        # Define the final 14 features (after engineering, in correct order)
        # This MUST match the order from notebook cell 36
        self.final_features = [
            'age', 'sex', 'test_time',
            'Jitter(%)', 'Jitter(Abs)',
            'Shimmer', 'Shimmer:APQ11',
            'NHR', 'HNR',
            'RPDE', 'DFA', 'PPE',
            'voice_quality_ratio', 'test_time_squared'
        ]

        self._load_scaler()

    def _load_scaler(self):
        """Load the fitted StandardScaler from disk."""
        if not self.scaler_path.exists():
            raise FileNotFoundError(
                f"Scaler file not found at {self.scaler_path}. "
                "Please run save_model.py first to generate model artifacts."
            )

        try:
            self.scaler = joblib.load(self.scaler_path)
        except Exception as e:
            raise RuntimeError(f"Error loading scaler: {str(e)}")

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features.

        This MUST match the exact logic from notebook cell 33:
        - voice_quality_ratio = HNR / (NHR + 0.001)
        - test_time_squared = test_time ** 2

        Args:
            df: DataFrame with raw features

        Returns:
            DataFrame with engineered features added
        """
        df = df.copy()

        # Feature 1: Voice quality ratio (signal-to-noise composite)
        # Higher values indicate better voice quality
        df['voice_quality_ratio'] = df['HNR'] / (df['NHR'] + 0.001)

        # Feature 2: Test time squared (captures accelerating disease progression)
        df['test_time_squared'] = df['test_time'] ** 2

        return df

    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select the final 14 features in the correct order.

        This matches notebook cell 36 where correlated features were dropped.

        Args:
            df: DataFrame with all features (including engineered ones)

        Returns:
            DataFrame with only the final 14 features in correct order
        """
        return df[self.final_features]

    def scale_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Apply StandardScaler transformation.

        Uses the scaler fitted on training data (notebook cell 38).

        Args:
            df: DataFrame with final features

        Returns:
            Scaled feature array (numpy array)
        """
        if self.scaler is None:
            raise RuntimeError("Scaler not loaded. Cannot scale features.")

        return self.scaler.transform(df)

    def preprocess(self, input_data: Union[Dict, pd.DataFrame]) -> np.ndarray:
        """
        Full preprocessing pipeline: raw input → engineered → selected → scaled.

        This is the main method to use for prediction preprocessing.

        Args:
            input_data: Either a dictionary with feature values or a DataFrame

        Returns:
            Scaled feature array ready for model prediction

        Raises:
            ValueError: If required features are missing
        """
        # Convert dict to DataFrame if necessary
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()

        # Validate that all required raw features are present
        missing_features = set(self.required_raw_features) - set(df.columns)
        if missing_features:
            raise ValueError(
                f"Missing required features: {sorted(missing_features)}. "
                f"Required features are: {self.required_raw_features}"
            )

        # Step 1: Engineer features
        df = self.engineer_features(df)

        # Step 2: Select final features in correct order
        df = self.select_features(df)

        # Step 3: Scale features
        scaled_array = self.scale_features(df)

        return scaled_array

    def get_feature_names(self) -> list:
        """
        Get the list of required raw feature names (before engineering).

        Returns:
            List of feature names that user must provide
        """
        return self.required_raw_features.copy()

    def get_final_feature_names(self) -> list:
        """
        Get the list of final feature names (after engineering).

        Returns:
            List of final feature names used by the model
        """
        return self.final_features.copy()

    def preprocess_batch(self, input_df: pd.DataFrame) -> np.ndarray:
        """
        Preprocess multiple samples at once (batch prediction).

        Args:
            input_df: DataFrame where each row is a patient sample

        Returns:
            Scaled feature array for all samples
        """
        return self.preprocess(input_df)


def create_sample_input() -> Dict:
    """
    Create a sample input dictionary with average values from the dataset.

    Useful for testing and as a template for users.

    Returns:
        Dictionary with sample feature values
    """
    return {
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


if __name__ == "__main__":
    """
    Test the preprocessor with sample data.
    """
    print("Testing UPDRSPreprocessor...")

    try:
        # Initialize preprocessor
        preprocessor = UPDRSPreprocessor()

        # Test with sample input
        sample = create_sample_input()
        print(f"\nSample input: {sample}")

        # Preprocess
        scaled_features = preprocessor.preprocess(sample)

        print(f"\nPreprocessed output shape: {scaled_features.shape}")
        print(f"Expected shape: (1, 14)")
        print(f"Final features: {preprocessor.get_final_feature_names()}")

        print("\n✓ Preprocessor test successful!")

    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("Please run save_model.py first to generate the scaler file.")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
