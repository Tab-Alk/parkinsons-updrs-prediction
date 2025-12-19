"""
Input Validation Module for Parkinson's UPDRS Prediction
========================================================

This module provides three-tier validation for user inputs:
- Tier 1 (Hard Errors): Invalid inputs that prevent prediction
- Tier 2 (Warnings): Unusual but plausible inputs
- Tier 3 (Info): Contextual information about inputs
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd


class ValidationResult:
    """Container for validation results."""

    def __init__(self):
        self.is_valid = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []

    def add_error(self, message: str):
        """Add a hard error (prevents prediction)."""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str):
        """Add a warning (allows prediction but shows concern)."""
        self.warnings.append(message)

    def add_info(self, message: str):
        """Add informational message."""
        self.info.append(message)

    def has_issues(self) -> bool:
        """Check if there are any errors or warnings."""
        return bool(self.errors or self.warnings)

    def __str__(self) -> str:
        """Format validation results as a string."""
        lines = []
        if self.errors:
            lines.append("ERRORS:")
            for error in self.errors:
                lines.append(f"  ✗ {error}")
        if self.warnings:
            lines.append("\nWARNINGS:")
            for warning in self.warnings:
                lines.append(f"  ⚠ {warning}")
        if self.info:
            lines.append("\nINFO:")
            for info_msg in self.info:
                lines.append(f"  ℹ {info_msg}")
        return "\n".join(lines) if lines else "✓ All validations passed"


class InputValidator:
    """
    Validates user inputs for UPDRS prediction.

    Uses three-tier validation:
    1. Hard errors: Missing features, wrong types, impossible values
    2. Warnings: Unusual but plausible values
    3. Info: Context about input values
    """

    def __init__(self, config_path: str = "models/feature_config.json"):
        """
        Initialize validator with feature configuration.

        Args:
            config_path: Path to feature configuration JSON file
        """
        self.config_path = Path(config_path)
        self.config = None
        self.validation_ranges = None
        self.feature_names = None

        self._load_config()

    def _load_config(self):
        """Load feature configuration from JSON."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
                self.validation_ranges = self.config.get('validation_ranges', {})
                self.feature_names = self.config.get('feature_names', [])
            except Exception as e:
                print(f"Warning: Could not load config from {self.config_path}: {e}")
                self._set_default_config()
        else:
            self._set_default_config()

    def _set_default_config(self):
        """Set default validation ranges if config file not available."""
        self.feature_names = [
            'age', 'sex', 'test_time', 'Jitter(%)', 'Jitter(Abs)',
            'Shimmer', 'Shimmer:APQ11', 'NHR', 'HNR',
            'RPDE', 'DFA', 'PPE'
        ]

        self.validation_ranges = {
            "age": {"min": 30, "max": 90},
            "sex": {"min": 0, "max": 1},
            "test_time": {"min": -5, "max": 250},
            "Jitter(%)": {"min": 0.0001, "max": 0.15},
            "Jitter(Abs)": {"min": 0.000001, "max": 0.0005},
            "Shimmer": {"min": 0.005, "max": 0.3},
            "Shimmer:APQ11": {"min": 0.002, "max": 0.3},
            "NHR": {"min": 0.0001, "max": 0.8},
            "HNR": {"min": 1, "max": 40},
            "RPDE": {"min": 0.1, "max": 1.0},
            "DFA": {"min": 0.5, "max": 0.9},
            "PPE": {"min": 0.02, "max": 0.8}
        }

    def validate(self, input_data: Dict) -> ValidationResult:
        """
        Perform comprehensive validation on input data.

        Args:
            input_data: Dictionary with feature names and values

        Returns:
            ValidationResult object with errors, warnings, and info
        """
        result = ValidationResult()

        # Tier 1: Hard Error Checks
        self._check_completeness(input_data, result)
        self._check_types(input_data, result)
        self._check_hard_limits(input_data, result)

        # Only proceed to warnings if no hard errors
        if result.is_valid:
            # Tier 2: Warning Checks
            self._check_unusual_values(input_data, result)
            self._check_logical_consistency(input_data, result)

            # Tier 3: Informational Messages
            self._add_context_info(input_data, result)

        return result

    def _check_completeness(self, input_data: Dict, result: ValidationResult):
        """Tier 1: Check that all required features are present."""
        # Only check raw features (before engineering)
        required_raw = [
            'age', 'sex', 'test_time', 'Jitter(%)', 'Jitter(Abs)',
            'Shimmer', 'Shimmer:APQ11', 'NHR', 'HNR',
            'RPDE', 'DFA', 'PPE'
        ]

        missing = set(required_raw) - set(input_data.keys())
        if missing:
            result.add_error(
                f"Missing required features: {', '.join(sorted(missing))}"
            )

        # Check for unexpected extra features (not an error, just info)
        extra = set(input_data.keys()) - set(required_raw)
        if extra:
            result.add_info(
                f"Extra features provided (will be ignored): {', '.join(sorted(extra))}"
            )

    def _check_types(self, input_data: Dict, result: ValidationResult):
        """Tier 1: Check that all values are numeric."""
        for feature, value in input_data.items():
            if feature not in self.validation_ranges:
                continue  # Skip features not in our validation set

            try:
                float(value)
            except (ValueError, TypeError):
                result.add_error(
                    f"Feature '{feature}' must be numeric, got: {type(value).__name__}"
                )

    def _check_hard_limits(self, input_data: Dict, result: ValidationResult):
        """Tier 1: Check for impossible/extreme values."""
        for feature, value in input_data.items():
            if feature not in self.validation_ranges:
                continue

            try:
                value_float = float(value)
            except (ValueError, TypeError):
                continue  # Already caught in type check

            ranges = self.validation_ranges[feature]

            # Special handling for binary sex feature
            if feature == 'sex':
                if value_float not in [0, 1]:
                    result.add_error(
                        f"Feature 'sex' must be 0 (male) or 1 (female), got: {value_float}"
                    )
                continue

            # Check extreme outliers (beyond reasonable limits)
            if value_float < ranges['min']:
                result.add_error(
                    f"Feature '{feature}' value {value_float} is below minimum threshold {ranges['min']}"
                )
            elif value_float > ranges['max']:
                result.add_error(
                    f"Feature '{feature}' value {value_float} exceeds maximum threshold {ranges['max']}"
                )

    def _check_unusual_values(self, input_data: Dict, result: ValidationResult):
        """Tier 2: Warn about unusual but plausible values."""
        # Age warnings
        age = input_data.get('age')
        if age is not None:
            age_float = float(age)
            if age_float < 40:
                result.add_warning(
                    f"Age {age_float} is younger than typical Parkinson's onset age (usually 50+)"
                )
            elif age_float > 85:
                result.add_warning(
                    f"Age {age_float} is at the upper range of the dataset (max: 85)"
                )

        # Check voice features against dataset percentiles
        voice_high_threshold = {
            'Jitter(%)': 0.013,  # ~95th percentile
            'Shimmer': 0.06,
            'NHR': 0.087
        }

        for feature, threshold in voice_high_threshold.items():
            value = input_data.get(feature)
            if value is not None and float(value) > threshold:
                result.add_warning(
                    f"Feature '{feature}' value {value} is higher than 95% of training data"
                )

        # Check HNR (low values indicate voice problems)
        hnr = input_data.get('HNR')
        if hnr is not None and float(hnr) < 10:
            result.add_warning(
                f"HNR value {hnr} is very low, indicating significant voice quality issues"
            )

    def _check_logical_consistency(self, input_data: Dict, result: ValidationResult):
        """Tier 2: Check for logically inconsistent feature combinations."""
        # HNR and NHR should have inverse relationship
        hnr = input_data.get('HNR')
        nhr = input_data.get('NHR')

        if hnr is not None and nhr is not None:
            hnr_float = float(hnr)
            nhr_float = float(nhr)

            # If HNR is very high, NHR should be low
            if hnr_float > 30 and nhr_float > 0.05:
                result.add_warning(
                    "Unusual combination: High HNR (good voice) but elevated NHR (noise). "
                    "These features typically have an inverse relationship."
                )

            # If HNR is very low, NHR should be high
            if hnr_float < 15 and nhr_float < 0.01:
                result.add_warning(
                    "Unusual combination: Low HNR (poor voice) but very low NHR (little noise). "
                    "Please verify measurements."
                )

    def _add_context_info(self, input_data: Dict, result: ValidationResult):
        """Tier 3: Add contextual information about the input."""
        # Dataset means for comparison
        dataset_means = {
            'age': 64.8,
            'test_time': 92.9,
            'Jitter(%)': 0.0062,
            'Shimmer': 0.025,
            'NHR': 0.032,
            'HNR': 21.7,
            'RPDE': 0.541,
            'DFA': 0.653,
            'PPE': 0.220
        }

        unusual_count = 0
        for feature, mean in dataset_means.items():
            value = input_data.get(feature)
            if value is not None:
                value_float = float(value)
                # Check if value deviates significantly from mean (>50% difference)
                percent_diff = abs(value_float - mean) / mean * 100
                if percent_diff > 50:
                    unusual_count += 1

        if unusual_count >= 3:
            result.add_info(
                f"{unusual_count} features deviate significantly from dataset averages. "
                "This may indicate unusual measurements or an atypical patient."
            )

    def validate_batch(self, input_df: pd.DataFrame) -> List[ValidationResult]:
        """
        Validate multiple inputs at once.

        Args:
            input_df: DataFrame where each row is a patient sample

        Returns:
            List of ValidationResult objects, one per row
        """
        results = []
        for idx, row in input_df.iterrows():
            result = self.validate(row.to_dict())
            results.append(result)
        return results

    def get_valid_range(self, feature_name: str) -> Optional[Dict]:
        """
        Get the valid range for a specific feature.

        Args:
            feature_name: Name of the feature

        Returns:
            Dictionary with min/max values, or None if feature not found
        """
        return self.validation_ranges.get(feature_name)


if __name__ == "__main__":
    """
    Test the validator with sample data.
    """
    print("Testing InputValidator...")

    validator = InputValidator()

    # Test 1: Valid input
    print("\n--- Test 1: Valid Input ---")
    valid_input = {
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
    result = validator.validate(valid_input)
    print(result)

    # Test 2: Missing feature
    print("\n--- Test 2: Missing Feature ---")
    invalid_input = valid_input.copy()
    del invalid_input['age']
    result = validator.validate(invalid_input)
    print(result)

    # Test 3: Invalid sex value
    print("\n--- Test 3: Invalid Sex Value ---")
    invalid_input = valid_input.copy()
    invalid_input['sex'] = 2
    result = validator.validate(invalid_input)
    print(result)

    # Test 4: Unusual age
    print("\n--- Test 4: Unusual Age (Warning) ---")
    unusual_input = valid_input.copy()
    unusual_input['age'] = 35
    result = validator.validate(unusual_input)
    print(result)

    print("\n✓ Validator tests complete!")
