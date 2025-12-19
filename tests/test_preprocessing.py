"""
Unit Tests for Preprocessing Module
===================================

Tests the feature engineering and preprocessing pipeline.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.preprocessing import UPDRSPreprocessor, create_sample_input


@pytest.fixture
def sample_input():
    """Create a sample input for testing."""
    return create_sample_input()


@pytest.fixture
def preprocessor():
    """Create a preprocessor instance (will fail if model not generated)."""
    try:
        return UPDRSPreprocessor()
    except FileNotFoundError:
        pytest.skip("Scaler file not found. Run save_model.py first.")


def test_create_sample_input():
    """Test that sample input creation returns correct structure."""
    sample = create_sample_input()

    # Check it's a dictionary
    assert isinstance(sample, dict)

    # Check it has required features
    required_features = [
        'age', 'sex', 'test_time', 'Jitter(%)', 'Jitter(Abs)',
        'Shimmer', 'Shimmer:APQ11', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE'
    ]
    for feature in required_features:
        assert feature in sample

    # Check values are numeric
    for value in sample.values():
        assert isinstance(value, (int, float))


def test_engineer_features(preprocessor, sample_input):
    """Test feature engineering creates correct new features."""
    df = pd.DataFrame([sample_input])
    engineered_df = preprocessor.engineer_features(df)

    # Check new features exist
    assert 'voice_quality_ratio' in engineered_df.columns
    assert 'test_time_squared' in engineered_df.columns

    # Check voice_quality_ratio calculation
    expected_ratio = sample_input['HNR'] / (sample_input['NHR'] + 0.001)
    assert np.isclose(engineered_df['voice_quality_ratio'].iloc[0], expected_ratio)

    # Check test_time_squared calculation
    expected_squared = sample_input['test_time'] ** 2
    assert np.isclose(engineered_df['test_time_squared'].iloc[0], expected_squared)


def test_select_features(preprocessor, sample_input):
    """Test feature selection returns correct 14 features."""
    df = pd.DataFrame([sample_input])
    engineered_df = preprocessor.engineer_features(df)
    selected_df = preprocessor.select_features(engineered_df)

    # Check correct number of features
    assert selected_df.shape[1] == 14

    # Check exact feature names
    expected_features = [
        'age', 'sex', 'test_time', 'Jitter(%)', 'Jitter(Abs)',
        'Shimmer', 'Shimmer:APQ11', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE',
        'voice_quality_ratio', 'test_time_squared'
    ]
    assert list(selected_df.columns) == expected_features


def test_scale_features(preprocessor, sample_input):
    """Test feature scaling returns correct shape."""
    df = pd.DataFrame([sample_input])
    engineered_df = preprocessor.engineer_features(df)
    selected_df = preprocessor.select_features(engineered_df)
    scaled_array = preprocessor.scale_features(selected_df)

    # Check output is numpy array
    assert isinstance(scaled_array, np.ndarray)

    # Check shape is correct
    assert scaled_array.shape == (1, 14)


def test_preprocess_full_pipeline(preprocessor, sample_input):
    """Test the full preprocessing pipeline end-to-end."""
    result = preprocessor.preprocess(sample_input)

    # Check output is numpy array with correct shape
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 14)

    # Check values are scaled (should have both positive and negative values typically)
    # This is a rough check - scaled values should not all be identical
    assert not np.all(result == result[0, 0])


def test_preprocess_missing_feature(preprocessor):
    """Test that preprocessing raises error for missing features."""
    incomplete_input = {
        'age': 65,
        'sex': 0
        # Missing other required features
    }

    with pytest.raises(ValueError, match="Missing required features"):
        preprocessor.preprocess(incomplete_input)


def test_preprocess_batch(preprocessor, sample_input):
    """Test batch preprocessing with multiple samples."""
    # Create DataFrame with 3 identical samples
    df = pd.DataFrame([sample_input] * 3)

    result = preprocessor.preprocess_batch(df)

    # Check output shape
    assert result.shape == (3, 14)

    # Check all rows are identical (same input)
    assert np.allclose(result[0], result[1])
    assert np.allclose(result[1], result[2])


def test_get_feature_names(preprocessor):
    """Test getting required feature names."""
    features = preprocessor.get_feature_names()

    # Check it's a list
    assert isinstance(features, list)

    # Check it has 12 raw features (before engineering)
    assert len(features) == 12

    # Check specific features are present
    assert 'age' in features
    assert 'HNR' in features
    assert 'PPE' in features


def test_get_final_feature_names(preprocessor):
    """Test getting final feature names (after engineering)."""
    features = preprocessor.get_final_feature_names()

    # Check it's a list
    assert isinstance(features, list)

    # Check it has 14 final features
    assert len(features) == 14

    # Check engineered features are present
    assert 'voice_quality_ratio' in features
    assert 'test_time_squared' in features


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
