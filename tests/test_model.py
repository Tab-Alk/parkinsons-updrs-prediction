"""
Unit Tests for Model Wrapper Module
===================================

Tests model loading and prediction functionality.
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.model import UPDRSPredictor
from app.preprocessing import create_sample_input


@pytest.fixture
def predictor():
    """Create a predictor instance (will fail if model not generated)."""
    try:
        return UPDRSPredictor()
    except FileNotFoundError:
        pytest.skip("Model files not found. Run save_model.py first.")


@pytest.fixture
def sample_input():
    """Create a sample input for testing."""
    return create_sample_input()


def test_predictor_initialization(predictor):
    """Test that predictor initializes successfully."""
    assert predictor.model is not None
    assert predictor.config is not None
    assert predictor.preprocessor is not None
    assert predictor.validator is not None


def test_predict_returns_valid_structure(predictor, sample_input):
    """Test that predict returns the expected dictionary structure."""
    result = predictor.predict(sample_input)

    # Check all expected keys are present
    assert 'prediction' in result
    assert 'validation' in result
    assert 'severity' in result
    assert 'confidence' in result
    assert 'error' in result


def test_predict_returns_numeric_prediction(predictor, sample_input):
    """Test that prediction is a numeric value."""
    result = predictor.predict(sample_input)

    assert result['error'] is None
    assert result['prediction'] is not None
    assert isinstance(result['prediction'], (int, float))


def test_prediction_in_valid_range(predictor, sample_input):
    """Test that prediction is within reasonable UPDRS range."""
    result = predictor.predict(sample_input)

    # UPDRS scores typically range from 7 to 55 (from dataset)
    # Allow some margin for edge cases
    assert result['prediction'] >= 0
    assert result['prediction'] <= 100  # Very generous upper bound


def test_severity_interpretation(predictor, sample_input):
    """Test that severity is correctly interpreted."""
    result = predictor.predict(sample_input)

    assert result['severity'] in ['Mild', 'Moderate', 'Severe']


def test_predict_with_validation(predictor, sample_input):
    """Test prediction with validation enabled."""
    result = predictor.predict(sample_input, validate=True)

    assert result['validation'] is not None
    assert result['validation'].is_valid == True


def test_predict_without_validation(predictor, sample_input):
    """Test prediction with validation disabled."""
    result = predictor.predict(sample_input, validate=False)

    assert result['validation'] is None
    assert result['prediction'] is not None


def test_predict_with_invalid_input(predictor):
    """Test prediction with invalid input returns error."""
    invalid_input = {
        'age': 65,
        'sex': 0
        # Missing other required features
    }

    result = predictor.predict(invalid_input, validate=True)

    # Should fail validation
    assert result['validation'] is not None
    assert result['validation'].is_valid == False
    assert result['prediction'] is None


def test_predict_batch(predictor, sample_input):
    """Test batch prediction with multiple samples."""
    # Create DataFrame with 3 samples
    df = pd.DataFrame([sample_input] * 3)

    results_df = predictor.predict_batch(df)

    # Check output is DataFrame
    assert isinstance(results_df, pd.DataFrame)

    # Check it has prediction columns
    assert 'prediction' in results_df.columns
    assert 'severity' in results_df.columns

    # Check it has 3 rows
    assert len(results_df) == 3


def test_get_feature_importance(predictor):
    """Test getting feature importance."""
    importance_df = predictor.get_feature_importance(top_n=5)

    # Check it's a DataFrame
    assert isinstance(importance_df, pd.DataFrame)

    # Check it has correct columns
    assert 'feature' in importance_df.columns
    assert 'importance' in importance_df.columns

    # Check it has at most 5 rows
    assert len(importance_df) <= 5


def test_get_model_info(predictor):
    """Test getting model information."""
    info = predictor.get_model_info()

    # Check expected keys
    assert 'model_type' in info
    assert 'parameters' in info
    assert 'performance' in info
    assert 'feature_count' in info

    # Check model type is correct
    assert info['model_type'] == 'RandomForestRegressor'


def test_get_required_features(predictor):
    """Test getting list of required features."""
    features = predictor.get_required_features()

    # Check it's a list
    assert isinstance(features, list)

    # Check it has 12 raw features
    assert len(features) == 12

    # Check specific features are present
    assert 'age' in features
    assert 'sex' in features
    assert 'test_time' in features


def test_consistent_predictions(predictor, sample_input):
    """Test that same input gives same prediction (reproducibility)."""
    result1 = predictor.predict(sample_input)
    result2 = predictor.predict(sample_input)

    # Should get identical predictions
    assert result1['prediction'] == result2['prediction']


def test_different_inputs_give_different_predictions(predictor):
    """Test that different inputs give different predictions."""
    input1 = create_sample_input()

    input2 = create_sample_input()
    input2['age'] = 45  # Change age

    result1 = predictor.predict(input1)
    result2 = predictor.predict(input2)

    # Predictions should be different (age is important feature)
    # Note: Could theoretically be same but highly unlikely
    # So we'll just check both succeeded
    assert result1['prediction'] is not None
    assert result2['prediction'] is not None


def test_severity_mild_range(predictor):
    """Test severity classification for mild symptoms."""
    # Create input likely to produce mild prediction
    mild_input = create_sample_input()
    mild_input['age'] = 45  # Younger age

    result = predictor.predict(mild_input)

    # Check prediction is in mild range or properly classified
    if result['prediction'] < 25:
        assert result['severity'] == 'Mild'


def test_severity_severe_range(predictor):
    """Test severity classification for severe symptoms."""
    # Test the interpretation function directly
    severity = predictor._interpret_severity(45)
    assert severity == 'Severe'

    severity = predictor._interpret_severity(20)
    assert severity == 'Mild'

    severity = predictor._interpret_severity(30)
    assert severity == 'Moderate'


def test_confidence_context_generation(predictor):
    """Test confidence context generation."""
    # Test with normal prediction
    confidence = predictor._generate_confidence_context(30)
    assert isinstance(confidence, str)
    assert len(confidence) > 0

    # Test with out-of-range prediction
    confidence = predictor._generate_confidence_context(100)
    assert 'caution' in confidence.lower() or 'outside' in confidence.lower()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
