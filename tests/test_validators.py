"""
Unit Tests for Input Validation Module
======================================

Tests the three-tier validation system.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.validators import InputValidator, ValidationResult
from app.preprocessing import create_sample_input


@pytest.fixture
def validator():
    """Create a validator instance."""
    return InputValidator()


@pytest.fixture
def valid_input():
    """Create a valid sample input."""
    return create_sample_input()


def test_validation_result_initialization():
    """Test ValidationResult initialization."""
    result = ValidationResult()

    assert result.is_valid == True
    assert len(result.errors) == 0
    assert len(result.warnings) == 0
    assert len(result.info) == 0


def test_validation_result_add_error():
    """Test adding errors to ValidationResult."""
    result = ValidationResult()

    result.add_error("Test error")

    assert result.is_valid == False
    assert len(result.errors) == 1
    assert result.errors[0] == "Test error"


def test_validation_result_add_warning():
    """Test adding warnings to ValidationResult."""
    result = ValidationResult()

    result.add_warning("Test warning")

    assert result.is_valid == True  # Warnings don't invalidate
    assert len(result.warnings) == 1
    assert result.warnings[0] == "Test warning"


def test_valid_input_passes(validator, valid_input):
    """Test that a valid input passes all validation."""
    result = validator.validate(valid_input)

    assert result.is_valid == True
    assert len(result.errors) == 0


def test_missing_feature_fails(validator, valid_input):
    """Test that missing required feature fails validation."""
    incomplete_input = valid_input.copy()
    del incomplete_input['age']

    result = validator.validate(incomplete_input)

    assert result.is_valid == False
    assert len(result.errors) > 0
    assert any('age' in error.lower() for error in result.errors)


def test_invalid_sex_value_fails(validator, valid_input):
    """Test that invalid sex value fails validation."""
    invalid_input = valid_input.copy()
    invalid_input['sex'] = 2  # Should be 0 or 1

    result = validator.validate(invalid_input)

    assert result.is_valid == False
    assert any('sex' in error.lower() for error in result.errors)


def test_non_numeric_value_fails(validator, valid_input):
    """Test that non-numeric values fail validation."""
    invalid_input = valid_input.copy()
    invalid_input['age'] = "not a number"

    result = validator.validate(invalid_input)

    assert result.is_valid == False
    assert any('numeric' in error.lower() for error in result.errors)


def test_value_below_minimum_fails(validator, valid_input):
    """Test that value below minimum threshold fails."""
    invalid_input = valid_input.copy()
    invalid_input['age'] = 25  # Below minimum of 30

    result = validator.validate(invalid_input)

    assert result.is_valid == False
    assert any('age' in error.lower() and 'minimum' in error.lower() for error in result.errors)


def test_value_above_maximum_fails(validator, valid_input):
    """Test that value above maximum threshold fails."""
    invalid_input = valid_input.copy()
    invalid_input['age'] = 95  # Above maximum of 90

    result = validator.validate(invalid_input)

    assert result.is_valid == False
    assert any('age' in error.lower() and 'maximum' in error.lower() for error in result.errors)


def test_unusual_age_generates_warning(validator, valid_input):
    """Test that unusual (but valid) age generates warning."""
    unusual_input = valid_input.copy()
    unusual_input['age'] = 35  # Valid but unusually young

    result = validator.validate(unusual_input)

    # Should be valid but with warnings
    assert result.is_valid == True
    assert len(result.warnings) > 0


def test_high_voice_feature_generates_warning(validator, valid_input):
    """Test that unusually high voice features generate warnings."""
    unusual_input = valid_input.copy()
    unusual_input['Jitter(%)'] = 0.015  # High but not invalid

    result = validator.validate(unusual_input)

    # Should be valid but may have warnings
    assert result.is_valid == True


def test_extra_features_generate_info(validator, valid_input):
    """Test that extra features generate info messages."""
    extra_input = valid_input.copy()
    extra_input['extra_feature'] = 123

    result = validator.validate(extra_input)

    # Should still be valid
    assert result.is_valid == True


def test_get_valid_range(validator):
    """Test getting valid range for a feature."""
    age_range = validator.get_valid_range('age')

    assert age_range is not None
    assert 'min' in age_range
    assert 'max' in age_range
    assert age_range['min'] == 30
    assert age_range['max'] == 90


def test_get_valid_range_unknown_feature(validator):
    """Test getting valid range for unknown feature returns None."""
    result = validator.get_valid_range('unknown_feature')

    assert result is None


def test_validation_result_string_representation():
    """Test ValidationResult string formatting."""
    result = ValidationResult()
    result.add_error("Error 1")
    result.add_warning("Warning 1")
    result.add_info("Info 1")

    string_repr = str(result)

    assert "Error 1" in string_repr
    assert "Warning 1" in string_repr
    assert "Info 1" in string_repr


def test_multiple_errors(validator):
    """Test that multiple errors are all captured."""
    invalid_input = {
        'age': "not a number",
        'sex': 2,
        'test_time': -10,
        # Missing other features
    }

    result = validator.validate(invalid_input)

    assert result.is_valid == False
    assert len(result.errors) > 1  # Should have multiple errors


def test_edge_case_values(validator, valid_input):
    """Test edge case values at boundaries."""
    # Test minimum valid age
    edge_input = valid_input.copy()
    edge_input['age'] = 30  # Exact minimum

    result = validator.validate(edge_input)
    assert result.is_valid == True

    # Test maximum valid age
    edge_input['age'] = 90  # Exact maximum
    result = validator.validate(edge_input)
    assert result.is_valid == True


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
