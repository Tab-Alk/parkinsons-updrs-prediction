"""
Model Serialization Script for Parkinson's UPDRS Prediction
=============================================================

This script extracts the trained Random Forest model and preprocessing components
from the Assignment #1 notebook and serializes them for deployment.

It recreates the exact training pipeline to ensure consistency between
training and deployment predictions.
"""

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

print("="*70)
print("Parkinson's UPDRS Model Serialization")
print("="*70)

# Create models directory if it doesn't exist
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

# Step 1: Load the dataset
print("\n[1/6] Loading dataset...")
parkinsons = pd.read_csv('parkinsons_updrs.data')
print(f"   Dataset loaded: {parkinsons.shape[0]} recordings, {parkinsons.shape[1]} columns")

# Step 2: Prepare features and target (matching notebook cell 28-29)
print("\n[2/6] Preparing features and target...")
X = parkinsons.drop(['subject#', 'motor_UPDRS', 'total_UPDRS'], axis=1)
y = parkinsons['total_UPDRS']

# Train-test split (matching notebook cell 29)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)
print(f"   Training set: {len(X_train)} recordings")
print(f"   Test set: {len(X_test)} recordings")

# Step 3: Outlier handling - Winsorization at 1st and 99th percentiles (matching notebook cell 31)
print("\n[3/6] Applying outlier handling (Winsorization)...")
voice_features_for_capping = ['Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP',
                               'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11',
                               'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE']

features_to_cap = [f for f in voice_features_for_capping if f in X_train.columns]
capped_count = 0

for feature in features_to_cap:
    lower_percentile = X_train[feature].quantile(0.01)
    upper_percentile = X_train[feature].quantile(0.99)

    train_capped = ((X_train[feature] < lower_percentile) | (X_train[feature] > upper_percentile)).sum()
    capped_count += train_capped

    X_train[feature] = X_train[feature].clip(lower=lower_percentile, upper=upper_percentile)
    X_test[feature] = X_test[feature].clip(lower=lower_percentile, upper=upper_percentile)

print(f"   Capped {capped_count} values ({capped_count / (len(X_train) * len(features_to_cap)) * 100:.2f}% of training data)")

# Step 4: Feature engineering (matching notebook cell 33)
print("\n[4/6] Engineering features...")
# Voice quality ratio
X_train['voice_quality_ratio'] = X_train['HNR'] / (X_train['NHR'] + 0.001)
X_test['voice_quality_ratio'] = X_test['HNR'] / (X_test['NHR'] + 0.001)

# Test time squared
X_train['test_time_squared'] = X_train['test_time'] ** 2
X_test['test_time_squared'] = X_test['test_time'] ** 2
print("   Created features: voice_quality_ratio, test_time_squared")

# Step 5: Feature selection - drop correlated features (matching notebook cell 36)
print("\n[5/6] Selecting final features...")
correlated_features = ['Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP',
                       'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:DDA']

X_train = X_train.drop(columns=correlated_features)
X_test = X_test.drop(columns=correlated_features)
print(f"   Final feature count: {X_train.shape[1]} features")
print(f"   Features: {X_train.columns.tolist()}")

# Step 6: Feature scaling (matching notebook cell 38)
print("\n[6/6] Fitting StandardScaler...")
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = pd.DataFrame(
    scaler.transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)
print("   StandardScaler fitted on training data")

# Train the optimized Random Forest with best parameters (from notebook cell 59)
print("\n[Training] Training Random Forest with optimal parameters...")
print("   Parameters from GridSearchCV:")
print("   - max_depth: None")
print("   - max_features: sqrt")
print("   - min_samples_leaf: 1")
print("   - min_samples_split: 2")
print("   - n_estimators: 100")

best_rf = RandomForestRegressor(
    max_depth=None,
    max_features='sqrt',
    min_samples_leaf=1,
    min_samples_split=2,
    n_estimators=100,
    random_state=42
)

best_rf.fit(X_train_scaled, y_train)
print("   Model training complete!")

# Evaluate on test set to verify
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
y_test_pred = best_rf.predict(X_test_scaled)
test_r2 = r2_score(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print("\n[Verification] Model Performance:")
print(f"   Test R²: {test_r2:.4f}")
print(f"   Test MAE: {test_mae:.4f}")
print(f"   Test RMSE: {test_rmse:.4f}")

# Save the model
print("\n[Saving] Serializing model artifacts...")
model_path = models_dir / "random_forest_model.pkl"
joblib.dump(best_rf, model_path)
print(f"   ✓ Model saved to: {model_path}")

# Save the scaler
scaler_path = models_dir / "scaler.pkl"
joblib.dump(scaler, scaler_path)
print(f"   ✓ Scaler saved to: {scaler_path}")

# Create feature configuration JSON
feature_config = {
    "feature_names": X_train.columns.tolist(),
    "feature_count": len(X_train.columns),
    "model_type": "RandomForestRegressor",
    "model_parameters": {
        "max_depth": None,
        "max_features": "sqrt",
        "min_samples_leaf": 1,
        "min_samples_split": 2,
        "n_estimators": 100,
        "random_state": 42
    },
    "performance_metrics": {
        "test_r2": float(test_r2),
        "test_mae": float(test_mae),
        "test_rmse": float(test_rmse)
    },
    "validation_ranges": {
        "age": {"min": 30, "max": 90, "dataset_min": 36, "dataset_max": 85},
        "sex": {"min": 0, "max": 1},
        "test_time": {"min": -5, "max": 250, "dataset_min": -4.26, "dataset_max": 215.49},
        "Jitter(%)": {"min": 0.0001, "max": 0.15, "dataset_min": 0.00083, "dataset_max": 0.09999},
        "Jitter(Abs)": {"min": 0.000001, "max": 0.0005, "dataset_min": 0.000002, "dataset_max": 0.000446},
        "Shimmer": {"min": 0.005, "max": 0.3, "dataset_min": 0.0095, "dataset_max": 0.2197},
        "Shimmer:APQ11": {"min": 0.002, "max": 0.3, "dataset_min": 0.00249, "dataset_max": 0.27546},
        "NHR": {"min": 0.0001, "max": 0.8, "dataset_min": 0.000286, "dataset_max": 0.74826},
        "HNR": {"min": 1, "max": 40, "dataset_min": 1.659, "dataset_max": 37.875},
        "RPDE": {"min": 0.1, "max": 1.0, "dataset_min": 0.15102, "dataset_max": 0.96608},
        "DFA": {"min": 0.5, "max": 0.9, "dataset_min": 0.51404, "dataset_max": 0.8656},
        "PPE": {"min": 0.02, "max": 0.8, "dataset_min": 0.021983, "dataset_max": 0.73173}
    },
    "feature_descriptions": {
        "age": "Patient age in years",
        "sex": "Gender (0=male, 1=female)",
        "test_time": "Days since baseline measurement",
        "Jitter(%)": "Frequency variation in voice (%)",
        "Jitter(Abs)": "Absolute jitter measure",
        "Shimmer": "Amplitude variation in voice",
        "Shimmer:APQ11": "Eleven-point amplitude perturbation quotient",
        "NHR": "Noise-to-harmonics ratio",
        "HNR": "Harmonics-to-noise ratio",
        "RPDE": "Recurrence period density entropy (nonlinear complexity)",
        "DFA": "Detrended fluctuation analysis (fractal scaling exponent)",
        "PPE": "Pitch period entropy",
        "voice_quality_ratio": "Engineered feature: HNR / (NHR + 0.001)",
        "test_time_squared": "Engineered feature: test_time²"
    },
    "feature_importance": {
        feature: float(importance)
        for feature, importance in zip(X_train.columns, best_rf.feature_importances_)
    }
}

config_path = models_dir / "feature_config.json"
with open(config_path, 'w') as f:
    json.dump(feature_config, f, indent=2)
print(f"   ✓ Feature config saved to: {config_path}")

print("\n" + "="*70)
print("Model serialization complete!")
print("="*70)
print("\nGenerated files:")
print(f"  1. {model_path} ({model_path.stat().st_size / 1024:.1f} KB)")
print(f"  2. {scaler_path} ({scaler_path.stat().st_size / 1024:.1f} KB)")
print(f"  3. {config_path} ({config_path.stat().st_size / 1024:.1f} KB)")
print("\nNext steps:")
print("  1. Run: pip install -r requirements.txt")
print("  2. Run: streamlit run app/streamlit_app.py")
print("="*70)
