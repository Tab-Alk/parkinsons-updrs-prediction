# Parkinson's Disease UPDRS Prediction System

A machine learning web application for predicting Parkinson's disease severity (UPDRS scores) from voice biomarker measurements.

## Table of Contents

- [Overview](#overview)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
- [Input Features](#input-features)
- [Sample Inputs & Outputs](#sample-inputs--outputs)
- [Architecture](#architecture)
- [Testing](#testing)
- [Cloud Deployment](#cloud-deployment)
- [Project Structure](#project-structure)
- [References](#references)

## Overview

This application deploys a **Random Forest Regressor** model trained on voice measurements from 42 Parkinson's disease patients. The model predicts the **UPDRS (Unified Parkinson's Disease Rating Scale)** score, which measures disease severity from 7 (minimal symptoms) to 55+ (severe symptoms).

Traditional UPDRS assessment requires in-person clinical evaluation, which is time-consuming, expensive, and subjective. This system enables **remote, objective, and frequent monitoring** using only voice recordings, making disease progression tracking more accessible and consistent for both clinicians and patients. The application provides real-time predictions through an interactive web interface, supporting individual assessments and batch processing of multiple patient records.

## Model Performance

**Algorithm:** Tuned Random Forest Regressor

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Test R² | **0.7958** | > 0.7 | Exceeded |
| Test MAE | **3.81 points** | < 5.0 | Met |
| Test RMSE | **4.79 points** | - | - |
| Cross-Val RMSE | **5.34 ± 0.24** | - | - |

**Key Features (by importance):**
1. Age (45.5%)
2. DFA - Detrended Fluctuation Analysis (13.4%)
3. HNR - Harmonics-to-Noise Ratio (7.7%)
4. Jitter(Abs) - Absolute Jitter (5.3%)
5. RPDE - Recurrence Period Density Entropy (5.2%)

## Installation

**Prerequisites:** Python 3.8+ and pip

```bash
# Clone repository
git clone https://github.com/Tab-Alk/parkinsons-updrs-prediction.git
cd parkinsons-updrs-prediction

# Install dependencies
pip install -r requirements.txt

# Generate model files
python save_model.py
```

The `save_model.py` script trains the Random Forest model and saves artifacts to `models/` directory (takes 2-3 minutes).

## Usage

**Run the application:**
```bash
streamlit run app/streamlit_app.py
```

The app opens at `http://localhost:8501` with three input methods:
1. **Manual Input** - Interactive form for single patient assessment
2. **Batch Upload** - CSV file upload for multiple patients

**CSV Format Example:**
```csv
age,sex,test_time,Jitter(%),Jitter(Abs),Shimmer,Shimmer:APQ11,NHR,HNR,RPDE,DFA,PPE
65,0,92.0,0.0049,0.000035,0.0253,0.0227,0.0184,21.92,0.5422,0.6436,0.2055
```

## Input Features

All 12 raw features must be provided (2 additional features are auto-generated):

| Feature | Description | Range | Units |
|---------|-------------|-------|-------|
| **age** | Patient age | 30-90 | years |
| **sex** | Gender | 0 or 1 | 0=male, 1=female |
| **test_time** | Days since baseline | -5 to 250 | days |
| **Jitter(%)** | Frequency variation | 0.0001-0.15 | percentage |
| **Jitter(Abs)** | Absolute jitter | 0.000001-0.0005 | - |
| **Shimmer** | Amplitude variation | 0.005-0.3 | - |
| **Shimmer:APQ11** | 11-point amplitude perturbation | 0.002-0.3 | - |
| **NHR** | Noise-to-harmonics ratio | 0.0001-0.8 | - |
| **HNR** | Harmonics-to-noise ratio | 1-40 | dB |
| **RPDE** | Recurrence period density entropy | 0.1-1.0 | - |
| **DFA** | Detrended fluctuation analysis | 0.5-0.9 | - |
| **PPE** | Pitch period entropy | 0.02-0.8 | - |

**Auto-generated features:**
- **voice_quality_ratio** = HNR / (NHR + 0.001)
- **test_time_squared** = test_time²

## Sample Inputs & Outputs

**Example Input (Average Patient):**
```json
{
  "age": 65, "sex": 0, "test_time": 92.0,
  "Jitter(%)": 0.0049, "Jitter(Abs)": 0.000035,
  "Shimmer": 0.0253, "Shimmer:APQ11": 0.0227,
  "NHR": 0.0184, "HNR": 21.92,
  "RPDE": 0.5422, "DFA": 0.6436, "PPE": 0.2055
}
```

**Expected Output:**
- UPDRS Score: 26-32
- Severity: Moderate

Additional test cases available in `data/sample_inputs.json` covering mild (15-25), moderate (26-40), and severe (40+) ranges.

## Architecture

**Data Flow:**
```
User Input → Validation → Preprocessing → Model Prediction → Output Display
             (3-tier)     (feature eng.)   (Random Forest)    (Streamlit UI)
```

**Module Structure:**
- **app/validators.py**: Three-tier input validation (errors, warnings, info)
- **app/preprocessing.py**: Feature engineering and StandardScaler transformation
- **app/model.py**: Model wrapper and prediction logic
- **app/streamlit_app.py**: Web interface with session state management
- **save_model.py**: Model training and serialization
- **tests/**: Comprehensive unit tests (43 tests covering all modules)

## Testing

**Run all tests:**
```bash
pytest tests/ -v
```

**Test coverage:**
- **test_model.py** (17 tests): Model loading, predictions, severity classification, batch processing
- **test_preprocessing.py** (9 tests): Feature engineering, scaling, edge cases
- **test_validators.py** (17 tests): Input validation, error handling, warnings

**Coverage report:**
```bash
pytest tests/ --cov=app --cov-report=html
```

All 43 tests validate robustness, accuracy, and error handling across the entire prediction pipeline.

## Cloud Deployment

**Deploy to Streamlit Cloud (Free):**

1. Push code to GitHub (if not already done)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app"
5. Configure deployment:
   - Repository: `Tab-Alk/parkinsons-updrs-prediction`
   - Branch: `main`
   - Main file: `app/streamlit_app.py`
6. Click "Deploy"

Deployment completes in 3-5 minutes. No credit card required.

## Project Structure

```
parkinsons-updrs-prediction/
├── app/
│   ├── streamlit_app.py      # Main web interface
│   ├── model.py               # Prediction wrapper
│   ├── preprocessing.py       # Feature engineering
│   └── validators.py          # Input validation
├── models/
│   ├── random_forest_model.pkl
│   ├── scaler.pkl
│   └── feature_config.json
├── data/
│   └── sample_inputs.json
├── tests/
│   ├── test_model.py
│   ├── test_preprocessing.py
│   └── test_validators.py
├── save_model.py
├── requirements.txt
└── README.md
```

## References

- **Dataset**: [UCI Machine Learning Repository - Parkinson's Telemonitoring](https://archive.ics.uci.edu/ml/datasets/Parkinsons+Telemonitoring)
- **Research Paper**: Tsanas, A., Little, M.A., McSharry, P.E., Ramig, L.O. (2009). "Accurate telemonitoring of Parkinson's disease progression by non-invasive speech tests." *IEEE Transactions on Biomedical Engineering*.
- **UPDRS Scale**: [Movement Disorder Society - UPDRS](https://www.movementdisorders.org/MDS/MDS-Rating-Scales/MDS-Unified-Parkinsons-Disease-Rating-Scale-MDS-UPDRS.htm)
