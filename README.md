# Parkinson's Disease UPDRS Prediction System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A machine learning web application for predicting Parkinson's disease severity (UPDRS scores) from voice biomarker measurements.

## 📋 Table of Contents

- [Overview](#overview)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
- [Input Features](#input-features)
- [Sample Inputs & Outputs](#sample-inputs--outputs)
- [Architecture](#architecture)
- [Testing](#testing)
- [Cloud Deployment](#cloud-deployment)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)

## 🎯 Overview

This application deploys a **Random Forest Regressor** model trained on voice measurements from 42 Parkinson's disease patients. The model predicts the **UPDRS (Unified Parkinson's Disease Rating Scale)** score, which measures disease severity from 7 (minimal symptoms) to 55+ (severe symptoms).

### Business Problem

Parkinson's disease is a progressive neurodegenerative disorder. Traditional UPDRS assessment requires in-person clinical evaluation, which is:
- Time-consuming and expensive
- Subjective and variable between clinicians
- Difficult for patients with mobility limitations

### Solution

This system enables **remote, objective, and frequent monitoring** using only voice recordings, making disease progression tracking more accessible and consistent.

## 📊 Model Performance

**Algorithm:** Tuned Random Forest Regressor

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Test R² | **0.7958** | > 0.7 | ✅ **Exceeded** |
| Test MAE | **3.81 points** | < 5.0 | ✅ **Met** |
| Test RMSE | **4.79 points** | - | - |
| Cross-Val RMSE | **5.34 ± 0.24** | - | - |

**Key Features (by importance):**
1. Age (45.5%)
2. DFA - Detrended Fluctuation Analysis (13.4%)
3. HNR - Harmonics-to-Noise Ratio (7.7%)
4. Jitter(Abs) - Absolute Jitter (5.3%)
5. RPDE - Recurrence Period Density Entropy (5.2%)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Download the Repository

```bash
# If using Git
git clone <repository-url>
cd parkinsons+telemonitoring

# Or download and extract the ZIP file
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Generate Model Artifacts

```bash
python save_model.py
```

This script will:
- Load the Parkinson's dataset
- Train the optimized Random Forest model
- Save the model, scaler, and configuration files to `models/`
- Take approximately **2-3 minutes** to complete

**Expected output:**
```
Parkinson's UPDRS Model Serialization
======================================================================
[1/6] Loading dataset...
[2/6] Preparing features and target...
[3/6] Applying outlier handling (Winsorization)...
[4/6] Engineering features...
[5/6] Selecting final features...
[6/6] Fitting StandardScaler...
[Training] Training Random Forest with optimal parameters...
[Verification] Model Performance:
   Test R²: 0.7958
   Test MAE: 3.81
[Saving] Serializing model artifacts...
   ✓ Model saved
   ✓ Scaler saved
   ✓ Feature config saved
======================================================================
```

## 💻 Usage

### Local Deployment

```bash
streamlit run app/streamlit_app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Input Methods

The application supports three input methods:

#### 1. Manual Input

Enter all 14 required voice measurements through an interactive form:
- Demographics: age, sex
- Time: test_time
- Jitter features: Jitter(%), Jitter(Abs)
- Shimmer features: Shimmer, Shimmer:APQ11
- Voice quality: NHR, HNR
- Complexity: RPDE, DFA, PPE

#### 2. Sample Cases

Click pre-loaded sample buttons to quickly test:
- Young Patient, Mild Symptoms
- Elderly Patient, Severe Symptoms
- Average Patient (Dataset Mean)
- Female Patient, Moderate Symptoms
- Early Stage, Recent Diagnosis

#### 3. CSV Upload (Batch Prediction)

Upload a CSV file with multiple patient records:

**CSV Format:**
```csv
age,sex,test_time,Jitter(%),Jitter(Abs),Shimmer,Shimmer:APQ11,NHR,HNR,RPDE,DFA,PPE
65,0,92.0,0.0049,0.000035,0.0253,0.0227,0.0184,21.92,0.5422,0.6436,0.2055
78,1,150.0,0.012,0.00008,0.065,0.055,0.045,18.5,0.62,0.72,0.35
```

## 📥 Input Features

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

## 📝 Sample Inputs & Outputs

### Sample 1: Young Patient, Mild Symptoms

**Input:**
```json
{
  "age": 45,
  "sex": 0,
  "test_time": 30.5,
  "Jitter(%)": 0.004,
  "Jitter(Abs)": 0.00003,
  "Shimmer": 0.020,
  "Shimmer:APQ11": 0.018,
  "NHR": 0.015,
  "HNR": 24.0,
  "RPDE": 0.45,
  "DFA": 0.65,
  "PPE": 0.18
}
```

**Expected Output:**
- UPDRS Score: **15-25** (Mild)
- Severity: **Mild**

### Sample 2: Elderly Patient, Severe Symptoms

**Input:**
```json
{
  "age": 78,
  "sex": 1,
  "test_time": 150.0,
  "Jitter(%)": 0.012,
  "Jitter(Abs)": 0.00008,
  "Shimmer": 0.065,
  "Shimmer:APQ11": 0.055,
  "NHR": 0.045,
  "HNR": 18.5,
  "RPDE": 0.62,
  "DFA": 0.72,
  "PPE": 0.35
}
```

**Expected Output:**
- UPDRS Score: **35-48** (Severe)
- Severity: **Severe**

### Sample 3: Average Patient

**Input:**
```json
{
  "age": 65,
  "sex": 0,
  "test_time": 92.0,
  "Jitter(%)": 0.0049,
  "Jitter(Abs)": 0.000035,
  "Shimmer": 0.0253,
  "Shimmer:APQ11": 0.0227,
  "NHR": 0.0184,
  "HNR": 21.92,
  "RPDE": 0.5422,
  "DFA": 0.6436,
  "PPE": 0.2055
}
```

**Expected Output:**
- UPDRS Score: **26-32** (Moderate)
- Severity: **Moderate**

## 🏗️ Architecture

### Data Flow

```
User Input (12 features)
    ↓
Input Validation (validators.py)
    ├── Tier 1: Hard Errors (missing features, invalid types)
    ├── Tier 2: Warnings (unusual values)
    └── Tier 3: Info (contextual feedback)
    ↓
Preprocessing (preprocessing.py)
    ├── Feature Engineering (voice_quality_ratio, test_time_squared)
    ├── Feature Selection (14 final features)
    └── Scaling (StandardScaler)
    ↓
Model Prediction (model.py)
    ├── Random Forest Regressor
    └── UPDRS Score (7-55 range)
    ↓
Output Display (streamlit_app.py)
    ├── Predicted UPDRS Score
    ├── Severity Level (Mild/Moderate/Severe)
    ├── Confidence Context
    └── Validation Feedback
```

### Module Structure

- **app/preprocessing.py**: Feature engineering and scaling
- **app/validators.py**: Three-tier input validation
- **app/model.py**: Model wrapper and prediction logic
- **app/streamlit_app.py**: Web interface
- **save_model.py**: Model serialization script
- **tests/**: Unit tests for all modules

## 🧪 Testing

### Run Unit Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_preprocessing.py -v

# Run with coverage report
pytest tests/ --cov=app --cov-report=html
```

### Test Coverage

- **test_preprocessing.py**: Tests feature engineering, scaling, batch processing
- **test_validators.py**: Tests three-tier validation system
- **test_model.py**: Tests model loading, predictions, severity classification

### Manual Testing

Use the sample cases in `data/sample_inputs.json` to verify:
1. Young patient (mild) → UPDRS 15-25
2. Elderly patient (severe) → UPDRS 35-48
3. Average patient → UPDRS 26-32

## ☁️ Cloud Deployment

### Streamlit Cloud (Free & Easiest)

1. **Create a GitHub repository**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Deploy to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Main file: `app/streamlit_app.py`
   - Click "Deploy"

3. **Access your app**:
   - URL: `https://[username]-parkinsons-updrs.streamlit.app`
   - Deployment takes 2-5 minutes

**No credit card required. 100% free.**

## 🔧 Troubleshooting

### Issue: "Model file not found"

**Solution:**
```bash
python save_model.py
```
Ensure `models/` directory is created with `.pkl` and `.json` files.

### Issue: "Module not found" errors

**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: Streamlit won't start

**Solution:**
```bash
# Check if port 8501 is in use
lsof -i :8501  # macOS/Linux
netstat -ano | findstr :8501  # Windows

# Or specify a different port
streamlit run app/streamlit_app.py --server.port 8502
```

### Issue: Predictions seem incorrect

**Solution:**
1. Verify model was trained: Check `models/feature_config.json` for performance metrics
2. Check input values are in valid ranges
3. Review validation warnings in the UI

### Issue: CSV upload fails

**Solution:**
- Ensure CSV has all 12 required columns (exact names, case-sensitive)
- Check for special characters in column names
- Verify all values are numeric
- Remove any header rows beyond the first

## 📁 Project Structure

```
parkinsons+telemonitoring/
├── app/
│   ├── __init__.py
│   ├── streamlit_app.py          # Main Streamlit application
│   ├── model.py                   # Model loading & prediction
│   ├── preprocessing.py           # Feature engineering & scaling
│   └── validators.py              # Input validation
├── models/
│   ├── random_forest_model.pkl    # Trained model (generated)
│   ├── scaler.pkl                 # Fitted scaler (generated)
│   └── feature_config.json        # Feature metadata (generated)
├── data/
│   ├── parkinsons_updrs.data      # Original dataset
│   └── sample_inputs.json         # Sample test cases
├── tests/
│   ├── __init__.py
│   ├── test_model.py              # Model tests
│   ├── test_preprocessing.py      # Preprocessing tests
│   └── test_validators.py         # Validation tests
├── notebooks/
│   └── parkinsons_updrs_prediction.ipynb  # Training notebook (Assignment #1)
├── save_model.py                  # Model serialization script
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── .gitignore                     # Git ignore rules
```

## 📚 References

- **Dataset**: [UCI Machine Learning Repository - Parkinson's Telemonitoring](https://archive.ics.uci.edu/ml/datasets/Parkinsons+Telemonitoring)
- **Research Paper**: Tsanas, A., Little, M.A., McSharry, P.E., Ramig, L.O. (2009). "Accurate telemonitoring of Parkinson's disease progression by non-invasive speech tests." *IEEE Transactions on Biomedical Engineering*.
- **UPDRS Scale**: [Movement Disorder Society - UPDRS](https://www.movementdisorders.org/MDS/MDS-Rating-Scales/MDS-Unified-Parkinsons-Disease-Rating-Scale-MDS-UPDRS.htm)

## 👨‍💻 Author

**Assignment #2: Deploying a Machine Learning Model**
Course: Machine Learning Engineering
Date: December 2025

## 📄 License

This project is for educational purposes as part of a university assignment.

---

**Need help?** Check the [Troubleshooting](#troubleshooting) section or review the inline code comments in each module.
