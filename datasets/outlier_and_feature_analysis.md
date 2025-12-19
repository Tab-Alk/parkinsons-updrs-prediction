# Outlier Analysis & Feature Engineering Strategy

## 📊 OUTLIER ANALYSIS (Based on Box Plots in Cell 23)

### Current Situation:
- **Cell 23** creates box plots showing outliers in all 16 voice features
- Outliers are clearly visible in multiple features
- **NO outlier handling has been implemented yet**

### Features with Notable Outliers (from visual inspection):

**High Outlier Features:**
1. **NHR** (Noise-to-Harmonics Ratio) - Many extreme outliers
2. **Jitter(%)** - Several high outliers
3. **Jitter(Abs)** - Multiple extreme values
4. **Jitter:PPQ5** - High-end outliers
5. **Shimmer features** - Moderate outliers

**Lower Outlier Features:**
1. **HNR** - Few outliers
2. **DFA** - Minimal outliers
3. **RPDE** - Few outliers

---

## 🎯 OUTLIER HANDLING STRATEGIES

### Strategy 1: **Domain-Informed Approach (RECOMMENDED)**

**Rationale:** In Parkinson's disease research, extreme voice measurements may represent:
- Real disease severity (not measurement errors)
- Important biological variation
- Critical information for prediction

**Recommendation:**
```python
# Keep outliers BUT cap extreme values at 99th percentile (Winsorization)
# This preserves variation while preventing extreme values from dominating

from scipy.stats import mstats

def winsorize_features(X, lower_percentile=0.01, upper_percentile=0.99):
    """
    Cap extreme values at specified percentiles
    """
    X_winsorized = X.copy()

    for col in X.columns:
        if col not in ['subject#', 'sex', 'test_time', 'age']:
            lower = X[col].quantile(lower_percentile)
            upper = X[col].quantile(upper_percentile)
            X_winsorized[col] = X[col].clip(lower=lower, upper=upper)

    return X_winsorized

# Usage:
# X_train_winsorized = winsorize_features(X_train)
# X_test_winsorized = winsorize_features(X_test)
```

**Pros:**
- Preserves biological variation
- Reduces impact of extreme outliers
- Maintains data distribution shape
- Keeps all samples (no data loss)

**Cons:**
- Still allows some outliers
- May not help if outliers are measurement errors

---

### Strategy 2: **IQR-Based Removal**

**Rationale:** Remove statistical outliers beyond 1.5*IQR or 3*IQR

**Recommendation:**
```python
def remove_outliers_iqr(X, y, multiplier=3.0):
    """
    Remove outliers using IQR method
    multiplier=1.5 (standard) or 3.0 (only extreme outliers)
    """
    X_clean = X.copy()
    outlier_mask = pd.Series([False] * len(X), index=X.index)

    voice_features = X.select_dtypes(include=[np.number]).columns
    voice_features = [f for f in voice_features if f not in ['age', 'sex', 'test_time']]

    for feature in voice_features:
        Q1 = X[feature].quantile(0.25)
        Q3 = X[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        # Mark outliers
        feature_outliers = (X[feature] < lower_bound) | (X[feature] > upper_bound)
        outlier_mask = outlier_mask | feature_outliers

    print(f"Removing {outlier_mask.sum()} outlier samples ({outlier_mask.sum()/len(X)*100:.2f}%)")

    return X_clean[~outlier_mask], y[~outlier_mask]

# Usage (use multiplier=3.0 to only remove extreme outliers):
# X_train_clean, y_train_clean = remove_outliers_iqr(X_train, y_train, multiplier=3.0)
```

**Pros:**
- Statistical rigor
- Removes measurement errors
- May improve model performance

**Cons:**
- Loses data samples
- May remove important disease variation
- Could reduce generalization

---

### Strategy 3: **Hybrid Approach (BEST BALANCE)**

**Recommendation:**
```python
# Step 1: Remove only EXTREME outliers (3*IQR)
# Step 2: Winsorize remaining outliers at 99th percentile

def hybrid_outlier_handling(X, y):
    """
    1. Remove extreme outliers (beyond 3*IQR)
    2. Winsorize remaining data at 99th percentile
    """
    # Step 1: Remove extreme outliers
    X_clean, y_clean = remove_outliers_iqr(X, y, multiplier=3.0)

    # Step 2: Winsorize
    X_final = winsorize_features(X_clean, lower_percentile=0.01, upper_percentile=0.99)

    return X_final, y_clean
```

**Why this is best:**
- Removes only truly extreme/erroneous values
- Preserves biological variation
- Minimal data loss
- Balances statistical rigor with domain knowledge

---

### Strategy 4: **Document and Keep (Valid Alternative)**

**Rationale:** Voice variability is a real symptom of Parkinson's

**Recommendation:**
```markdown
**Note on Outliers:** Box plot analysis (Cell 23) identified outliers in voice
features, particularly in NHR, Jitter, and Shimmer measurements. These outliers
were retained because:

1. Voice variability is a genuine symptom of Parkinson's disease
2. Extreme voice measurements may indicate disease severity
3. Outliers appear across many patients (not isolated measurement errors)
4. Removing them could eliminate important predictive information

Instead, we use robust models (Random Forest, SVR) that are less sensitive to
outliers than linear methods.
```

**Pros:**
- No data loss
- Preserves all biological information
- Honest about methodology

**Cons:**
- May impact model performance
- Requires robust models

---

## 🔧 FEATURE ENGINEERING IDEAS

### Category 1: **Composite Voice Quality Features**

```python
# 1. Overall Jitter Score (voice frequency stability)
X['jitter_composite'] = (
    X['Jitter(%)'] +
    X['Jitter(Abs)'] +
    X['Jitter:RAP']
) / 3

# 2. Overall Shimmer Score (voice amplitude stability)
X['shimmer_composite'] = (
    X['Shimmer'] +
    X['Shimmer(dB)'] +
    X['Shimmer:APQ5']
) / 3

# 3. Voice Quality Ratio (signal-to-noise)
X['voice_quality_ratio'] = X['HNR'] / (X['NHR'] + 0.001)  # Avoid division by zero

# 4. Voice Instability Index (combined jitter + shimmer)
X['voice_instability'] = (
    X['jitter_composite'] +
    X['shimmer_composite']
) / 2
```

**Rationale:** These combine related measurements into single meaningful scores.

---

### Category 2: **Interaction Features**

```python
# 5. Age-Disease Interaction
X['age_disease_interaction'] = X['age'] * X['test_time']

# 6. Age-Voice Quality
X['age_voice_quality'] = X['age'] * X['voice_quality_ratio']

# 7. Time-based Disease Progression Rate
# (assuming test_time represents days since baseline)
X['progression_rate'] = X['test_time'] / (X['age'] + 1)
```

**Rationale:** Disease progression may differ by age and over time.

---

### Category 3: **Polynomial Features (Non-linear Effects)**

```python
# 8. Age squared (non-linear age effects)
X['age_squared'] = X['age'] ** 2

# 9. Test time squared (accelerating disease progression)
X['test_time_squared'] = X['test_time'] ** 2

# 10. Logarithmic transformations (for skewed distributions)
X['log_nhr'] = np.log(X['NHR'] + 0.001)
X['log_jitter_pct'] = np.log(X['Jitter(%)'] + 0.001)
```

**Rationale:** Parkinson's progression is non-linear; quadratic terms capture this.

---

### Category 4: **Domain-Specific Medical Features**

```python
# 11. Voice Complexity Score
# Combines entropy measures (RPDE, DFA, PPE)
X['voice_complexity'] = (X['RPDE'] + X['DFA'] + X['PPE']) / 3

# 12. Acoustic Perturbation Index
# Combines multiple perturbation measures
X['acoustic_perturbation'] = (
    X['Jitter(%)'] * 0.4 +
    X['Shimmer'] * 0.4 +
    X['NHR'] * 0.2
)

# 13. Disease Severity Indicator (based on literature)
# Higher values = more severe voice impairment
X['severity_indicator'] = (
    (1 / (X['HNR'] + 1)) * X['NHR'] * X['jitter_composite']
)
```

**Rationale:** Based on medical literature about Parkinson's voice biomarkers.

---

### Category 5: **Statistical Features per Patient**

```python
# 14. Patient-level aggregations (if you have patient# in features)
# NOTE: Only if keeping subject# in features

# Rolling mean of voice features over time (for each patient)
patient_groups = df.groupby('subject#')

# Voice trend (increasing/decreasing over time)
X['voice_trend'] = patient_groups['HNR'].transform(
    lambda x: x.rolling(window=5, min_periods=1).mean()
)
```

**Rationale:** Capture temporal trends in disease progression.

---

## 📋 RECOMMENDED IMPLEMENTATION PLAN

### **Recommended Approach: Start Simple, Then Expand**

#### **Phase 1: Basic Feature Engineering (Add 4-6 features)**
```python
# Most impactful features based on domain knowledge:
1. jitter_composite
2. shimmer_composite
3. voice_quality_ratio
4. age_squared
5. test_time_squared
6. voice_complexity
```

**Why:** These have strong medical/domain justification and low correlation risk.

---

#### **Phase 2: Test Impact**
- Train models with new features
- Compare performance to baseline
- Use feature importance to identify which new features help

---

#### **Phase 3: Expand If Helpful**
- If Phase 1 features improve performance, add:
  - Interaction features
  - More polynomial terms
  - Domain-specific indices

---

## 🎯 FINAL RECOMMENDATIONS

### **For Outliers:**
**Use Strategy 3 (Hybrid Approach)**
1. Remove extreme outliers (3*IQR) - likely measurement errors
2. Winsorize at 99th percentile - preserve variation
3. Document in markdown cell

**Expected Impact:**
- Remove ~1-3% of extreme values
- Cap remaining outliers
- Improve model stability without losing information

---

### **For Feature Creation:**
**Start with 6 features (Phase 1)**

Priority order:
1. **voice_quality_ratio** (HNR/NHR) - Most medically meaningful
2. **jitter_composite** - You mentioned this earlier
3. **shimmer_composite** - You mentioned this earlier
4. **age_squared** - Non-linear age effects
5. **test_time_squared** - Disease progression acceleration
6. **voice_complexity** - Combines entropy measures

**Expected Impact:**
- Better capture non-linear relationships
- Provide domain-meaningful features
- Potentially improve R² by 0.05-0.15

---

## 📝 CODE TEMPLATE FOR NOTEBOOK

```python
## 5a. Outlier Handling

# Hybrid approach: Remove extreme outliers + Winsorization

def hybrid_outlier_handling(X, y, extreme_multiplier=3.0, winsorize_percentile=0.99):
    """
    Two-step outlier handling:
    1. Remove extreme outliers (beyond extreme_multiplier * IQR)
    2. Winsorize remaining data at specified percentile
    """
    from scipy.stats.mstats import winsorize

    X_clean = X.copy()
    outlier_mask = pd.Series([False] * len(X), index=X.index)

    # Identify voice features (exclude demographic/time features)
    voice_features = [col for col in X.columns if col not in ['age', 'sex', 'test_time']]

    # Step 1: Mark extreme outliers for removal
    for feature in voice_features:
        Q1 = X[feature].quantile(0.25)
        Q3 = X[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - extreme_multiplier * IQR
        upper_bound = Q3 + extreme_multiplier * IQR

        feature_outliers = (X[feature] < lower_bound) | (X[feature] > upper_bound)
        outlier_mask = outlier_mask | feature_outliers

    print(f"Step 1: Removing {outlier_mask.sum()} extreme outliers ({outlier_mask.sum()/len(X)*100:.2f}%)")

    # Remove extreme outliers
    X_clean = X_clean[~outlier_mask]
    y_clean = y[~outlier_mask]

    # Step 2: Winsorize remaining outliers
    for feature in voice_features:
        lower = X_clean[feature].quantile(1 - winsorize_percentile)
        upper = X_clean[feature].quantile(winsorize_percentile)
        X_clean[feature] = X_clean[feature].clip(lower=lower, upper=upper)

    print(f"Step 2: Winsorized features at {winsorize_percentile*100}th percentile")
    print(f"Final dataset size: {len(X_clean)} samples")

    return X_clean, y_clean

# Apply to training data (BEFORE train-test split if possible, or after on train only)
# X_train_clean, y_train_clean = hybrid_outlier_handling(X_train, y_train)
```

```python
## 5b. Feature Engineering

def create_engineered_features(X):
    """
    Create domain-informed features for Parkinson's prediction
    """
    X_enhanced = X.copy()

    # 1. Jitter Composite (voice frequency stability)
    if all(col in X.columns for col in ['Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP']):
        X_enhanced['jitter_composite'] = (
            X['Jitter(%)'] + X['Jitter(Abs)'] + X['Jitter:RAP']
        ) / 3

    # 2. Shimmer Composite (voice amplitude stability)
    if all(col in X.columns for col in ['Shimmer', 'Shimmer(dB)', 'Shimmer:APQ5']):
        X_enhanced['shimmer_composite'] = (
            X['Shimmer'] + X['Shimmer(dB)'] + X['Shimmer:APQ5']
        ) / 3

    # 3. Voice Quality Ratio (signal-to-noise)
    if all(col in X.columns for col in ['HNR', 'NHR']):
        X_enhanced['voice_quality_ratio'] = X['HNR'] / (X['NHR'] + 0.001)

    # 4. Age Squared (non-linear age effects)
    if 'age' in X.columns:
        X_enhanced['age_squared'] = X['age'] ** 2

    # 5. Test Time Squared (disease progression acceleration)
    if 'test_time' in X.columns:
        X_enhanced['test_time_squared'] = X['test_time'] ** 2

    # 6. Voice Complexity (entropy measures)
    if all(col in X.columns for col in ['RPDE', 'DFA', 'PPE']):
        X_enhanced['voice_complexity'] = (X['RPDE'] + X['DFA'] + X['PPE']) / 3

    print(f"Created {len(X_enhanced.columns) - len(X.columns)} new features")
    print(f"New features: {[col for col in X_enhanced.columns if col not in X.columns]}")

    return X_enhanced

# Apply BEFORE feature selection (so you can drop correlated features after)
# X_train_enhanced = create_engineered_features(X_train)
# X_test_enhanced = create_engineered_features(X_test)
```

---

## 📊 WHERE TO INSERT IN NOTEBOOK

**Current Structure:**
- Cell 23: Box plots (outlier detection)
- Cell 26: Train-test split markdown
- Cell 27: Prepare features
- Cell 28: Train-test split
- Cell 29: Feature Selection markdown
- Cell 30: Correlation analysis
- Cell 31: Drop correlated features

**Recommended Insertion Points:**

**Option A: Handle outliers BEFORE split (recommended if not patient-aware)**
```
Cell 23: Box plots
→ NEW Cell 24: Markdown "### Outlier Handling"
→ NEW Cell 25: Implement hybrid_outlier_handling()
→ NEW Cell 26: Markdown "### Feature Engineering"
→ NEW Cell 27: Implement create_engineered_features()
[Continue with current cells]
```

**Option B: Handle outliers AFTER split (better for avoiding leakage)**
```
Cell 28: Train-test split
→ NEW Cell 29: Markdown "## 5. Outlier Handling"
→ NEW Cell 30: Apply hybrid_outlier_handling to train set
→ NEW Cell 31: Markdown "## 6. Feature Engineering"
→ NEW Cell 32: Apply create_engineered_features to train and test
[Then continue with feature selection...]
```

I recommend **Option B** to avoid data leakage.

---

## ✅ CHECKLIST

Before implementing:
- [ ] Decide on outlier strategy (Hybrid recommended)
- [ ] Decide on feature set (Start with 6 recommended features)
- [ ] Determine where to insert in notebook (After train-test split recommended)
- [ ] Test on small sample first
- [ ] Document rationale in markdown cells
- [ ] Compare model performance before/after
- [ ] Update feature importance analysis with new features

After implementing:
- [ ] Check for new highly correlated features (may need to drop some)
- [ ] Verify feature scaling includes new features
- [ ] Re-run all models
- [ ] Compare performance metrics
- [ ] Update feature importance visualization
- [ ] Document impact in conclusions section
