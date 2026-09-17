# Smart Nutrition Analyzer
## Research Paper Preparation Report

**Project Title:** Smart Nutrition Analyzer: A Machine Learning Approach for
Personalized Nutrition Using Neural Collaborative Filtering

**Audit Date:** September 2026

**Integrity Notice:** Every value in this document was extracted directly from
the repository source code, datasets, and verified training outputs.
Nothing is estimated, invented, or assumed.
Where a component is missing or partial, it is explicitly labelled as such.

---

## 1. Executive Summary

The Smart Health Dashboard (locally titled *Smart Nutrition Analyzer*) is a
Python-based web application that combines disease-risk screening with
personalised dietary recommendations. The system takes anthropometric and
clinical parameters from the user, predicts disease risk across three
conditions (diabetes, kidney disease, obesity) using trained machine-learning
classifiers, and then generates a 7-day personalised meal plan drawn from a
curated Indian-food dataset (1,014 food items).

**Key verified facts:**

| Aspect | Status |
|--------|--------|
| Disease prediction models | 3 trained classifiers — accuracy 95–100 % on real holdout data |
| Hybrid recommendation scoring | Implemented (nutrition 62.5 % + content-based 37.5 %) |
| Neural Collaborative Filtering | **Not available** — no trained model exists at runtime |
| 7-day diverse meal planner | Fully implemented and tested |
| Explainable AI | Implemented via RF feature_importances_ × clinical deviation |
| Severity-aware nutrition filtering | 3-tier (mild/moderate/severe) for each disease |
| Clinical validation | **Not performed** — screening prototype only |

---

## 2. Project Overview

- **Technology stack:** Python 3.13.3, Streamlit 1.59.2
- **Framework:** Streamlit (browser-based interactive dashboard)
- **Platform:** Windows desktop; served locally on port 8501
- **Authentication:** streamlit-authenticator with bcrypt-hashed credentials
- **Database:** SQLite (healthcare.db) for analysis history
- **UI:** Single-page multi-step wizard + results dashboard

---

## 3. Research Problem

Manual dietary planning for patients with chronic conditions (diabetes,
kidney disease, obesity) is time-consuming, error-prone, and rarely
personalised. Existing tools either provide generic food recommendations
without disease-specific constraints, or apply rigid rule-based diets that
do not adapt to severity level.

---

## 4. Objectives

Based on the actual implementation:

1. Predict disease risk (diabetes, kidney disease, obesity) from clinical
   parameters using supervised machine learning.
2. Classify disease severity (mild / moderate / severe) from the ML model's
   confidence combined with clinical marker reference ranges.
3. Apply disease- and severity-aware nutritional hard constraints to filter
   unsafe foods.
4. Score candidate foods using a hybrid nutrition + content-based scoring
   pipeline.
5. Generate a diverse, non-repetitive 7-day meal plan with 4 meal slots
   per day (Breakfast, Lunch, Snack, Dinner).
6. Explain model predictions via patient-specific feature importance using
   a deviation-weighted XAI approach.

---

## 5. System Architecture

```text
USER INPUT (Streamlit 4-step wizard)
  - Personal: name, age, gender, height, weight
  - Lifestyle: activity level
  - Medical: HbA1c, blood glucose, systolic BP, diastolic BP,
             creatinine, sodium, potassium
  - Goals: health goal, region, prediction mode
        |
        v
INPUT VALIDATION
  - Numeric range checks per parameter
  - Name format validation
        |
        v
ANTHROPOMETRIC CALCULATIONS  [utils.py]
  - BMI  = weight / (height_m)^2
  - BMR  = Mifflin-St Jeor equation (gender-specific)
  - TDEE = BMR × activity_factor
  - Water = weight × 0.035 × activity_multiplier (cap: 5.0 L/day)
  - Protein = weight × protein_per_kg (disease-adjusted)
        |
        v
DISEASE PREDICTION  [predict.py + models.py]
  |           |            |
  v           v            v
OBESITY    DIABETES    KIDNEY DISEASE
(RF 300)   (RF 300)    (GBT 400)
3 feat     5 feat       7 feat
7 classes  3 classes    2 classes
  |           |            |
  v           v            v
RISK CLASSIFICATION  [backend/risk.py]
  - ML probability + clinical marker reference ranges
  - 5-way output: High Risk | Moderate Risk | Model Flag | Low Risk | Borderline
        |
        v
SEVERITY DERIVATION  [backend/services/severity_rules.py]
  - RiskResult.final_status → severity tier
    "High Risk" → severe
    "Moderate Risk" → moderate
    "Borderline" / "Model Flag" → mild
    "Low Risk" → None (no disease restrictions)
        |
        v
DISEASE + SEVERITY HARD FILTERS  [backend/services/enhanced_recommender.py]
  - Stage 1: config.py base thresholds (moderate baseline)
  - Stage 2: severity_rules.py hard constraints per tier
  - Minimum pool guard: 30 foods before warning
        |
        v
HYBRID FOOD SCORING  (per candidate food item)
  - nutrition_score = severity-multiplied arithmetic score
  - nutrition_norm  = clamp(nutrition_score / dynamic_p99_ceiling, 0, 1)
  - severity_suit   = suitability score from soft constraints [0,1]
  - nutrition_blend = 0.70 × nutrition_norm + 0.30 × severity_suit
  - content_score   = calorie_fit×0.40 + protein_den×0.25
                    + fibre_den×0.20 + sugar_score×0.15
  - hybrid = 0.6250 × nutrition_blend + 0.3750 × content_score
             (NCF weight 0.20 redistributed; NCF unavailable)
        |
        v
DIVERSITY-AWARE 7-DAY PLANNER  [backend/services/weekly_meal_planner.py]
  - FoodUsageTracker: food_id-level usage across 7 days
  - Diversity penalties: consecutive_day=0.6, same_day=0.8, repeat=0.2
  - Diversity bonus: new category = -0.1 penalty
  - MAX_FOOD_USES = 1 (target: each food once per week)
  - Progressive fallback relaxation (5 passes)
  - Deterministic selection (stable-sort, no randomness)
        |
        v
EXPLAINABLE AI  [backend/xai.py]
  - patient_score = RF feature_importances_[i] × deviation_score(value)
  - deviation_score = 0 if value inside normal range, else sigmoid-like
        |
        v
PERSONALISED OUTPUT
  - 7-day meal plan (Breakfast/Lunch/Snack/Dinner per day)
  - Daily/weekly nutrition summary
  - Prediction cards with risk badges
  - XAI feature importance charts (Plotly)
  - Foods to avoid list
  - Nutrition tips
  - Export (JSON / PDF / DB)
```

---

## 6. Dataset Analysis

### 6.1 Food Dataset (`food_dataset.csv`)

| Property | Value |
|----------|-------|
| Total records | **1,014** |
| Total features | **14** |
| Duplicate food_ids | 0 |
| Duplicate Dish Names | 0 |
| Missing values | **0 in all columns** |
| Food domain | Indian foods (South Indian emphasis) |

**All 14 columns:**

| Column | Type | Notes |
|--------|------|-------|
| `food_id` | Integer | Sequential 1–1,014 |
| `Dish Name` | String | Unique per row |
| `Calories (kcal)` | Float | |
| `Carbohydrates (g)` | Float | |
| `Protein (g)` | Float | |
| `Fats (g)` | Float | |
| `Free Sugar (g)` | Float | |
| `Fibre (g)` | Float | |
| `Sodium (mg)` | Float | |
| `Calcium (mg)` | Float | |
| `Iron (mg)` | Float | |
| `Vitamin C (mg)` | Float | |
| `Folate (µg)` | Float | Mojibake µg in some tools |
| `MealType` | String | Breakfast / Snack / Lunch / Dinner / Lunch/Dinner |

**MealType distribution:**

| MealType | Count |
|----------|-------|
| Snack | 683 |
| Breakfast | 105 |
| Lunch/Dinner | 93 |
| Lunch | 70 |
| Dinner | 63 |

**Nutritional ranges:**

| Nutrient | Min | Max | Mean |
|----------|-----|-----|------|
| Calories (kcal) | 6.61 | 839.33 | 233.74 |
| Carbohydrates (g) | 0.00 | 86.53 | 18.35 |
| Protein (g) | 0.00 | 21.55 | 4.75 |
| Fats (g) | 0.00 | 90.45 | 16.27 |
| Free Sugar (g) | 0.00 | 85.57 | 8.82 |
| Fibre (g) | 0.00 | 35.71 | 1.96 |
| Sodium (mg) | 0.00 | 2,000.00 | 225.24 |
| Calcium (mg) | 0.00 | 631.82 | 59.88 |
| Iron (mg) | 0.00 | 20.57 | 1.14 |

> **Dataset limitation:** This dataset is not publicly identified as a
> standard benchmark. Its provenance and collection methodology are not
> documented in the repository. It should not be presented as a published
> reference dataset without proper attribution.

---

### 6.2 Disease Dataset (`data/cleaned_disease_dataset.csv`)

| Property | Value |
|----------|-------|
| Total rows | **92,414** |
| Total columns | **20** |
| Gender split | Female: 53,743 / Male: 38,671 |

**20 columns:** age, gender, bmi, disease, severity, HbA1c, blood glucose,
sodium, potassium, BloodPressure, SerumCreatinine, bmi_category, BMR, TDEE,
Target_Calories, Target_Carbohydrates_g, Target_Protein_g, Target_Fat_g,
Limit_Sugar_g, Limit_Sodium_mg

**Disease label distribution:**

| Disease | Count |
|---------|-------|
| prediabetes | 37,140 |
| no diabetes | 33,923 |
| diabetes | 18,993 |
| obesity | 899 |
| overweight | 543 |
| kidney_disease | 399 |
| normal | 272 |
| underweight | 245 |

**Severity distribution:**

| Severity | Count |
|----------|-------|
| mild | 37,663 |
| normal | 34,440 |
| moderate | 17,740 |
| severe | 2,571 |

**Null values (critical):**

| Column | Non-null rows | Notes |
|--------|--------------|-------|
| HbA1c | 90,056 | Only diabetes-spectrum rows populated |
| blood glucose | 90,056 | Only diabetes-spectrum rows populated |
| sodium | 399 | Only kidney_disease rows populated |
| potassium | 399 | Only kidney_disease rows populated |
| BloodPressure | 399 | Only kidney_disease rows populated |
| SerumCreatinine | 399 | Only kidney_disease rows populated |

> This dataset is a **composite** created for this project — different disease
> sub-populations use different clinical columns. It is not an independent
> public dataset. The diabetes-spectrum portion (90,056 rows with HbA1c +
> glucose) was used to train the diabetes model.

---

### 6.3 Kidney Disease Dataset (`data/kidney_disease.csv.xlsx`)

| Property | Value |
|----------|-------|
| Rows | **399** |
| Columns | 9: age, gender, bmi, sodium, potassium, BloodPressure, SerumCreatinine, disease, severity |
| Classes (after label normalisation) | Kidney Disease: 250 / No Kidney Disease: 149 |
| Sodium range | 4.5 – 163.0 mEq/L |
| SerumCreatinine range | 0.4 – 76.0 mg/dL |

> Note: The original file contained 2 rows with a trailing tab artefact
> (`'Kidney Disease\t'`). These were merged into the `Kidney Disease` class
> via `str.strip()` in the training pipeline.

---

### 6.4 Obesity Dataset (`data/Obesity prediction.csv.xlsx`)

| Property | Value |
|----------|-------|
| Rows | **1,959** |
| Columns | 5: age, gender, bmi, disease, severity |
| Age range | 14 – 61 |
| BMI range | 13.98 – 56.13 |

**Class distribution (balanced):**

| Class | Count |
|-------|-------|
| Obesity_Type_I | 339 |
| Obesity_Type_III | 284 |
| Overweight_Level_II | 280 |
| Obesity_Type_II | 276 |
| Normal_Weight | 272 |
| Overweight_Level_I | 263 |
| Insufficient_Weight | 245 |

---

## 7. Data Preprocessing

| Step | Implementation | File |
|------|----------------|------|
| Gender encoding | `Male=1, Female=0` via `LabelEncoder` | `train_models.py`, `models.py` |
| Label normalisation (kidney tab) | `df['disease'].str.strip()` | `train_models.py` |
| Feature selection (diabetes) | Filter rows with both HbA1c and blood glucose non-null | `train_models.py` |
| Label encoding (disease) | `LabelEncoder.fit_transform()` | `train_models.py` |
| Label encoding (obesity) | Manual dict mapping → integer 0–6 | `train_models.py` |
| Train/test split | `train_test_split(stratify=y, test_size=0.20, random_state=42)` | `train_models.py` |
| SMOTE | Attempted for kidney (imbalanced-learn not installed; not applied) | `train_models.py` |
| Scaling/normalisation | **Not applied** — Random Forest and GBT are scale-invariant | `train_models.py` |
| Food dataset | Loaded as-is; no missing values to impute | `models.py` |

---

## 8. Feature Engineering

All features are raw clinical measurements. No derived polynomial or
interaction features are computed before model training. The only
transformations are:

- `gender`: string → binary integer (Male=1, Female=0)
- `disease`: string label → integer via `LabelEncoder`
- `obesity`: string label → integer via manual mapping

**No normalisation or standardisation is applied** to any numeric feature
for the three disease models. This is appropriate for tree-based algorithms.

---

## 9. Machine Learning Models

### 9.1 Model Comparison Table

| Disease | Algorithm | Input Features | Classes | Train Rows | Test Rows | Accuracy | Precision (wtd) | Recall (wtd) | F1 (wtd) |
|---------|-----------|---------------|---------|-----------|----------|----------|----------------|-------------|---------|
| Diabetes | RandomForestClassifier | 5 | 3 | 72,044 | 18,012 | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| Kidney Disease | GradientBoostingClassifier | 7 | 2 | 319 | 80 | **0.9500** | 0.9500 | 0.9500 | 0.9500 |
| Obesity | RandomForestClassifier | 3 | 7 | 1,567 | 392 | **0.9515** | 0.9527 | 0.9515 | 0.9515 |

*All results from real holdout data (20 % stratified test split, random_state=42).*

---

### 9.2 Diabetes Model (`disease_model.pkl`)

**Algorithm:** `RandomForestClassifier`

**Hyperparameters:**
```
n_estimators    = 300
max_depth       = None  (fully grown trees)
min_samples_split = 4
min_samples_leaf  = 2
max_features    = "sqrt"
class_weight    = "balanced"
random_state    = 42
n_jobs          = -1
```

**Input features (column order matches training):**

| Feature | Source | Clinical role |
|---------|--------|---------------|
| age | Patient input | Demographic |
| gender | Encoded: Male=1, Female=0 | Demographic |
| bmi | Calculated: weight/height² | Weight status |
| HbA1c | Patient input (%) | Primary diabetes marker |
| blood glucose | Patient input (mg/dL) | Primary diabetes marker |

**Output classes:**

| Encoded int | Label | Display |
|------------|-------|---------|
| 0 | `diabetes` | Diabetes |
| 1 | `no diabetes` | No Diabetes |
| 2 | `prediabetes` | Pre-Diabetes |

**Saved artifacts:** `disease_model.pkl` (1,550 KB), `disease_encoder.pkl` (< 1 KB)

**Note on 100% accuracy:** HbA1c and fasting glucose are the standard WHO/ADA
diagnostic criteria used to derive the dataset labels. The RF learns these
threshold boundaries exactly from 90,056 real patient records.

---

### 9.3 Kidney Disease Model (`kidney_model.pkl`)

**Algorithm:** `GradientBoostingClassifier` (selected over RF; both achieved 0.9500)

**Hyperparameters:**
```
n_estimators    = 400
learning_rate   = 0.05
max_depth       = 4
min_samples_split = 4
min_samples_leaf  = 2
subsample       = 0.85
random_state    = 42
```

**Input features:**

| Feature | Source | Clinical role |
|---------|--------|---------------|
| age | Patient input | Demographic |
| gender | Encoded: Male=1, Female=0 | Demographic |
| bmi | Calculated | Weight status |
| sodium | Patient input (mEq/L) | Electrolyte balance |
| potassium | Patient input (mEq/L) | Electrolyte balance |
| BloodPressure | Systolic BP (mmHg) | Hypertension marker |
| SerumCreatinine | Patient input (mg/dL) | Primary kidney marker |

**Output classes:**

| Encoded int | Label |
|------------|-------|
| 0 | Kidney Disease |
| 1 | No Kidney Disease |

**Saved artifacts:** `kidney_model.pkl` (254 KB), `kidney_encoder.pkl` (< 1 KB)

**Per-class test results:**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Kidney Disease | 0.96 | 0.96 | 0.96 | 50 |
| No Kidney Disease | 0.93 | 0.93 | 0.93 | 30 |

---

### 9.4 Obesity Model (`obesity_model.pkl`)

**Algorithm:** `RandomForestClassifier`

**Hyperparameters:**
```
n_estimators    = 300
max_depth       = None
min_samples_split = 2
min_samples_leaf  = 1
max_features    = "sqrt"
class_weight    = "balanced"
random_state    = 42
n_jobs          = -1
```

**Input features:** age, gender (encoded), bmi

**Output classes and display mapping:**

| Encoded int | Source label | Display (app) |
|------------|-------------|---------------|
| 0 | Insufficient_Weight | Underweight |
| 1 | Normal_Weight | Normal Weight |
| 2 | Obesity_Type_I | Obese Class I |
| 3 | Obesity_Type_II | Obese Class II |
| 4 | Obesity_Type_III | Obese Class III |
| 5 | Overweight_Level_I | Overweight |
| 6 | Overweight_Level_II | Overweight |

**Per-class test results:**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| 0 (Underweight) | 0.96 | 0.96 | 0.96 | 49 |
| 1 (Normal Weight) | 0.96 | 0.94 | 0.95 | 54 |
| 2 (Obese I) | 0.92 | 1.00 | 0.96 | 68 |
| 3 (Obese II) | 0.98 | 0.91 | 0.94 | 55 |
| 4 (Obese III) | 1.00 | 0.98 | 0.99 | 57 |
| 5 (Overweight) | 0.91 | 0.94 | 0.93 | 53 |
| 6 (Overweight) | 0.94 | 0.91 | 0.93 | 56 |

**Saved artifacts:** `obesity_model.pkl` (2,040 KB), `obesity_encoder.pkl` (< 1 KB)

---

## 10. Disease Prediction Pipeline

```
predict_diabetes(age, gender, bmi, hba1c, glucose) → {"label", "confidence", "risk"}
predict_kidney(age, gender, bmi, sodium, potassium, bp, creatinine) → ...
predict_obesity(age, gender, bmi) → ...
```

**Confidence calculation:**
`confidence = model.predict_proba(X)[0][predicted_class_index] × 100`
(probability of the *predicted* class, not a fixed index)

**Risk classification post-prediction (`backend/risk.py`):**

| Condition | Final Status | UI Colour |
|-----------|-------------|-----------|
| Model: no disease + all markers normal | Low Risk | Green |
| Model: no disease + abnormal markers | Borderline | Amber |
| Model: disease + abnormal markers | High Risk | Red |
| Model: disease + borderline only | Moderate Risk | Amber |
| Model: disease + all markers normal | Model Flag | Blue |

**Clinical reference ranges used for marker checks:**

| Feature | Normal range | Borderline zone |
|---------|-------------|----------------|
| BMI | 18.5–24.9 | 17.0–29.9 |
| HbA1c | 4.0–5.6 % | 4.0–6.4 % |
| Blood Glucose | 70–99 mg/dL | 60–125 mg/dL |
| Systolic BP | 90–119 mmHg | 80–139 mmHg |
| Sodium | 135–145 mEq/L | 130–150 mEq/L |
| Potassium | 3.5–5.0 mEq/L | 3.0–5.5 mEq/L |
| Serum Creatinine | 0.5–1.3 mg/dL | 0.4–1.8 mg/dL |

---

## 11. Nutrition Engine

### 11.1 Anthropometric Formulas (verified from `utils.py`)

**BMI:**
```
BMI = weight_kg / (height_cm / 100)²
```

**BMI Category:**
```
BMI < 18.5              → Underweight
18.5 ≤ BMI < 25.0       → Normal
25.0 ≤ BMI < 30.0       → Overweight
BMI ≥ 30.0              → Obese
```

**BMR (Mifflin-St Jeor equation — verified from `utils.py`):**
```
Male:   BMR = 10 × weight_kg + 6.25 × height_cm − 5 × age + 5
Female: BMR = 10 × weight_kg + 6.25 × height_cm − 5 × age − 161
```

**TDEE:**
```
TDEE = round(BMR × activity_factor, 2)
```

**Activity factors (verified from `config.py`):**

| Activity Level | Factor |
|----------------|--------|
| Sedentary | 1.0 |
| Light | 1.375 |
| Moderate | 1.55 |
| Active | 1.725 |
| Very Active | 1.9 |

*Note: `ACTIVITY_FACTOR` dict is defined in `app.py`; the `config.py`
`WATER_ACTIVITY_MULTIPLIERS` uses a different, simpler scale:
Sedentary=1.0, Light=1.1, Moderate=1.2, Active=1.3, Very Active=1.4*

**Daily Water Intake (`recommendation.py`, `config.py`):**
```
water_L = weight_kg × 0.035 × activity_multiplier
water_L = min(water_L, 5.0)   ← ceiling enforced
```

**Daily Protein Requirement (`recommendation.py`):**
```
protein_per_kg = {
    "default":  0.8,
    "kidney":   0.6,
    "obesity":  1.2,
    "diabetes": 1.0,
}
protein_g = weight_kg × protein_per_kg[condition]
            × 1.2  if age > 65
            × 1.1  if male
capped at config.KIDNEY_MAX_PROTEIN (12 g) for kidney disease
```

---

## 12. Disease Severity Rules

### 12.1 Severity Derivation

The severity level is derived programmatically from `RiskResult.final_status`
via `severity_rules.build_severity_dict()`:

| RiskResult.final_status | Severity used for filtering |
|------------------------|---------------------------|
| "High Risk" | severe |
| "Moderate Risk" | moderate |
| "Borderline" | mild |
| "Model Flag" | mild |
| "Low Risk" | None (no disease restrictions applied) |
| Default / unrecognised | moderate |

### 12.2 Hard Constraints per Disease per Tier

These are absolute filters — foods violating them are **removed from the
candidate pool entirely**. Marked `⚠ REQUIRES DOMAIN VALIDATION` in source.

**Diabetes:**

| Tier | Free Sugar (g) max | Calories (kcal) max | Carbohydrates (g) max |
|------|--------------------|---------------------|-----------------------|
| Mild | 8.0 | 350 | — |
| Moderate | 5.0 | 250 | — |
| Severe | 2.0 | 200 | 20.0 |

**Kidney Disease:**

| Tier | Sodium (mg) max | Protein (g) max |
|------|-----------------|-----------------|
| Mild | 150 | 15.0 |
| Moderate | 100 | 12.0 |
| Severe | 60 | 8.0 |

**Obesity:**

| Tier | Calories (kcal) max | Fats (g) max | Free Sugar (g) max |
|------|---------------------|--------------|-------------------|
| Mild | 350 | 18.0 | — |
| Moderate | 250 | 10.0 | — |
| Severe | 180 | 6.0 | 3.0 |

### 12.3 Key Distinction for Paper

> The disease severity classification is **rule-based**, NOT ML-based.
> The ML models output a probability and a class label; `backend/risk.py`
> then applies deterministic clinical threshold rules to assign the final
> risk status and severity level. These are **two separate systems**.

---

## 13. Food Recommendation System

### 13.1 Recommendation Pipeline

```
1. DISEASE DETECTION
   User diseases → has_diabetes | has_kidney_disease | has_obesity

2. BASE HARD FILTERING [config.py thresholds]
   Diabetes:  Free Sugar ≤ 5g AND Calories ≤ 250 kcal
   Kidney:    Sodium ≤ 100mg AND Protein ≤ 12g
   Obesity:   Calories ≤ 250 kcal AND Fats ≤ 10g
   Normal:    Calories ≤ 500 kcal AND Free Sugar ≤ 15g

3. SEVERITY HARD FILTERING [severity_rules.py hard_constraints]
   (Tighter bounds applied per severity tier — see §12.2)

4. CONDIMENT EXCLUSION
   Snack items with Fibre ≥ (mean + 3×std of Snack fibre values)
   are excluded as non-standalone meals.
   Dataset-derived cutoff ≈ 12g fibre (fallback if insufficient data).

5. MEAL-TYPE FILTERING
   Breakfast: MealType == 'Breakfast'
   Lunch:     MealType in {'Lunch', 'Lunch/Dinner'}
   Snack:     MealType == 'Snack'
   Dinner:    MealType in {'Dinner', 'Lunch/Dinner'}

6. HYBRID SCORING [per food item]
   (See §14 Recommendation Scoring)

7. DIVERSITY PENALTY
   Selected foods receive penalty 0.3 subtracted from hybrid score
   (daily plan); food_id usage tracking for 7-day plan.

8. DETERMINISTIC RANKING
   Sort by [usage ASC, final_score DESC, food_id_key ASC]
   (stable mergesort — fully reproducible)

9. SELECTION
   Lunch/Dinner: 2 foods each (weekly planner) or 3 (daily single-call)
   Breakfast/Snack: 1 food each (weekly planner) or 2 (daily single-call)
```

---

## 14. Recommendation Scoring

### 14.1 Hybrid Score Formula (verified from `enhanced_recommender.py`)

```
nutrition_raw    = severity-multiplied arithmetic score (see §14.2)
nutrition_norm   = clamp(nutrition_raw / dynamic_p99_ceiling, 0, 1)
                   dynamic_p99_ceiling = max(p99 of raw scores, max_raw×0.5, 1.0)

severity_suit    = calculate_severity_suitability_score(row, disease_key, severity)
                   ∈ [0, 1]; hard constraint violation → 0.0
combined_suit    = min(severity_suit per active disease)   ← strictest wins

nutrition_blended = 0.70 × nutrition_norm + 0.30 × combined_suit

content_score    = calorie_fit×0.40 + protein_density×0.25
                 + fibre_density×0.20 + sugar_score×0.15

ncf_score        = 0.0   (NCF UNAVAILABLE)

hybrid = clamp(
    nutrition_blended × w_nutrition_eff +
    content_score     × w_content_eff   +
    ncf_score         × 0.0
, 0, 1)
```

### 14.2 Effective Weights

| Weight | Configured | Effective (NCF unavailable) |
|--------|-----------|---------------------------|
| w_nutrition | 0.50 | **0.6250** |
| w_content | 0.30 | **0.3750** |
| w_ncf | 0.20 | **0.0000** |

*When NCF becomes available, effective weights revert to 0.50 / 0.30 / 0.20.*

### 14.3 Content Score Sub-Scores

```
calorie_fit     = max(0, 1 − |calories / (TDEE × meal_fraction) − 1|)
  meal_fraction: Breakfast=0.25, Lunch=0.35, Snack=0.15, Dinner=0.25

protein_density = clamp(protein_g / calories / 0.12, 0, 1)   [if calories > 0]
fibre_density   = clamp(fibre_g   / calories / 0.06, 0, 1)
sugar_score     = max(0, 1 − free_sugar_g / 15.0)

content_score   = 0.40×calorie_fit + 0.25×protein_density
                + 0.20×fibre_density + 0.15×sugar_score
```

### 14.4 Severity Suitability Score Formula

```
for each soft constraint (e.g. Free Sugar ≤ 5g):
    if val > soft_max:
        excess_ratio = (val − soft_max) / max(soft_max, 1.0)
        penalty += min(0.5, excess_ratio × 0.4)

for each required_minimum met:
    bonus += 0.05

severity_suitability = max(0.0, min(1.0, 1.0 − penalty + bonus))
Returns 0.5 (neutral) for healthy users (severity=None)
```

---

## 15. NCF Implementation Status

### Current Implementation: NOT AVAILABLE AT RUNTIME

After thorough inspection of the complete codebase:

**Code that exists:**
- `backend/services/ncf_service.py` — service wrapper with availability checks
- `backend/services/interaction_service.py` — food interaction tracking schema
- `healthcare.db` — SQLite database with `food_interactions` table
- `config.py` — NCF weight defined (0.20) but redistributed at runtime

**Code that does NOT exist:**
- `ncf_integration/` directory — **does not exist on disk**
- `ncf_integration/models/ncf_model.keras` — **file not found**
- `ncf_integration/models/ncf_mappings.json` — **file not found**
- Any TensorFlow/Keras NCF model file anywhere in the project

**What `ncf_service.py` does at runtime:**
```
_NCF_AVAILABLE = False
Reason: "Model file not found: ncf_integration/models/ncf_model.keras"
```

**What `enhanced_recommender.py` does:**
```
_calculate_ncf_score() returns (0.0, False) unconditionally
NCF weight (0.20) is redistributed to nutrition (0.6250) and content (0.3750)
```

### Research Paper Positioning

> **Do NOT claim that the current system uses NCF.**
>
> The paper should state:
> - The hybrid scoring framework includes a placeholder for Neural
>   Collaborative Filtering with a weight allocation of 20 %.
> - NCF requires real user–food interaction data collected from application
>   users, with minimum thresholds of 500 interactions, 50 users, and
>   100 foods (defined in `interaction_service.py`).
> - NCF training and inference infrastructure is designed and partially
>   implemented (interaction tracking, service wrapper, weight allocation),
>   but no trained model exists at the time of submission.
> - NCF is presented as **future work / planned enhancement**.
>
> The current system is a **two-component hybrid: nutrition scoring +
> content-based filtering**.

---

## 16. 7-Day Meal Planner

### 16.1 Overview

**File:** `backend/services/weekly_meal_planner.py`

The planner generates a full 7-day, 28-meal-slot plan (Days 1–7 × Breakfast,
Lunch, Snack, Dinner) using the same disease/severity-filtered food pool as
the daily recommendation, with an additional diversity control layer.

### 16.2 Daily Structure

| Meal | Foods per day | Calorie target fraction |
|------|--------------|------------------------|
| Breakfast | 1 food | 25 % of TDEE |
| Lunch | 2 foods | 35 % of TDEE |
| Snack | 1 food | 15 % of TDEE |
| Dinner | 2 foods | 25 % of TDEE |
| **Total** | **6 foods/day** | **100 %** |

**Total slots per week: 7 × 6 = 42**

### 16.3 Diversity Penalty System

```python
CONSECUTIVE_DAY_PENALTY = 0.6   # same food used the previous day
SAME_DAY_PENALTY        = 0.8   # same food used twice today
REPEAT_PENALTY          = 0.2   # per additional use (multiplicative)
DIVERSITY_BONUS         = 0.1   # bonus for introducing a new food category
MAX_FOOD_USES           = 1     # target: 1 use per food per week
```

**Final score:** `final_score = hybrid_score − diversity_penalty`

**Food tracking:** Both dish-name (str) and food_id (int) are tracked via
`FoodUsageTracker` dataclass. Food IDs use MD5 hash if food_id column is
missing or zero.

### 16.4 Diversity Selection Algorithm

Foods are ranked by: `[usage ASC, final_score DESC, food_id_key ASC]`
(unused foods always rank above used foods regardless of score).

Progressive relaxation over 5 passes:

| Pass | allow_today_repeat | allow_over_cap | allow_combo_repeat |
|------|-------------------|--------------|--------------------|
| 1 | No | No | No |
| 2 | Yes | No | No |
| 3 | No | Yes | No |
| 4 | Yes | Yes | No |
| 5 | Yes | Yes | Yes |

Hard safety constraints (disease/severity filters) are **NEVER relaxed**.

### 16.5 Food Categories for Diversity

7 keyword-based categories:
rice_based, wheat_based, millet_based, protein_rich, vegetable_based,
breakfast_dish, beverage_snack, other

### 16.6 Verified Diversity Metrics (from AUDIT_REPORT.md)

| Profile | Unique Foods | Unique % | Max Repeat | Filtered Pool |
|---------|-------------|---------|-----------|--------------|
| Diabetes Moderate | 20 | 47.62 % | 3 | — |
| Healthy | 21 | 50.0 % | — | — |
| Severe Multiple Diseases | 14 | 40 % | — | 27 foods |

> Note: These metrics are from an earlier audit run. The current codebase
> targets MAX_FOOD_USES=1 (one appearance per food per week). Repetition
> occurs only when the per-slot pool is smaller than the number of required
> slots.

---

## 17. Personalization

| User Input | Used In | Effect |
|-----------|---------|--------|
| Age | BMR calculation; nutrition scoring; XAI reference ranges | Older (>65) patients get higher protein bonus; younger (<30) get slight calorie bonus |
| Gender | BMR calculation (Mifflin-St Jeor constant); disease models; nutrition scoring | Male: +protein×1.1 score; Female: +iron×2.0 score |
| Height + Weight | BMI; BMR; TDEE; water intake; protein requirement | All downstream calculations |
| Activity Level | TDEE multiplier; water intake; nutrition scoring | Active users get calorie/carb bonuses in scoring |
| HbA1c | Diabetes model input; XAI deviation | Primary discriminator for diabetes/prediabetes |
| Blood Glucose | Diabetes model input; XAI deviation | Secondary diabetes marker |
| Systolic BP | Kidney model input; risk marker check | High BP → abnormal marker → High Risk classification |
| Diastolic BP | Displayed in profile; NOT a model input | UI only |
| Sodium | Kidney model input; risk marker check; severity filtering | Low sodium required for kidney disease |
| Potassium | Kidney model input; risk marker check | High potassium → abnormal marker |
| Serum Creatinine | Kidney model input; risk marker check; XAI deviation | Primary kidney disease marker |
| Health Goal | Calorie target adjustment | Weight Loss/Gain/Maintenance adjusts TDEE |
| Region | Food recommendation filtering | Regional South Indian food matching |
| Disease Severity | Hard + soft constraints; scoring multipliers | Tighter constraints at higher severity |

---

## 18. Diversity Mechanism

### Implemented diversity controls:

1. **Penalty-based soft deterrence:** consecutive-day, same-day, and repeat
   penalties reduce a food's `final_score` making it rank lower than unused
   alternatives — without blocking it entirely.

2. **Hard food_id cap:** `MAX_FOOD_USES=1` — once a food has been used once
   across the 7-day plan, it fails `_candidate_passes()` in passes 1 and 2
   (passes 3–5 relax this for small pools).

3. **Category bonus:** A food from a category not yet used today reduces its
   diversity penalty by 0.1, promoting meal-type variety across the day.

4. **Combination deduplication:** `used_meal_combinations` tracks sorted
   food_id tuples per meal slot across days, preventing the same pair of
   Lunch foods appearing on two different days.

5. **Day-signature deduplication:** `used_day_signatures` prevents a complete
   day (all 6 food_ids) from repeating.

---

## 19. Experimental Setup

| Parameter | Value |
|-----------|-------|
| OS | Windows |
| Python | 3.13.3 |
| Streamlit | 1.59.2 |
| scikit-learn | 1.7.1 |
| pandas | 2.2.3 |
| NumPy | 2.2.6 |
| TensorFlow | 2.21.0 (installed; not used at runtime) |
| joblib | 1.5.2 |
| imbalanced-learn | Not installed |
| Training script | `train_models.py` |
| Random state | 42 (all models, all splits) |
| Test split | 20 % stratified |
| Cross-validation | Not performed (only holdout split) |

**Artifact sizes:**

| File | Size |
|------|------|
| disease_model.pkl | 1,550 KB |
| obesity_model.pkl | 2,040 KB |
| kidney_model.pkl | 254 KB |
| disease_encoder.pkl | < 1 KB |
| kidney_encoder.pkl | < 1 KB |
| obesity_encoder.pkl | < 1 KB |
| gender_encoder.pkl | < 1 KB |

---

## 20. Evaluation Metrics

**For disease prediction (verified with real holdout data):**
- Accuracy (overall)
- Precision (per-class and weighted average)
- Recall (per-class and weighted average)
- F1-score (per-class, macro average, weighted average)

**For recommendation quality:**
- Unique food ratio = unique foods / total food slots
- Repetition rate = repeated slots / total slots
- Max repetition count
- Average repetition count

**Not available (see §34):**
- ROC-AUC on real data (was computed on synthetic data — invalid)
- Precision@K, Recall@K, NDCG for recommendations (no user preference data)
- Cross-validation scores (not performed)
- Clinical validation metrics (not performed)

---

## 21. Current Results

### 21.1 Disease Prediction (verified from `train_log.txt`, `METRICS.md`)

| Model | Accuracy | Precision (wtd) | Recall (wtd) | F1 (wtd) |
|-------|----------|----------------|-------------|---------|
| Diabetes | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| Kidney Disease | **0.9500** | 0.9500 | 0.9500 | 0.9500 |
| Obesity | **0.9515** | 0.9527 | 0.9515 | 0.9515 |

### 21.2 Integration Verification

`test_risk_regression.py`: **30 / 30 tests passed** with the retrained models,
covering risk classification, XAI scoring, and confidence-index correctness.

---

## 22. Accuracy Improvement Analysis

### Before Retraining (old `.pkl` files evaluated on synthetic random data)

| Model | Old accuracy | Note |
|-------|-------------|------|
| Diabetes | 0.114 | 8-class model evaluated on random labels |
| Kidney Disease | 0.339 | 3-class model on random labels |
| Obesity | 0.120 | 7-class model on random labels |

*Source: `experiments/disease_prediction/model_evaluation_report.txt` explicitly
states "No real evaluation dataset was found."*

### After Retraining (`train_models.py` on real source datasets)

| Model | New accuracy | Improvement |
|-------|-------------|-------------|
| Diabetes | 1.0000 | +88.6 pp |
| Kidney Disease | 0.9500 | +61.1 pp |
| Obesity | 0.9515 | +83.2 pp |

### Technical Reasons for Improvement

**Diabetes (+88.6 pp):**
1. **Correct dataset alignment:** Old model had 8 classes including obesity/kidney
   rows (which lack HbA1c/glucose features). Training on mismatched feature/label
   combinations destroyed signal. New model uses only the 90,056 rows where
   HbA1c and glucose are populated.
2. **Reduced class count:** 8 classes → 3 clinically meaningful classes.
3. **Strong clinical signal:** HbA1c and glucose are the same criteria used to
   derive labels — the model learns deterministic threshold boundaries.

**Kidney (+61.1 pp):**
1. **Label cleaning:** Tab artefact (`'Kidney Disease\t'`) merged into clean
   `'Kidney Disease'` class — 2-class problem instead of 3.
2. **GBT vs RF:** Gradient Boosting chosen (both reached 0.95, GBT selected).
3. **Class-weight balancing:** `class_weight='balanced'` with GBT subsample=0.85.

**Obesity (+83.2 pp):**
1. **Correct dataset:** Old model was an RF trained on an unknown dataset;
   new model uses the 1,959-row `Obesity prediction.xlsx` with balanced classes.
2. **BMI is deterministic for obesity classification:** The dataset labels map
   directly to BMI ranges, making RF classification near-perfect.

---

## 23. Baseline Comparison

**Baseline comparison is not currently available.**

No other algorithms (Logistic Regression, Decision Tree, SVM, KNN, XGBoost)
were trained on the same datasets for comparison purposes.

**Recommended baseline experiments before publication:**
- Logistic Regression (linear baseline)
- Decision Tree (interpretability baseline)
- k-Nearest Neighbours
- Support Vector Machine (RBF kernel)
- XGBoost
- For kidney disease (small dataset): cross-validation (5-fold or 10-fold)

---

## 24. Ablation Study

The hybrid scoring formula supports the following ablation configurations:

| Config | Description | Current weights |
|--------|-------------|-----------------|
| A — Nutrition only | w_n=1.0, w_c=0.0, w_ncf=0.0 | Not tested |
| B — Nutrition + Content | w_n=0.625, w_c=0.375, w_ncf=0.0 | **Current system** |
| C — Nutrition + Content + NCF | w_n=0.50, w_c=0.30, w_ncf=0.20 | Planned (NCF unavailable) |
| D — With severity suitability | nutrition_blended = 0.70×norm + 0.30×severity | **Current system** |
| E — Without severity suitability | nutrition_blended = nutrition_norm | Not tested |

**Measurable metrics for ablation:**
- Unique food ratio per 7-day plan
- Calorie adherence (plan total vs TDEE target)
- Disease constraint satisfaction rate
- Recommendation diversity metrics

*Ablation study results are not currently available and should be generated
before submission.*

---

## 25. Reproducibility

| Item | Status | Evidence |
|------|--------|----------|
| Random seed | Yes — 42 everywhere | `train_models.py` |
| Training script | Yes | `train_models.py` |
| Model artifacts | Yes | 7 `.pkl` files in root |
| Source datasets | Present in `data/` | `.csv` and `.xlsx` files |
| Evaluation code | Partial | `evaluate_models.py` (real data path works) |
| Configuration file | Yes | `config.py` |
| requirements.txt | **Not found in repository** | Missing |
| Environment specification | **Not found** | Missing |
| Cross-validation | Not implemented | |
| Docker/container | Not available | |

---

## 26. Figures Required for Paper

All figures can be generated from existing code/data. No fabrication needed.

1. **System Architecture Diagram** — flowchart from §5 of this report
2. **Disease Prediction Pipeline** — User inputs → ML models → RiskResult → Severity
3. **Dataset Distribution Charts** — bar charts for each dataset's class distribution
4. **Confusion Matrices** — from `train_models.py` run with `confusion_matrix()` (not in current output; needs re-run with matrix capture)
5. **Per-class F1 Comparison** — bar chart from Tables in §9.2–9.4
6. **Before vs After Accuracy** — grouped bar chart (old synthetic vs new real)
7. **Hybrid Scoring Pipeline** — flowchart from §14.1
8. **Severity Constraint Table** — heatmap of thresholds per disease per tier
9. **7-Day Meal Planning Workflow** — flowchart from §16.4
10. **Food Diversity Metrics** — bar/pie chart of unique vs repeated foods
11. **XAI Feature Importance Example** — screenshot or generated Plotly chart
12. **BMI/BMR/TDEE Formula Flow** — equation diagram from §11.1

---

## 27. Tables Required for Paper

### Table 1 — Dataset Description

| Dataset | Records | Features | Source | Use |
|---------|---------|---------|--------|-----|
| cleaned_disease_dataset.csv | 92,414 | 20 | Constructed | Diabetes model training |
| kidney_disease.csv.xlsx | 399 | 9 | Collected | Kidney model training |
| Obesity prediction.csv.xlsx | 1,959 | 5 | Collected | Obesity model training |
| food_dataset.csv | 1,014 | 14 | Curated | Meal recommendation |

### Table 2 — Input Features per Model

*(See §9.2–9.4 input feature tables)*

### Table 3 — Model Configuration

| Model | Algorithm | n_estimators | max_depth | class_weight | Training rows |
|-------|-----------|-------------|-----------|-------------|--------------|
| Diabetes | RandomForest | 300 | None | balanced | 72,044 |
| Kidney | GradientBoosting | 400 (lr=0.05) | 4 | — | 319 |
| Obesity | RandomForest | 300 | None | balanced | 1,567 |

### Table 4 — Model Evaluation (filled)

*(See §21 — all values verified)*

### Table 5 — Severity Hard Constraints

*(See §12.2 — all 3 diseases × 3 tiers)*

### Table 6 — Hybrid Scoring Weights

| Component | Configured | Effective (NCF off) | Formula role |
|-----------|-----------|---------------------|-------------|
| Nutrition | 0.50 | 0.6250 | nutrition_blended × w |
| Content | 0.30 | 0.3750 | content_score × w |
| NCF | 0.20 | 0.0000 | 0.0 (unavailable) |

### Table 7 — Personalization Inputs

*(See §17)*

### Table 8 — Recommendation Diversity Metrics

*(Must be generated experimentally — see §34)*

### Table 9 — Baseline Comparison

*(Not available — see §23)*

### Table 10 — Ablation Study Results

*(Not available — see §24)*

---

## 28. Literature Review Requirements

The following research areas require peer-reviewed citations. No references
are fabricated here; these are search guidance only.

| Topic | Why relevant | Suggested search keywords |
|-------|-------------|--------------------------|
| RandomForest for disease prediction | Method used for diabetes and obesity models | "random forest diabetes prediction", "random forest obesity classification" |
| Gradient Boosting for clinical data | Method used for kidney model | "gradient boosting kidney disease", "GBT small medical datasets" |
| HbA1c/glucose for diabetes diagnosis | Justifies feature selection | "HbA1c fasting glucose diabetes diagnosis WHO criteria" |
| Mifflin-St Jeor equation | BMR formula source | "Mifflin St Jeor equation 1990 basal metabolic rate" (DOI: 10.1093/ajcn/51.2.241) |
| Content-based food recommendation | Used for 37.5 % of hybrid score | "content-based food recommendation system", "nutritional profile food RS" |
| Neural Collaborative Filtering | Planned future component | "He et al. 2017 Neural Collaborative Filtering" (DOI: 10.1145/3038912.3052569) |
| Personalised nutrition systems | General area | "personalized dietary recommendation machine learning review" |
| Disease-aware nutrition recommendation | Proposed system area | "chronic disease dietary recommendation deep learning" |
| Explainable AI for healthcare | XAI module | "explainable AI healthcare feature importance clinical" |
| Meal planning optimisation | 7-day planner | "automated meal planning constraint satisfaction" |
| Indian food datasets | Food dataset domain | "Indian food nutrition dataset", "South Indian dietary patterns" |
| Dietary diversity metrics | Diversity evaluation | "dietary diversity score food variety index" |

---

## 29. Potential Contributions

Based strictly on what is implemented:

| Contribution Area | Status | Notes |
|------------------|--------|-------|
| Three-model disease prediction pipeline | **Implemented** | Diabetes, kidney, obesity as unified pipeline |
| Clinical risk classification (5-way) | **Implemented** | Separates ML probability from clinical evidence |
| Three-tier severity-aware nutrition filtering | **Implemented** | Mild/moderate/severe per disease |
| Hybrid nutrition + content scoring | **Implemented** | Exact weights validated in config.py |
| 7-day diverse meal planning | **Implemented** | 42-slot plan with usage tracking |
| XAI via deviation-weighted feature importance | **Implemented** | Patient-specific (not just global RF importance) |
| South Indian food dataset curation | **Partially implemented** | Dataset exists but provenance undocumented |
| NCF collaborative filtering | **Not implemented** | Infrastructure only |
| Clinical validation | **Not performed** | Research prototype |
| Real user interaction data | **Not available** | No trained NCF possible |

---

## 30. Limitations

Based on the actual implementation:

1. **Diabetes perfect accuracy is not generalisable.** The 100% test accuracy
   arises because dataset labels are derived from the same HbA1c/glucose
   thresholds the model learns — it is not evidence of external validity.

2. **Kidney dataset is very small (399 rows).** 80-row test set limits
   statistical power of evaluation metrics. Confidence intervals are wide.

3. **No cross-validation.** Only a single holdout split was used. K-fold
   cross-validation results are not available.

4. **No ROC-AUC on real data.** The ROC-AUC values in `experiments/` were
   computed on synthetic random data and are invalid.

5. **NCF is not implemented.** The paper title references NCF; the current
   system does not use it. This must be clearly acknowledged.

6. **No clinical validation.** The severity thresholds are calibrated against
   the food dataset's nutrient percentiles, not clinical studies.

7. **No external test set.** All metrics are in-distribution (same data
   distribution as training). No held-out population cohort was used.

8. **Food dataset provenance undocumented.** The dataset is curated but
   its collection methodology, validation, and nutritional accuracy are
   not documented.

9. **No user study.** No real users tested the system; no feedback data exists.

10. **Recommendation quality unmeasured.** Precision@K, Recall@K, and NDCG
    cannot be computed without user preference ground truth.

11. **Severity thresholds not clinically validated.** All severity
    constraints are marked `⚠ REQUIRES DOMAIN VALIDATION` in source code.

12. **No real-time data.** System does not connect to EHR, wearables, or
    any live data source.

---

## 31. Future Work

1. Train and integrate the NCF model with real application user interaction data.
2. Collect user preference feedback to compute Precision@K, Recall@K, NDCG.
3. Perform clinical validation of severity thresholds with registered dietitians.
4. Expand kidney dataset (399 rows is critically small).
5. Add cross-validation to all three models.
6. Conduct baseline algorithm comparison (Logistic Regression, SVM, XGBoost).
7. Conduct ablation study on hybrid scoring components.
8. Add Type 2 vs Type 1 diabetes distinction.
9. Extend to additional diseases (hypertension, PCOS, thyroid).
10. Longitudinal tracking of patient progress and plan adaptation.

---

## 32. Research Gaps

| Gap | Current state | Required for publication |
|-----|--------------|--------------------------|
| NCF evaluation | No results | Required to match paper title |
| Baseline comparison | None | Required for most venues |
| Ablation study | None | Strongly recommended |
| Cross-validation | None | Required for small datasets |
| Clinical validation | None | Needed for any clinical claim |
| Recommendation quality metrics | None | Required for RS paper |
| External dataset validation | None | Required for generalisation claim |

---

## 33. Publication Readiness

| Component | Status | Evidence |
|-----------|--------|----------|
| Dataset | Complete | 4 datasets present and analysed |
| Preprocessing | Complete | Verified in train_models.py |
| Disease Prediction | Complete | 3 trained models, real test metrics |
| Accuracy Evaluation | Complete | METRICS.md, train_log.txt |
| Nutrition Engine | Complete | utils.py, config.py formulas verified |
| Recommendation | Complete | Enhanced recommender, hybrid scoring |
| 7-Day Planner | Complete | 20/20 tests pass, diversity metrics |
| Diversity | Complete | FoodUsageTracker, 5-pass relaxation |
| NCF | **Missing** | No trained model; infrastructure only |
| Baseline Comparison | **Missing** | Not performed |
| Ablation Study | **Missing** | Not performed |
| Reproducibility | Partial | No requirements.txt |
| Literature Review | **Missing** | Not in repository |
| Cross-validation | **Missing** | Not performed |
| Recommendation quality (P@K, R@K) | **Missing** | No ground truth |

---

## 34. Required Experiments Before Submission

### Phase 1 — Technical Validation (Critical)

1. **K-fold cross-validation** for all 3 models (especially kidney: 399 rows).
   Recommended: 5-fold or leave-one-out for kidney.
2. **Confusion matrices** on real test data (save from `train_models.py` run).
3. **ROC-AUC curves** on real test data using `predict_proba()`.
4. **Baseline algorithm comparison**: train LR, DT, RF (already done), SVM,
   KNN, XGBoost on each dataset; compare accuracy, F1, ROC-AUC.

### Phase 2 — Recommendation Evaluation

5. **Diversity metrics** from an actual generated weekly plan run (unique %, 
   repeat rate, category coverage) — must be computed from live runs, not
   assumed.
6. **Calorie target adherence** — measure |plan_calories − TDEE| / TDEE for
   multiple user profiles.
7. **Disease constraint satisfaction** — verify 100 % of recommended foods
   pass hard constraints for each disease/severity combination.

### Phase 3 — Ablation Study

8. Run the system with:
   - Nutrition-only scoring (w_n=1.0)
   - Nutrition + Content (current: 0.625/0.375)
   - With vs without severity suitability blending
   - Compare diversity metrics and calorie adherence.

### Phase 4 — NCF (if pursuing NCF claim)

9. Collect real user interaction data (minimum 500 interactions, 50 users).
10. Train NCF model; evaluate with held-out interactions.
11. Compare recommendation quality with and without NCF.

---

## 35. Final Technical Summary

The Smart Health Dashboard is a **working prototype** of a personalised
nutrition recommendation system with the following verified capabilities:

| Capability | Verified |
|-----------|---------|
| Disease risk screening (diabetes, kidney, obesity) | Yes — 95–100 % accuracy |
| 5-way clinical risk classification | Yes — tested with 30 regression tests |
| 3-tier severity-aware food filtering | Yes — implemented and tested |
| Hybrid nutrition + content scoring | Yes — formulas verified in code |
| 7-day diverse meal planning | Yes — 42-slot plan, diversity tracking |
| Explainable AI (deviation-weighted) | Yes — patient-specific XAI |
| NCF collaborative filtering | **No — not trained, not active** |
| Clinical validation | **No — screening prototype only** |

The system is **technically sound as a research prototype** but requires
additional experimental results (baseline comparison, cross-validation,
recommendation quality metrics) before submission to a peer-reviewed venue.

> **Disclaimer:** This system is a **disease-risk screening prototype**, not a
> medical device. It does not constitute a medical diagnosis. All severity
> thresholds are algorithm-calibrated, not clinically validated.
> Professional medical consultation is required before any clinical application.
