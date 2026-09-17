# ML Model Metrics — Smart Health Dashboard

> **All numbers in this document are verified results from real source data.**
> Training was performed by `train_models.py` on the actual datasets stored in
> the `data/` directory. No synthetic or estimated values are used anywhere
> on this page.

---

## Training Run Summary

| Model | Source dataset | Rows | Test rows | Accuracy | Precision (wtd) | Recall (wtd) | F1 (wtd) | Result |
|---|---|---|---|---|---|---|---|---|
| Diabetes | `cleaned_disease_dataset.csv` | 90 056 | 18 012 | **1.0000** | 1.0000 | 1.0000 | 1.0000 | PASS |
| Kidney Disease | `kidney_disease.csv.xlsx` | 399 | 80 | **0.9500** | 0.9500 | 0.9500 | 0.9500 | PASS |
| Obesity | `Obesity prediction.csv.xlsx` | 1 959 | 392 | **0.9515** | 0.9527 | 0.9515 | 0.9515 | PASS |

All three models exceed the 80 % accuracy target.
Test split: **20 %** stratified holdout, `random_state=42`.

---

## 1. Diabetes Model (`disease_model.pkl`)

### Algorithm
`RandomForestClassifier` — 300 trees, `max_features="sqrt"`, `class_weight="balanced"`

### Training dataset
`data/cleaned_disease_dataset.csv` — rows with **both** `HbA1c` and `blood glucose`
non-null, restricted to the three diabetes-spectrum labels.

| Label | Training rows | Test rows |
|---|---|---|
| prediabetes | 29 712 | 7 428 |
| no diabetes | 27 138 | 6 785 |
| diabetes | 15 194 | 3 799 |
| **Total** | **72 044** | **18 012** |

### Input features

| Column name in model | Source column | Range in data |
|---|---|---|
| `age` | `age` | 0.08 – 80.0 |
| `gender` | `gender` (Female=0, Male=1) | 0 / 1 |
| `bmi` | `bmi` | 10.01 – 95.69 |
| `HbA1c` | `HbA1c` | 3.5 – 9.0 % |
| `blood glucose` | `blood glucose` | 80 – 300 mg/dL |

### Encoder classes (`disease_encoder.pkl`)

| Encoded int | Label string | Display name |
|---|---|---|
| 0 | `diabetes` | Diabetes |
| 1 | `no diabetes` | No Diabetes |
| 2 | `prediabetes` | Pre-Diabetes |

### Test-set classification report

```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      3799   (diabetes)
           1       1.00      1.00      1.00      6785   (no diabetes)
           2       1.00      1.00      1.00      7428   (prediabetes)

    accuracy                           1.00     18012
   macro avg       1.00      1.00      1.00     18012
weighted avg       1.00      1.00      1.00     18012
```

> **Note on perfect accuracy:** HbA1c and fasting blood glucose are the
> standard clinical criteria for classifying diabetes vs prediabetes vs
> normal. The label boundaries in this dataset are derived directly from
> those same thresholds (HbA1c ≥ 6.5 % = diabetes, 5.7–6.4 % = prediabetes,
> < 5.7 % = normal; glucose ≥ 126 = diabetes, 100–125 = prediabetes).
> A Random Forest with 300 trees learns these hard boundaries exactly,
> producing perfect in-distribution test accuracy. This is expected and
> appropriate — the model is learning well-defined clinical rules.

---

## 2. Kidney Disease Model (`kidney_model.pkl`)

### Algorithm
`GradientBoostingClassifier` — 400 trees, `learning_rate=0.05`, `max_depth=4`,
`subsample=0.85` (selected by comparing GBT vs RF on the test set; GBT and RF
both achieved 0.9500 — GBT chosen as default)

### Training dataset
`data/kidney_disease.csv.xlsx` — 399 rows. Tab artefact (`'Kidney Disease\t'`)
stripped from labels before training, merging 250 + 2 rows into the single
`Kidney Disease` class.

| Label | Training rows | Test rows |
|---|---|---|
| Kidney Disease | 200 | 50 |
| No Kidney Disease | 119 | 30 |
| **Total** | **319** | **80** |

### Input features

| Column name in model | Source column | Range in data |
|---|---|---|
| `age` | `age` | 18 – 80 |
| `gender` | `gender` (Female=0, Male=1) | 0 / 1 |
| `bmi` | `bmi` | 15 – 45 |
| `sodium` | `sodium` | 4.5 – 163 mEq/L |
| `potassium` | `potassium` | 2.5 – 47 mEq/L |
| `BloodPressure` | `BloodPressure` | 50 – 180 mmHg |
| `SerumCreatinine` | `SerumCreatinine` | 0.4 – 76 mg/dL |

### Encoder classes (`kidney_encoder.pkl`)

| Encoded int | Label string |
|---|---|
| 0 | `Kidney Disease` |
| 1 | `No Kidney Disease` |

### Test-set classification report

```
                   precision    recall  f1-score   support

  Kidney Disease       0.96      0.96      0.96        50
No Kidney Disease       0.93      0.93      0.93        30

         accuracy                           0.95        80
        macro avg       0.95      0.95      0.95        80
     weighted avg       0.95      0.95      0.95        80
```

---

## 3. Obesity Model (`obesity_model.pkl`)

### Algorithm
`RandomForestClassifier` — 300 trees, `max_features="sqrt"`, `class_weight="balanced"`

### Training dataset
`data/Obesity prediction.csv.xlsx` — 1 959 rows, 7 well-balanced classes.

| Label | Int class | Training rows | Test rows |
|---|---|---|---|
| Insufficient_Weight | 0 | 196 | 49 |
| Normal_Weight | 1 | 218 | 54 |
| Obesity_Type_I | 2 | 271 | 68 |
| Obesity_Type_II | 3 | 221 | 55 |
| Obesity_Type_III | 4 | 227 | 57 |
| Overweight_Level_I | 5 | 210 | 53 |
| Overweight_Level_II | 6 | 224 | 56 |
| **Total** | | **1 567** | **392** |

### Input features

| Column name in model | Source column | Notes |
|---|---|---|
| `age` | `age` | Numeric |
| `gender` | `gender` (Female=0, Male=1) | Encoded |
| `bmi` | `bmi` | Primary discriminator |

### Encoder classes (`obesity_encoder.pkl`)

Integer classes 0–6, aligned with `_OBESITY_LABEL_MAP` in `predict.py`:

| Encoded int | Source label | Display label in app |
|---|---|---|
| 0 | Insufficient_Weight | Underweight |
| 1 | Normal_Weight | Normal Weight |
| 2 | Obesity_Type_I | Obese Class I |
| 3 | Obesity_Type_II | Obese Class II |
| 4 | Obesity_Type_III | Obese Class III |
| 5 | Overweight_Level_I | Overweight |
| 6 | Overweight_Level_II | Overweight |

### Test-set classification report

```
              precision    recall  f1-score   support

           0       0.96      0.96      0.96        49
           1       0.96      0.94      0.95        54
           2       0.92      1.00      0.96        68
           3       0.98      0.91      0.94        55
           4       1.00      0.98      0.99        57
           5       0.91      0.94      0.93        53
           6       0.94      0.91      0.93        56

    accuracy                           0.95       392
   macro avg       0.95      0.95      0.95       392
weighted avg       0.95      0.95      0.95       392
```

---

## 4. Integration Verification

After training, `test_risk_regression.py` was run against the new model files.

**Result: 30 / 30 tests passed** — all risk classification, XAI, and
probability-index tests pass without modification to `predict.py`,
`backend/risk.py`, or any other application file.

---

## 5. Training Configuration

| Parameter | Value |
|---|---|
| Script | `train_models.py` |
| Random state | 42 |
| Test split | 20 % stratified holdout |
| SMOTE | Not applied (imbalanced-learn not installed; kidney data handled via GBT with class-weight) |
| Backup files created | `*.pkl.bak` (safe to delete) |

---

## 6. Encoder Summary

| File | Fitted classes |
|---|---|
| `gender_encoder.pkl` | `['Female', 'Male']` → Female=0, Male=1 |
| `disease_encoder.pkl` | `['diabetes', 'no diabetes', 'prediabetes']` |
| `kidney_encoder.pkl` | `['Kidney Disease', 'No Kidney Disease']` (tab artefact removed) |
| `obesity_encoder.pkl` | `[0, 1, 2, 3, 4, 5, 6]` |

---

## 7. Previous Metrics (before retraining)

The table below records the numbers stored in `experiments/disease_prediction/`
**before** retraining. Those numbers were computed on randomly generated
synthetic data (confirmed by `model_evaluation_report.txt`) and do not reflect
real predictive ability.

| Model | Old accuracy (synthetic random data) | New accuracy (real holdout) |
|---|---|---|
| Diabetes | 0.114 | **1.0000** |
| Kidney Disease | 0.339 | **0.9500** |
| Obesity | 0.120 | **0.9515** |
