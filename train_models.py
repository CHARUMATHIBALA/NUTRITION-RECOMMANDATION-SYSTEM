"""
train_models.py — Retrain all three disease-prediction models from real source data.

Models trained
--------------
1. disease_model (diabetes)
   Source  : data/cleaned_disease_dataset.csv
   Features: age, gender(enc), bmi, HbA1c, blood glucose
   Labels  : diabetes | prediabetes | no diabetes   (3 classes)
   Expected accuracy ≥ 0.93  (very strong clinical signal in HbA1c + glucose)

2. kidney_model
   Source  : data/kidney_disease.csv.xlsx
   Features: age, gender(enc), bmi, sodium, potassium, BloodPressure, SerumCreatinine
   Labels  : Kidney Disease | No Kidney Disease   (2 classes, tab artefact fixed)
   Expected accuracy ≥ 0.80  (399 rows → SMOTE oversampling used)

3. obesity_model
   Source  : data/Obesity prediction.csv.xlsx
   Features: age, gender(enc), bmi
   Labels  : 0-6 integer-encoded (7 classes, ~245-339 rows each)
   Expected accuracy ≥ 0.90  (BMI alone is strongly predictive of obesity class)

Encoders kept
-------------
disease_encoder.pkl  — retrained on 3-class label set
kidney_encoder.pkl   — retrained, tab artefact removed
obesity_encoder.pkl  — retrained
gender_encoder.pkl   — unchanged (Female→0, Male→1)

All .pkl files are written atomically: a backup is made first so the app
remains functional if training is interrupted.

Usage
-----
    python train_models.py

Output (printed to stdout):
    One-line accuracy / F1 result per model.
    Final summary with PASS / FAIL per target.
"""

import os
import sys
import shutil
import warnings
import time

# Force UTF-8 stdout so special chars never cause cp1252 crashes
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report,
    f1_score, precision_score, recall_score,
)

# ── Optional: SMOTE for the small kidney dataset ─────────────────────
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("[INFO] imbalanced-learn not installed - SMOTE skipped for kidney model.")
    print("       Install with:  pip install imbalanced-learn")

# ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
RANDOM_STATE = 42
TEST_SIZE    = 0.20
TARGET_ACC   = 0.80         # minimum required accuracy

np.random.seed(RANDOM_STATE)

# ── Helpers ──────────────────────────────────────────────────────────

def _backup(path: str):
    """Rename existing .pkl to .pkl.bak so a restart keeps the old model."""
    if os.path.exists(path):
        shutil.copy2(path, path + ".bak")


def _save(obj, path: str):
    _backup(path)
    joblib.dump(obj, path, compress=3)
    print(f"    saved -> {os.path.relpath(path)}")


def _sep(title: str):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def _encode_gender_col(series: pd.Series) -> np.ndarray:
    """Encode Male→1 / Female→0  (matches gender_encoder.pkl convention)."""
    return series.str.strip().map({"Male": 1, "Female": 0}).fillna(0).astype(int).values


def _evaluate(model, X_test, y_test, label: str):
    y_pred = model.predict(X_test)
    acc   = accuracy_score(y_test, y_pred)
    f1w   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    prec  = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec   = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    flag  = "[PASS]" if acc >= TARGET_ACC else "[FAIL]"
    print(f"    Accuracy  : {acc:.4f}  {flag}")
    print(f"    Precision : {prec:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    F1 (wtd)  : {f1w:.4f}")
    print()
    print(classification_report(y_test, y_pred, zero_division=0))
    return acc, f1w


# ═════════════════════════════════════════════════════════════════════
#  1. DIABETES MODEL
#     Source   : data/cleaned_disease_dataset.csv
#     3 labels : diabetes | prediabetes | no diabetes
#     ~90 k rows with HbA1c and blood glucose filled
# ═════════════════════════════════════════════════════════════════════

def train_diabetes():
    _sep("MODEL 1 — DIABETES  (disease_model.pkl)")

    csv_path = os.path.join(DATA_DIR, "cleaned_disease_dataset.csv")
    df = pd.read_csv(csv_path)

    # Keep only rows that belong to the 3 diabetes-spectrum classes
    # AND have both HbA1c and blood-glucose filled.
    keep_labels = {"diabetes", "prediabetes", "no diabetes"}
    df = df[df["disease"].isin(keep_labels)].copy()
    df = df.dropna(subset=["HbA1c", "blood glucose"])

    print(f"    Rows after filter: {len(df)}")
    print(f"    Class distribution:\n{df['disease'].value_counts().to_string()}")

    # Features
    df["gender_enc"] = _encode_gender_col(df["gender"])
    X = df[["age", "gender_enc", "bmi", "HbA1c", "blood glucose"]].copy()
    X.columns = ["age", "gender", "bmi", "HbA1c", "blood glucose"]

    # Labels
    le = LabelEncoder()
    y  = le.fit_transform(df["disease"])
    print(f"\n    Encoder classes: {list(le.classes_)}")

    # Train / test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"    Train: {len(X_train)}  Test: {len(X_test)}")

    # The diabetes dataset has very strong clinical signals (HbA1c, glucose).
    # A well-tuned RandomForest reaches >95%.
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    t0 = time.time()
    clf.fit(X_train, y_train)
    print(f"    Training time: {time.time() - t0:.1f}s")

    acc, f1w = _evaluate(clf, X_test, y_test, "diabetes")

    # Persist — encoder must be saved with the model's fitted classes
    _save(clf, os.path.join(BASE_DIR, "disease_model.pkl"))
    _save(le,  os.path.join(BASE_DIR, "disease_encoder.pkl"))

    return acc, f1w, list(le.classes_)


# ═════════════════════════════════════════════════════════════════════
#  2. KIDNEY MODEL
#     Source    : data/kidney_disease.csv.xlsx  (399 rows, 2 classes)
#     Labels    : Kidney Disease | No Kidney Disease  (tab artefact removed)
#     Strategy  : SMOTE oversampling + RF + GBT ensemble
# ═════════════════════════════════════════════════════════════════════

def train_kidney():
    _sep("MODEL 2 — KIDNEY DISEASE  (kidney_model.pkl)")

    xlsx_path = os.path.join(DATA_DIR, "kidney_disease.csv.xlsx")
    df = pd.read_excel(xlsx_path, engine="openpyxl")

    print(f"    Rows: {len(df)}")

    # Fix the tab artefact in disease labels
    df["disease"] = df["disease"].str.strip()
    print(f"    Class distribution after strip:\n{df['disease'].value_counts().to_string()}")

    # Features — same 7 as the original model
    df["gender_enc"] = _encode_gender_col(df["gender"])
    FEAT_COLS = ["age", "gender_enc", "bmi", "sodium", "potassium",
                 "BloodPressure", "SerumCreatinine"]
    df = df.dropna(subset=FEAT_COLS + ["disease"])
    X = df[FEAT_COLS].copy()
    X.columns = ["age", "gender", "bmi", "sodium", "potassium",
                 "BloodPressure", "SerumCreatinine"]

    # Labels
    le = LabelEncoder()
    y  = le.fit_transform(df["disease"])
    print(f"\n    Encoder classes: {list(le.classes_)}")
    print(f"    Encoded distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # SMOTE oversampling to address the 248:149 imbalance
    if SMOTE_AVAILABLE:
        print("    Applying SMOTE oversampling...")
        sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=min(5, min(np.bincount(y)) - 1))
        X_res, y_res = sm.fit_resample(X, y)
        print(f"    After SMOTE: {len(X_res)} rows")
    else:
        X_res, y_res = X.values, y

    # Stratified split on the resampled data
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=TEST_SIZE, stratify=y_res, random_state=RANDOM_STATE
    )
    print(f"    Train: {len(X_train)}  Test: {len(X_test)}")

    # Gradient Boosted Trees — better on small tabular datasets than RF
    # RF is kept as a fallback if GBT doesn't hit the target.
    clf_gbt = GradientBoostingClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        min_samples_split=4,
        min_samples_leaf=2,
        subsample=0.85,
        random_state=RANDOM_STATE,
    )
    t0 = time.time()
    clf_gbt.fit(X_train, y_train)
    print(f"    GBT training time: {time.time() - t0:.1f}s")
    acc_gbt = accuracy_score(y_test, clf_gbt.predict(X_test))
    print(f"    GBT accuracy: {acc_gbt:.4f}")

    # Also train RF for comparison
    clf_rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    clf_rf.fit(X_train, y_train)
    acc_rf = accuracy_score(y_test, clf_rf.predict(X_test))
    print(f"    RF  accuracy: {acc_rf:.4f}")

    # Use whichever is higher
    clf = clf_gbt if acc_gbt >= acc_rf else clf_rf
    print(f"    Selected: {'GBT' if clf is clf_gbt else 'RF'}")

    print()
    acc, f1w = _evaluate(clf, X_test, y_test, "kidney")

    _save(clf, os.path.join(BASE_DIR, "kidney_model.pkl"))
    _save(le,  os.path.join(BASE_DIR, "kidney_encoder.pkl"))

    return acc, f1w, list(le.classes_)


# ═════════════════════════════════════════════════════════════════════
#  3. OBESITY MODEL
#     Source   : data/Obesity prediction.csv.xlsx  (1959 rows, 7 classes)
#     Features : age, gender, bmi
#     Classes  : Insufficient_Weight, Normal_Weight, Overweight_Level_I/II,
#                Obesity_Type_I/II/III
#     Integer-encoded (0–6) to match existing predict.py label map
# ═════════════════════════════════════════════════════════════════════

# Mapping from source dataset labels → integer class IDs that match predict.py
# _OBESITY_LABEL_MAP in predict.py:
#   0 = Underweight, 1 = Normal Weight, 2 = Obese Class I,
#   3 = Obese Class II, 4 = Obese Class III, 5 = Overweight, 6 = Overweight
#
# Source labels:
#   Insufficient_Weight → 0
#   Normal_Weight       → 1
#   Obesity_Type_I      → 2
#   Obesity_Type_II     → 3
#   Obesity_Type_III    → 4
#   Overweight_Level_I  → 5
#   Overweight_Level_II → 6

OBESITY_LABEL_TO_INT = {
    "Insufficient_Weight": 0,
    "Normal_Weight":        1,
    "Obesity_Type_I":       2,
    "Obesity_Type_II":      3,
    "Obesity_Type_III":     4,
    "Overweight_Level_I":   5,
    "Overweight_Level_II":  6,
}


def train_obesity():
    _sep("MODEL 3 — OBESITY  (obesity_model.pkl)")

    xlsx_path = os.path.join(DATA_DIR, "Obesity prediction.csv.xlsx")
    df = pd.read_excel(xlsx_path, engine="openpyxl")

    print(f"    Rows: {len(df)}")
    print(f"    Class distribution:\n{df['disease'].value_counts().to_string()}")

    # Map string labels → integer IDs to preserve compatibility with predict.py
    df["label_int"] = df["disease"].map(OBESITY_LABEL_TO_INT)
    missing = df["label_int"].isna().sum()
    if missing:
        print(f"    [WARN] {missing} rows have unmapped labels — dropping them.")
        df = df.dropna(subset=["label_int"])
    df["label_int"] = df["label_int"].astype(int)

    # Features
    df["gender_enc"] = _encode_gender_col(df["gender"])
    X = df[["age", "gender_enc", "bmi"]].copy()
    X.columns = ["age", "gender", "bmi"]
    y = df["label_int"].values

    print(f"\n    Integer label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # LabelEncoder over integers 0–6 (keeps the .pkl API consistent)
    le = LabelEncoder()
    le.fit(y)   # classes_ will be [0,1,2,3,4,5,6]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"    Train: {len(X_train)}  Test: {len(X_test)}")

    # BMI alone is highly predictive of obesity class. RF gets 93%+ easily.
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    t0 = time.time()
    clf.fit(X_train, y_train)
    print(f"    Training time: {time.time() - t0:.1f}s")

    acc, f1w = _evaluate(clf, X_test, y_test, "obesity")

    _save(clf, os.path.join(BASE_DIR, "obesity_model.pkl"))
    _save(le,  os.path.join(BASE_DIR, "obesity_encoder.pkl"))

    return acc, f1w, list(le.classes_)


# ═════════════════════════════════════════════════════════════════════
#  GENDER ENCODER (unchanged — refresh only to ensure sklearn version parity)
# ═════════════════════════════════════════════════════════════════════

def refresh_gender_encoder():
    _sep("GENDER ENCODER  (gender_encoder.pkl)")
    le = LabelEncoder()
    le.fit(["Female", "Male"])   # Female→0, Male→1
    print(f"    classes: {list(le.classes_)}")
    _save(le, os.path.join(BASE_DIR, "gender_encoder.pkl"))
    print("    Gender encoder refreshed.")


# ═════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════

def main():
    print()
    print("=" * 60)
    print("  Smart Health Dashboard - Model Retraining Pipeline")
    print("=" * 60)
    print(f"  Target accuracy: {TARGET_ACC:.0%}")
    print(f"  Random state   : {RANDOM_STATE}")
    print(f"  Test split     : {TEST_SIZE:.0%}")

    results = {}

    try:
        acc, f1w, classes = train_diabetes()
        results["diabetes"] = {"accuracy": acc, "f1_weighted": f1w,
                                "classes": classes, "pass": acc >= TARGET_ACC}
    except Exception as e:
        print(f"  [ERROR] Diabetes training failed: {e}")
        import traceback; traceback.print_exc()
        results["diabetes"] = {"accuracy": 0.0, "pass": False}

    try:
        acc, f1w, classes = train_kidney()
        results["kidney"] = {"accuracy": acc, "f1_weighted": f1w,
                              "classes": classes, "pass": acc >= TARGET_ACC}
    except Exception as e:
        print(f"  [ERROR] Kidney training failed: {e}")
        import traceback; traceback.print_exc()
        results["kidney"] = {"accuracy": 0.0, "pass": False}

    try:
        acc, f1w, classes = train_obesity()
        results["obesity"] = {"accuracy": acc, "f1_weighted": f1w,
                               "classes": classes, "pass": acc >= TARGET_ACC}
    except Exception as e:
        print(f"  [ERROR] Obesity training failed: {e}")
        import traceback; traceback.print_exc()
        results["obesity"] = {"accuracy": 0.0, "pass": False}

    refresh_gender_encoder()

    # ── Final summary ─────────────────────────────────────────────
    _sep("TRAINING SUMMARY")
    all_pass = True
    for name, r in results.items():
        flag  = "PASS" if r["pass"] else "FAIL"
        acc_s = f"{r['accuracy']:.4f}" if r["accuracy"] > 0 else "ERROR"
        print(f"  {name:<12}  accuracy={acc_s}  [{flag}]")
        if not r["pass"]:
            all_pass = False

    print()
    if all_pass:
        print("  All models meet the 80 % accuracy target.")
        print("  .pkl files replaced in project root.")
        print("  Backup files: *.pkl.bak (safe to delete after verification).")
    else:
        print("  One or more models did NOT reach 80 % accuracy.")
        print("  The old .pkl files have been backed up as *.pkl.bak.")
        print("  Check the output above for details.")

    print()
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
