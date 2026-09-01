"""models.py — ML artifact loading for Smart Health Dashboard.

Loads three RandomForestClassifier models and their associated encoders.
All artifacts are loaded once at import time so prediction is fast.

Artifacts
---------
obesity_model   : 3 features  → 7 classes (0–6, mapped in predict.py)
disease_model   : 5 features  → 8 classes (diabetes, prediabetes, …)
kidney_model    : 7 features  → 3 classes (Kidney Disease / No Kidney Disease)
obesity_encoder : LabelEncoder  (integer classes 0–6, used for completeness)
disease_encoder : LabelEncoder  (string class names)
kidney_encoder  : LabelEncoder  (string class names, one has trailing \t)
gender_encoder  : LabelEncoder  (Female → 0, Male → 1)
"""

import os
import warnings
import joblib
import pandas as pd

warnings.filterwarnings("ignore")          # suppress sklearn version mismatch

# ── safe loader ──────────────────────────────────────────────────────
_MISSING: list[str] = []          # accumulate missing files; checked in predict.py

def _load(filename: str):
    """Load a joblib artifact safely.  Returns None and records the filename if
    the file is absent, so the application can surface a useful error instead
    of crashing at import time.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    if not os.path.exists(path):
        _MISSING.append(filename)
        return None
    try:
        return joblib.load(path)
    except Exception as exc:
        _MISSING.append(f"{filename} (load error: {exc})")
        return None


# ════════════════════════════════════════════════════════════════════
#  ML MODELS
# ════════════════════════════════════════════════════════════════════
obesity_model  = _load("obesity_model.pkl")   # RandomForest — 3 feat, 7 classes
disease_model  = _load("disease_model.pkl")   # RandomForest — 5 feat, 8 classes (diabetes)
kidney_model   = _load("kidney_model.pkl")    # RandomForest — 7 feat, 3 classes

# ════════════════════════════════════════════════════════════════════
#  ENCODERS
# ════════════════════════════════════════════════════════════════════
obesity_encoder = _load("obesity_encoder.pkl")   # LabelEncoder — integer classes 0–6
disease_encoder = _load("disease_encoder.pkl")   # LabelEncoder — string class names
kidney_encoder  = _load("kidney_encoder.pkl")    # LabelEncoder — string class names
gender_encoder  = _load("gender_encoder.pkl")    # LabelEncoder — Female→0, Male→1

# ════════════════════════════════════════════════════════════════════
#  FOOD DATASET
# ════════════════════════════════════════════════════════════════════
_food_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "food_dataset.csv")
try:
    food_df = pd.read_csv(_food_path)
except Exception:
    food_df = pd.DataFrame()


# ════════════════════════════════════════════════════════════════════
#  ENCODER HELPERS
# ════════════════════════════════════════════════════════════════════

def encode_gender(gender: str) -> int:
    """Encode 'Male'/'Female' to 1/0 using the fitted LabelEncoder.

    Falls back to a safe integer mapping if the encoder is unavailable
    or if an unexpected gender string is passed, so prediction never
    crashes due to a missing or corrupted encoder.
    """
    if gender_encoder is None:
        # Fallback: use the mapping that was recorded during training
        return 1 if str(gender).strip().lower() == "male" else 0
    try:
        return int(gender_encoder.transform([str(gender).strip()])[0])
    except ValueError:
        # Unknown label — default to the majority class (Male=1)
        return 1 if str(gender).strip().lower() == "male" else 0


# ── Diabetes label formatting ────────────────────────────────────────
# The disease_encoder returns lowercase snake_case strings.
# Map them to clean Title Case for display.
_DIABETES_DISPLAY = {
    "diabetes":       "Diabetes",
    "prediabetes":    "Pre-Diabetes",
    "no diabetes":    "No Diabetes",
    "normal":         "Normal",
    "obesity":        "Obesity",
    "overweight":     "Overweight",
    "underweight":    "Underweight",
    "kidney_disease": "Kidney Disease",
}

def decode_diabetes(prediction) -> str:
    """Decode a diabetes model prediction to a display-ready string.

    Parameters
    ----------
    prediction : array-like
        Raw output of ``disease_model.predict()``.

    Returns
    -------
    str
        Human-readable label (Title Case, no underscores).
    """
    if disease_encoder is None:
        return "Unknown (encoder missing)"
    try:
        raw: str = str(disease_encoder.inverse_transform(prediction)[0])
        return _DIABETES_DISPLAY.get(raw.lower().strip(), raw.replace("_", " ").title())
    except Exception:
        return "Unknown"


def decode_kidney(prediction) -> str:
    """Decode a kidney model prediction to a clean display-ready string.

    The kidney encoder contains a class with a trailing tab character
    (``'Kidney Disease\\t'``).  We normalise all outputs with ``strip()``
    so the tab never reaches the UI.

    Parameters
    ----------
    prediction : array-like
        Raw output of ``kidney_model.predict()``.

    Returns
    -------
    str
        Human-readable label with all leading/trailing whitespace removed.
    """
    if kidney_encoder is None:
        return "Unknown (encoder missing)"
    try:
        raw: str = str(kidney_encoder.inverse_transform(prediction)[0])
        return raw.strip()          # removes the known trailing \t on class 1
    except Exception:
        return "Unknown"


def decode_obesity(prediction) -> str:
    """Decode obesity prediction via the integer LabelEncoder.

    Note: predict.py uses its own ``obesity_label_map`` dict (which maps
    integer class indices to readable strings) instead of this function.
    This helper is kept for completeness and backward compatibility.
    """
    if obesity_encoder is None:
        return "Unknown (encoder missing)"
    try:
        return str(obesity_encoder.inverse_transform(prediction)[0])
    except Exception:
        return "Unknown"
