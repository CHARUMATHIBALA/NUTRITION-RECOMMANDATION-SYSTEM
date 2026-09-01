"""backend/xai.py — Explainable AI module for Smart Health Dashboard.

Method: RandomForest global feature_importances_ weighted by a per-patient
        clinical deviation score to produce patient-specific feature rankings.

Why this approach
-----------------
All three disease models are RandomForestClassifier objects, which expose
a ``feature_importances_`` attribute (mean decrease in impurity across all
trees).  This is the most compatible, zero-dependency method — no SHAP,
no extra installation required.

To make the importance *patient-specific* (not just global), each global
weight is multiplied by a normalised deviation score that measures how far
the patient's value is from the healthy reference range for that feature.
The result is a ranked list of features that both the *model* and the
*patient's individual values* make important.

Public API
----------
explain_diabetes(age, gender, bmi, hba1c, glucose)       -> ExplanationResult
explain_obesity(age, gender, bmi)                          -> ExplanationResult
explain_kidney(age, gender, bmi, sodium, potassium, bp, creatinine)
                                                           -> ExplanationResult

Each returns an ExplanationResult dataclass with:
    .feature_rows   : list[FeatureRow]   — ranked features with scores
    .summary        : str                — 1–2 sentence patient-friendly text
    .method         : str                — description of method used
    .available      : bool               — False if model unavailable
    .error          : str | None         — error message if unavailable
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import models  # project-local — loads all RandomForest models & encoders


# ════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ════════════════════════════════════════════════════════════════════

@dataclass
class FeatureRow:
    """One row in the feature-importance table."""
    feature:    str          # Human-readable feature name
    value:      str          # Formatted patient value with unit
    importance: float        # 0.0–1.0 combined score (global × local deviation)
    impact:     str          # "High" | "Medium" | "Low"
    direction:  str          # "↑ Above normal" | "↓ Below normal" | "✓ Normal range"
    raw_value:  float        # Numeric patient value (for sorting / rendering)


@dataclass
class ExplanationResult:
    """Complete XAI result for one disease."""
    feature_rows: list[FeatureRow] = field(default_factory=list)
    summary:      str  = ""
    method:       str  = "RandomForest feature_importances_ × clinical deviation"
    available:    bool = True
    error:        str | None = None


# ════════════════════════════════════════════════════════════════════
#  CLINICAL REFERENCE RANGES
#  Used to compute how far a patient's value deviates from normal.
#  Source: standard clinical reference ranges (display purposes only).
# ════════════════════════════════════════════════════════════════════
_RANGES: dict[str, tuple[float, float]] = {
    # feature_key       : (low_normal, high_normal)
    # Age: no clinically 'abnormal' age exists — using the full valid human
    # range so no patient is ever labelled "above/below normal" for their age.
    "age":              (0.0,  120.0),
    "bmi":              (18.5,  24.9),
    "hba1c":            (4.0,    5.6),
    "blood_glucose":    (70.0,  99.0),
    "systolic_bp":      (90.0, 119.0),
    "sodium":           (135.0, 145.0),
    "potassium":        (3.5,    5.0),
    # Creatinine: male 0.7–1.2, female 0.5–1.0 (Mayo Clinic).
    # Using the broadest unisex span (0.5–1.3) to avoid incorrectly flagging
    # low-normal females (0.5–0.6) as below-normal.
    "creatinine":       (0.5,    1.3),
    "gender":           (0.0,    1.0),   # categorical — always 0 deviation
}

# Human-readable display names for every feature key used in each model
_DISPLAY: dict[str, str] = {
    "age":           "Age",
    "gender":        "Gender",
    "bmi":           "BMI",
    "hba1c":         "HbA1c",
    "blood_glucose": "Blood Glucose",
    "systolic_bp":   "Systolic BP",
    "sodium":        "Sodium",
    "potassium":     "Potassium",
    "creatinine":    "Serum Creatinine",
}

# Units for display
_UNITS: dict[str, str] = {
    "age":           " yrs",
    "gender":        "",
    "bmi":           "",
    "hba1c":         "%",
    "blood_glucose": " mg/dL",
    "systolic_bp":   " mmHg",
    "sodium":        " mEq/L",
    "potassium":     " mEq/L",
    "creatinine":    " mg/dL",
}

# Feature key order for each model (must match training column order exactly)
_OBESITY_KEYS:  list[str] = ["age", "gender", "bmi"]
_DIABETES_KEYS: list[str] = ["age", "gender", "bmi", "hba1c", "blood_glucose"]
_KIDNEY_KEYS:   list[str] = ["age", "gender", "bmi", "sodium", "potassium",
                              "systolic_bp", "creatinine"]


# ════════════════════════════════════════════════════════════════════
#  INTERNAL HELPERS
# ════════════════════════════════════════════════════════════════════

def _deviation_score(key: str, value: float) -> float:
    """Return 0.0–1.0 normalised deviation of *value* from the healthy range.

    0.0 = value is inside the normal range
    1.0 = value is very far outside the normal range

    Gender is a binary categorical — returns 0.0 (no deviation concept).
    """
    if key == "gender":
        return 0.0
    lo, hi = _RANGES.get(key, (0.0, 1.0))
    mid    = (lo + hi) / 2.0
    span   = max((hi - lo) / 2.0, 1e-9)   # half-width of normal band

    if lo <= value <= hi:
        return 0.0                          # inside normal range

    # How many half-widths outside the band?
    dist = max(value - hi, lo - value)      # positive distance outside band
    # Sigmoid-like clamp: 0→0, 1 half-width→~0.5, 3→~0.95
    return float(min(1.0, dist / (span + dist)))


def _direction(key: str, value: float) -> str:
    """Return an arrow string indicating whether the value is above / below / in range."""
    if key == "gender":
        return "✓ Normal range"
    lo, hi = _RANGES.get(key, (0.0, 1.0))
    if value < lo:
        return "↓ Below normal"
    if value > hi:
        return "↑ Above normal"
    return "✓ Normal range"


def _format_value(key: str, raw: float, gender_str: str = "") -> str:
    """Format a numeric feature value for display."""
    if key == "gender":
        return gender_str or ("Male" if raw == 1 else "Female")
    unit = _UNITS.get(key, "")
    if key in ("age",):
        return f"{int(raw)}{unit}"
    return f"{raw:.1f}{unit}"


def _impact_label(score: float) -> str:
    """Map a patient-specific contribution score to a verbal impact label.

    score = global_importance × deviation_score
    When a value is within its clinical reference range, score = 0 → "Low".
    A score of 0 must produce "Low", not "High", so a clinically normal
    value is never displayed as a high-impact contributor.
    """
    if score >= 0.15:
        return "High"
    if score >= 0.06:
        return "Medium"
    return "Low"


def _build_rows(
    model,
    feature_keys: list[str],
    patient_values: dict[str, float],
    gender_str: str,
) -> list[FeatureRow]:
    """Build a ranked list of FeatureRow from model feature importances
    combined with patient-specific deviation scores.

    Key design principle
    --------------------
    A normal clinical value must produce a patient-specific contribution of 0.
    The old formula ``global_imp * (1 + dev_score)`` was wrong because when
    dev_score = 0 it still returned global_imp, making Serum Creatinine show
    "High Impact" for a completely normal value (e.g. 0.7 mg/dL).

    New formula:
        patient_score = global_imp * dev_score

    When dev_score = 0 → patient_score = 0 → impact = "Low" / bar = empty.
    The sort uses patient_score + global_imp/1000 as a tiebreaker so that
    features with the same deviation are shown in model-importance order.
    raw_value is always the actual numeric patient value for display.
    """
    if not hasattr(model, "feature_importances_"):
        return []

    importances = model.feature_importances_   # shape (n_features,)

    # Build one entry per feature
    entries: list[tuple[float, float, FeatureRow]] = []   # (sort_key, raw_val, row)

    for i, key in enumerate(feature_keys):
        global_imp    = float(importances[i])
        raw_val       = float(patient_values.get(key, 0.0))
        dev_score     = _deviation_score(key, raw_val)

        # Patient-specific contribution — 0 when value is within normal range
        patient_score = global_imp * dev_score

        # Tiebreaker: preserve model-importance order within same deviation tier
        sort_key = patient_score + (global_imp / 1000.0)

        row = FeatureRow(
            feature    = _DISPLAY.get(key, key),
            value      = _format_value(key, raw_val, gender_str),
            importance = round(patient_score, 6),
            impact     = _impact_label(patient_score),
            direction  = _direction(key, raw_val),
            raw_value  = raw_val,          # actual patient value for display
        )
        entries.append((sort_key, raw_val, row))

    # Sort descending by sort_key (patient contribution + tiebreaker)
    entries.sort(key=lambda t: t[0], reverse=True)
    return [e[2] for e in entries[:6]]


def _abnormal_features(rows: list[FeatureRow]) -> list[str]:
    """Return display names of features that are outside the normal range."""
    return [r.feature for r in rows if r.direction != "✓ Normal range"]


def _normal_features(rows: list[FeatureRow]) -> list[str]:
    """Return display names of features that are within normal range."""
    return [r.feature for r in rows if r.direction == "✓ Normal range"]


# ════════════════════════════════════════════════════════════════════
#  SUMMARY GENERATORS
#  Produce short, patient-friendly sentences.
#  These are generated from actual patient values, never hard-coded.
# ════════════════════════════════════════════════════════════════════

def _diabetes_summary(rows: list[FeatureRow], label: str) -> str:
    """Generate a 1–2 sentence patient-friendly summary for diabetes."""
    abnormal = _abnormal_features(rows)
    label_l  = label.lower()

    if not abnormal:
        return (
            f"The screening model predicted <strong>{label}</strong>. "
            "All measured values are currently within their normal reference ranges."
        )

    factors = ", ".join(f"<strong>{f}</strong>" for f in abnormal[:3])
    if "no diabetes" in label_l or "normal" in label_l:
        return (
            f"The model predicted <strong>{label}</strong>. "
            f"While {factors} show some deviation from the reference range, "
            "the overall pattern does not indicate a high diabetes risk at this time."
        )
    return (
        f"The model predicted <strong>{label}</strong>. "
        f"The contributing factors with the most influence are {factors}. "
        "This is a screening result only — please consult a healthcare professional."
    )


def _obesity_summary(rows: list[FeatureRow], label: str) -> str:
    """Generate a patient-friendly summary for obesity."""
    abnormal = _abnormal_features(rows)
    if not abnormal:
        return (
            f"The model classified weight status as <strong>{label}</strong>. "
            "BMI and age are within their reference ranges."
        )
    factors = ", ".join(f"<strong>{f}</strong>" for f in abnormal[:3])
    return (
        f"Weight status was classified as <strong>{label}</strong>. "
        f"The main contributing factor(s): {factors}."
    )


def _kidney_summary(rows: list[FeatureRow], label: str) -> str:
    """Generate a patient-friendly summary for kidney disease."""
    abnormal = _abnormal_features(rows)
    label_l  = label.lower()

    if not abnormal:
        return (
            f"The model predicted <strong>{label}</strong>. "
            "All kidney-related markers are within their normal reference ranges."
        )

    factors = ", ".join(f"<strong>{f}</strong>" for f in abnormal[:3])
    if "no kidney" in label_l:
        return (
            f"The model predicted <strong>{label}</strong>. "
            f"Note that {factors} show some deviation but the overall "
            "pattern does not suggest kidney disease at this time."
        )
    return (
        f"The model predicted <strong>{label}</strong>. "
        f"Key contributing factor(s): {factors}. "
        "This is a screening result — please consult a nephrologist for confirmation."
    )


# ════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ════════════════════════════════════════════════════════════════════

def explain_diabetes(
    age:     float,
    gender:  str,
    bmi:     float,
    hba1c:   float,
    glucose: float,
    label:   str = "",
) -> ExplanationResult:
    """Generate XAI explanation for the diabetes prediction.

    Parameters
    ----------
    age, gender, bmi, hba1c, glucose : patient values (same as predict_diabetes)
    label : the prediction label already returned by predict_diabetes

    Returns
    -------
    ExplanationResult
    """
    if models.disease_model is None:
        return ExplanationResult(
            available=False,
            error="disease_model.pkl not loaded — explanation unavailable.",
        )
    try:
        gender_enc = models.encode_gender(gender)
        patient = {
            "age":          float(age),
            "gender":       float(gender_enc),
            "bmi":          float(bmi),
            "hba1c":        float(hba1c),
            "blood_glucose":float(glucose),
        }
        rows    = _build_rows(models.disease_model, _DIABETES_KEYS, patient, str(gender))
        summary = _diabetes_summary(rows, label)
        return ExplanationResult(feature_rows=rows, summary=summary)
    except Exception as exc:
        return ExplanationResult(
            available=False,
            error=f"Explanation error: {exc}",
        )


def explain_obesity(
    age:    float,
    gender: str,
    bmi:    float,
    label:  str = "",
) -> ExplanationResult:
    """Generate XAI explanation for the obesity prediction."""
    if models.obesity_model is None:
        return ExplanationResult(
            available=False,
            error="obesity_model.pkl not loaded — explanation unavailable.",
        )
    try:
        gender_enc = models.encode_gender(gender)
        patient = {
            "age":    float(age),
            "gender": float(gender_enc),
            "bmi":    float(bmi),
        }
        rows    = _build_rows(models.obesity_model, _OBESITY_KEYS, patient, str(gender))
        summary = _obesity_summary(rows, label)
        return ExplanationResult(feature_rows=rows, summary=summary)
    except Exception as exc:
        return ExplanationResult(
            available=False,
            error=f"Explanation error: {exc}",
        )


def explain_kidney(
    age:        float,
    gender:     str,
    bmi:        float,
    sodium:     float,
    potassium:  float,
    bp:         float,
    creatinine: float,
    label:      str = "",
) -> ExplanationResult:
    """Generate XAI explanation for the kidney disease prediction."""
    if models.kidney_model is None:
        return ExplanationResult(
            available=False,
            error="kidney_model.pkl not loaded — explanation unavailable.",
        )
    try:
        gender_enc = models.encode_gender(gender)
        patient = {
            "age":        float(age),
            "gender":     float(gender_enc),
            "bmi":        float(bmi),
            "sodium":     float(sodium),
            "potassium":  float(potassium),
            "systolic_bp":float(bp),
            "creatinine": float(creatinine),
        }
        rows    = _build_rows(models.kidney_model, _KIDNEY_KEYS, patient, str(gender))
        summary = _kidney_summary(rows, label)
        return ExplanationResult(feature_rows=rows, summary=summary)
    except Exception as exc:
        return ExplanationResult(
            available=False,
            error=f"Explanation error: {exc}",
        )
