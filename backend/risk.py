"""backend/risk.py — Single source of truth for risk classification.

Design principle
----------------
ML model probability  ≠  clinical risk.

A model can output a high probability (e.g. 77 %) for "Kidney Disease"
while every measured clinical value is within its normal reference range.
Collapsing probability directly into "High Risk" is misleading.

This module separates three concepts:

1. MODEL SIGNAL  — what the RandomForest predicted and how confident it is.
2. CLINICAL STATUS — are the patient's measured values within reference ranges?
3. FINAL STATUS  — the combination of both, used by every UI component.

Final status values
-------------------
"High Risk"              — model predicts disease  AND  ≥1 clinical marker abnormal
"Moderate Risk"          — model predicts disease  AND  borderline clinical values
"Model Flag"             — model predicts disease  BUT  ALL markers are within range
"Low Risk"               — model predicts no disease  AND  all markers normal
"Borderline"             — model predicts no disease  BUT  ≥1 marker is borderline

The UI must render the final_status string, not recalculate risk itself.

Public API
----------
classify_risk(disease_key, model_predicts_disease, model_probability,
              clinical_values) -> RiskResult

disease_key : "diabetes" | "obesity" | "kidney"

RiskResult fields
-----------------
.final_status   str   — one of the values above, used by prediction card
.card_level     str   — "high" | "medium" | "low" — controls card colour stripe
.card_risk_text str   — short label shown inside the badge
.banner_level   str   — "danger" | "warning" | "info" | "ok"
.banner_title   str   — bold title inside the alert banner
.banner_body    str   — body text (may contain <strong> HTML)
.model_probability float — 0–100 %, plain Python float
.abnormal_markers list[str] — feature names outside their reference range
.borderline_markers list[str] — feature names in borderline zone
.all_markers_normal bool
.clinical_status_text str  — short plain-text clinical summary
"""

from __future__ import annotations
from dataclasses import dataclass, field

# ════════════════════════════════════════════════════════════════════
#  CLINICAL REFERENCE RANGES  (single definition, shared with xai.py)
#
#  Each entry: (lo_normal, hi_normal, lo_borderline, hi_borderline)
#  Borderline zone = values just outside normal but not yet clearly abnormal.
#  If lo_borderline == lo_normal and hi_borderline == hi_normal, no borderline.
# ════════════════════════════════════════════════════════════════════
CLINICAL_RANGES: dict[str, tuple] = {
    # key:              (lo_norm, hi_norm, lo_border, hi_border,  display_name,   unit)
    "bmi":              (18.5,    24.9,    17.0,      29.9,       "BMI",          ""),
    "hba1c":            (4.0,     5.6,     4.0,       6.4,        "HbA1c",        "%"),
    "blood_glucose":    (70.0,    99.0,    60.0,      125.0,      "Blood Glucose","mg/dL"),
    "systolic_bp":      (90.0,    119.0,   80.0,      139.0,      "Systolic BP",  "mmHg"),
    "sodium":           (135.0,   145.0,   130.0,     150.0,      "Sodium",       "mEq/L"),
    "potassium":        (3.5,     5.0,     3.0,       5.5,        "Potassium",    "mEq/L"),
    "creatinine":       (0.5,     1.3,     0.4,       1.8,        "Serum Creatinine", "mg/dL"),
}

# Feature keys relevant to each disease (used to check clinical status)
_DISEASE_FEATURES: dict[str, list[str]] = {
    "diabetes": ["bmi", "hba1c", "blood_glucose"],
    "obesity":  ["bmi"],
    "kidney":   ["systolic_bp", "sodium", "potassium", "creatinine", "bmi"],
}

# ── Obesity model labels that indicate elevated weight ───────────────
_OBESITY_ELEVATED_LABELS = {
    "overweight", "obese class i", "obese class ii",
    "obese class iii", "obese class iv",
}
_OBESITY_DISEASE_LABELS = {
    "obese class i", "obese class ii", "obese class iii", "obese class iv",
}


# ════════════════════════════════════════════════════════════════════
#  OUTPUT DATACLASS
# ════════════════════════════════════════════════════════════════════

@dataclass
class RiskResult:
    """Unified risk result consumed by every UI component."""
    final_status:         str
    card_level:           str          # "high" | "medium" | "low"
    card_risk_text:       str          # text inside badge, e.g. "High Risk"
    banner_level:         str          # "danger" | "warning" | "info" | "ok"
    banner_title:         str
    banner_body:          str
    model_probability:    float        # 0–100
    abnormal_markers:     list[str]    = field(default_factory=list)
    borderline_markers:   list[str]    = field(default_factory=list)
    all_markers_normal:   bool         = True
    clinical_status_text: str         = ""


# ════════════════════════════════════════════════════════════════════
#  INTERNAL HELPERS
# ════════════════════════════════════════════════════════════════════

def _check_markers(disease_key: str,
                   clinical_values: dict[str, float]
                   ) -> tuple[list[str], list[str]]:
    """Return (abnormal_markers, borderline_markers) for the given disease."""
    abnormal: list[str] = []
    borderline: list[str] = []
    for feat_key in _DISEASE_FEATURES.get(disease_key, []):
        if feat_key not in CLINICAL_RANGES:
            continue
        val = clinical_values.get(feat_key)
        if val is None:
            continue
        lo_n, hi_n, lo_b, hi_b, display, unit = CLINICAL_RANGES[feat_key]
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue
        if lo_n <= v <= hi_n:
            pass  # normal
        elif lo_b <= v <= hi_b:
            borderline.append(display)
        else:
            abnormal.append(display)
    return abnormal, borderline


def _fmt_list(items: list[str]) -> str:
    return ", ".join(f"<strong>{x}</strong>" for x in items)


# ════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ════════════════════════════════════════════════════════════════════

def classify_risk(
    disease_key: str,
    model_label: str,
    model_probability: float,
    clinical_values: dict[str, float],
) -> RiskResult:
    """Classify risk by combining ML model output with clinical marker status.

    Parameters
    ----------
    disease_key       : "diabetes" | "obesity" | "kidney"
    model_label       : decoded prediction label from the model
    model_probability : 0–100 float, confidence of the predicted class
    clinical_values   : dict mapping feature keys to patient values
                        (same keys as CLINICAL_RANGES)

    Returns
    -------
    RiskResult — all fields populated, ready for direct UI consumption
    """
    label_lower = str(model_label).lower().strip()
    prob = float(model_probability) if model_probability is not None else 0.0

    abnormal, borderline = _check_markers(disease_key, clinical_values)
    all_normal = (len(abnormal) == 0 and len(borderline) == 0)
    has_abnormal = len(abnormal) > 0
    has_borderline = len(borderline) > 0

    # ── Determine whether the model signals disease ──────────────────
    if disease_key == "diabetes":
        model_disease = "no diabetes" not in label_lower and "normal" not in label_lower \
                        and "underweight" not in label_lower
        model_disease_confirmed = "diabetes" in label_lower  # includes prediabetes

    elif disease_key == "obesity":
        model_disease = any(x in label_lower for x in _OBESITY_ELEVATED_LABELS)
        model_disease_confirmed = any(x in label_lower for x in _OBESITY_DISEASE_LABELS)

    elif disease_key == "kidney":
        model_disease = "kidney disease" in label_lower
        model_disease_confirmed = model_disease

    else:
        model_disease = False
        model_disease_confirmed = False

    prob_str = f"{prob:.1f}%"

    # ── Classification logic ─────────────────────────────────────────
    if not model_disease:
        # Model says no disease
        if has_abnormal:
            # Unusual: model says OK but some markers are outside range
            return RiskResult(
                final_status       = "Borderline",
                card_level         = "medium",
                card_risk_text     = "Borderline",
                banner_level       = "warning",
                banner_title       = "Borderline — Monitor Values",
                banner_body        = (
                    f"The model predicts <strong>no {_disease_name(disease_key)}</strong> "
                    f"(model probability: {prob_str}), but "
                    f"{_fmt_list(abnormal)} "
                    "is outside the normal reference range. "
                    "Please monitor and consult a clinician."
                ),
                model_probability  = prob,
                abnormal_markers   = abnormal,
                borderline_markers = borderline,
                all_markers_normal = all_normal,
                clinical_status_text = _clinical_text(abnormal, borderline),
            )
        else:
            # Model OK, all markers normal — clear result
            return RiskResult(
                final_status       = "Low Risk",
                card_level         = "low",
                card_risk_text     = "Low Risk",
                banner_level       = "ok",
                banner_title       = _no_disease_title(disease_key),
                banner_body        = (
                    f"The model predicts <strong>no {_disease_name(disease_key)}</strong>. "
                    "All measured markers are within their normal reference ranges."
                ),
                model_probability  = prob,
                abnormal_markers   = [],
                borderline_markers = borderline,
                all_markers_normal = all_normal,
                clinical_status_text = "All markers normal.",
            )

    # ── Model predicts disease ───────────────────────────────────────
    if has_abnormal:
        # Model + clinical evidence agree — confirmed risk
        return RiskResult(
            final_status       = "High Risk",
            card_level         = "high",
            card_risk_text     = "High Risk",
            banner_level       = "danger",
            banner_title       = f"Elevated {_disease_name(disease_key)} Risk",
            banner_body        = (
                f"The model predicts <strong>{model_label}</strong> "
                f"(model probability: {prob_str}). "
                f"Clinical marker(s) outside normal range: {_fmt_list(abnormal)}. "
                "Please consult a specialist."
            ),
            model_probability  = prob,
            abnormal_markers   = abnormal,
            borderline_markers = borderline,
            all_markers_normal = False,
            clinical_status_text = _clinical_text(abnormal, borderline),
        )

    if has_borderline:
        # Model predicts disease + borderline values
        return RiskResult(
            final_status       = "Moderate Risk",
            card_level         = "medium",
            card_risk_text     = "Moderate Risk",
            banner_level       = "warning",
            banner_title       = f"Moderate {_disease_name(disease_key)} Risk",
            banner_body        = (
                f"The model predicts <strong>{model_label}</strong> "
                f"(model probability: {prob_str}). "
                f"Borderline marker(s): {_fmt_list(borderline)}. "
                "Monitor closely and consult a clinician."
            ),
            model_probability  = prob,
            abnormal_markers   = [],
            borderline_markers = borderline,
            all_markers_normal = False,
            clinical_status_text = _clinical_text([], borderline),
        )

    # Model predicts disease but ALL clinical markers are within normal range.
    # This is the key case: do NOT say "High Risk" or "Disease Detected".
    return RiskResult(
        final_status       = "Model Flag",
        card_level         = "medium",
        card_risk_text     = "Screening Flag",
        banner_level       = "info",
        banner_title       = f"Screening Flag — Clinical Markers Normal",
        banner_body        = (
            f"The screening model flagged <strong>{model_label}</strong> "
            f"(model probability: {prob_str}). "
            "However, <strong>all measured clinical markers are within "
            "their normal reference ranges</strong>. "
            "This result may reflect a model pattern rather than a clinical finding. "
            "A model flag alone is not a diagnosis — please seek professional evaluation."
        ),
        model_probability  = prob,
        abnormal_markers   = [],
        borderline_markers = [],
        all_markers_normal = True,
        clinical_status_text = "All markers normal.",
    )


# ════════════════════════════════════════════════════════════════════
#  PRIVATE LABEL HELPERS
# ════════════════════════════════════════════════════════════════════

def _disease_name(disease_key: str) -> str:
    return {"diabetes": "Diabetes", "obesity": "Obesity",
            "kidney": "Kidney Disease"}.get(disease_key, disease_key.title())

def _no_disease_title(disease_key: str) -> str:
    return {"diabetes": "No Diabetes Detected",
            "obesity":  "Healthy Weight",
            "kidney":   "Kidney Health Normal"}.get(disease_key, "No Disease Detected")

def _clinical_text(abnormal: list[str], borderline: list[str]) -> str:
    parts = []
    if abnormal:
        parts.append(f"Abnormal: {', '.join(abnormal)}")
    if borderline:
        parts.append(f"Borderline: {', '.join(borderline)}")
    return "; ".join(parts) if parts else "All markers normal."
