"""predict.py — Disease prediction functions for Smart Health Dashboard.

Three public functions
----------------------
predict_obesity(age, gender, bmi)
predict_diabetes(age, gender, bmi, hba1c, glucose)
predict_kidney(age, gender, bmi, sodium, potassium, bp, creatinine)

Return format (all three functions)
------------------------------------
{
    "label":      str,          # human-readable model output label
    "confidence": float | None, # 0–100 %, probability of the PREDICTED class
    "risk":       RiskResult,   # from backend.risk — single source of truth
}

The "risk" key carries every piece of information the UI needs:
    risk.final_status       → "High Risk" | "Moderate Risk" | "Model Flag" |
                              "Low Risk" | "Borderline"
    risk.card_level         → "high" | "medium" | "low"
    risk.card_risk_text     → short badge label
    risk.banner_level       → "danger" | "warning" | "info" | "ok"
    risk.banner_title       → alert title
    risk.banner_body        → alert body (may contain HTML)
    risk.model_probability  → 0–100 float
    risk.abnormal_markers   → list of display names outside normal range
    risk.all_markers_normal → bool

The UI MUST consume risk.* and MUST NOT re-derive risk from the label string.

Error handling
--------------
- Missing model file     → {"label": "...", "confidence": None, "risk": error_risk}
- Invalid input          → {"label": "...", "confidence": None, "risk": error_risk}
- Prediction exception   → {"label": "...", "confidence": None, "risk": error_risk}
- Never raises.
"""

import pandas as pd
import models
from backend.risk import classify_risk, RiskResult


# ════════════════════════════════════════════════════════════════════
#  OBESITY LABEL MAP
# ════════════════════════════════════════════════════════════════════
_OBESITY_LABEL_MAP: dict[int, str] = {
    0: "Underweight",
    1: "Normal Weight",
    2: "Obese Class I",
    3: "Obese Class II",
    4: "Obese Class III",
    5: "Overweight",
    6: "Overweight",
}


# ════════════════════════════════════════════════════════════════════
#  INTERNAL HELPERS
# ════════════════════════════════════════════════════════════════════

def _proba_of_predicted(model, X: pd.DataFrame) -> "float | None":
    """Return the probability of the PREDICTED class as a plain Python float.

    Uses model.predict_proba() and selects the probability that corresponds
    to the class actually predicted by model.predict(), not a fixed index.
    Always returns a native float (safe for json.dumps and Streamlit display).
    """
    if not hasattr(model, "predict_proba"):
        return None
    try:
        raw_pred  = model.predict(X)[0]
        proba     = model.predict_proba(X)[0]
        pred_idx  = list(model.classes_).index(raw_pred)
        return float(round(float(proba[pred_idx]) * 100, 1))
    except Exception:
        return None


def _check_missing(model_attr_name: str, filename: str) -> "dict | None":
    obj = getattr(models, model_attr_name, None)
    if obj is None:
        _err_risk = _error_risk(f"{filename} not loaded")
        return {"label": f"Model unavailable — {filename} not found",
                "confidence": None, "risk": _err_risk}
    return None


def _validate_numeric(value, name: str, lo: float, hi: float) -> "str | None":
    try:
        v = float(value)
    except (TypeError, ValueError):
        return f"Invalid input — {name} must be a number"
    if not (lo <= v <= hi):
        return f"Invalid input — {name} {v} is outside expected range [{lo}, {hi}]"
    return None


def _error_risk(reason: str) -> RiskResult:
    """Return a neutral RiskResult used when prediction cannot run."""
    return RiskResult(
        final_status       = "Unavailable",
        card_level         = "low",
        card_risk_text     = "Unavailable",
        banner_level       = "info",
        banner_title       = "Prediction Unavailable",
        banner_body        = f"Could not run prediction: {reason}.",
        model_probability  = 0.0,
        abnormal_markers   = [],
        borderline_markers = [],
        all_markers_normal = True,
        clinical_status_text = "",
    )


# ════════════════════════════════════════════════════════════════════
#  OBESITY PREDICTION
# ════════════════════════════════════════════════════════════════════

def predict_obesity(age, gender: str, bmi) -> dict:
    err = _check_missing("obesity_model", "obesity_model.pkl")
    if err:
        return err

    for val, name, lo, hi in [
        (age, "age", 1, 120),
        (bmi, "bmi", 10, 70),
    ]:
        msg = _validate_numeric(val, name, lo, hi)
        if msg:
            return {"label": msg, "confidence": None, "risk": _error_risk(msg)}

    if str(gender).strip() not in ("Male", "Female"):
        msg = f"Invalid input — gender must be Male or Female"
        return {"label": msg, "confidence": None, "risk": _error_risk(msg)}

    try:
        gender_enc = models.encode_gender(gender)
        X = pd.DataFrame(
            [[float(age), gender_enc, float(bmi)]],
            columns=["age", "gender", "bmi"],
        )
        raw_pred = models.obesity_model.predict(X)
        pred_val = int(raw_pred[0])
        label    = _OBESITY_LABEL_MAP.get(pred_val, f"Unknown (class {pred_val})")
        conf     = _proba_of_predicted(models.obesity_model, X)

        clinical_values = {"bmi": float(bmi)}
        risk = classify_risk("obesity", label, conf or 0.0, clinical_values)

        return {"label": label, "confidence": conf, "risk": risk}

    except Exception as exc:
        msg = f"Prediction error — {exc}"
        return {"label": msg, "confidence": None, "risk": _error_risk(str(exc))}


# ════════════════════════════════════════════════════════════════════
#  DIABETES PREDICTION
# ════════════════════════════════════════════════════════════════════

def predict_diabetes(age, gender: str, bmi, hba1c, glucose) -> dict:
    err = _check_missing("disease_model", "disease_model.pkl")
    if err:
        return err

    for val, name, lo, hi in [
        (age,     "age",           1,    120),
        (bmi,     "bmi",          10,     70),
        (hba1c,   "HbA1c",         3.0,  15.0),
        (glucose, "blood glucose", 50,   500),
    ]:
        msg = _validate_numeric(val, name, lo, hi)
        if msg:
            return {"label": msg, "confidence": None, "risk": _error_risk(msg)}

    if str(gender).strip() not in ("Male", "Female"):
        msg = "Invalid input — gender must be Male or Female"
        return {"label": msg, "confidence": None, "risk": _error_risk(msg)}

    try:
        gender_enc = models.encode_gender(gender)
        X = pd.DataFrame(
            [[float(age), gender_enc, float(bmi), float(hba1c), float(glucose)]],
            columns=["age", "gender", "bmi", "HbA1c", "blood glucose"],
        )
        raw_pred = models.disease_model.predict(X)
        label    = models.decode_diabetes(raw_pred)
        conf     = _proba_of_predicted(models.disease_model, X)

        clinical_values = {
            "bmi":          float(bmi),
            "hba1c":        float(hba1c),
            "blood_glucose": float(glucose),
        }
        risk = classify_risk("diabetes", label, conf or 0.0, clinical_values)

        return {"label": label, "confidence": conf, "risk": risk}

    except Exception as exc:
        msg = f"Prediction error — {exc}"
        return {"label": msg, "confidence": None, "risk": _error_risk(str(exc))}


# ════════════════════════════════════════════════════════════════════
#  KIDNEY DISEASE PREDICTION
# ════════════════════════════════════════════════════════════════════

def predict_kidney(age, gender: str, bmi,
                   sodium, potassium, bp, creatinine) -> dict:
    err = _check_missing("kidney_model", "kidney_model.pkl")
    if err:
        return err

    for val, name, lo, hi in [
        (age,        "age",           1,    120),
        (bmi,        "bmi",          10,     70),
        (sodium,     "sodium",       115,   170),
        (potassium,  "potassium",     2.0,   7.0),
        (bp,         "blood pressure", 80,  200),
        (creatinine, "creatinine",    0.1,  15.0),
    ]:
        msg = _validate_numeric(val, name, lo, hi)
        if msg:
            return {"label": msg, "confidence": None, "risk": _error_risk(msg)}

    if str(gender).strip() not in ("Male", "Female"):
        msg = "Invalid input — gender must be Male or Female"
        return {"label": msg, "confidence": None, "risk": _error_risk(msg)}

    try:
        gender_enc = models.encode_gender(gender)
        X = pd.DataFrame(
            [[float(age), gender_enc, float(bmi),
              float(sodium), float(potassium),
              float(bp), float(creatinine)]],
            columns=["age", "gender", "bmi",
                     "sodium", "potassium",
                     "BloodPressure", "SerumCreatinine"],
        )
        raw_pred = models.kidney_model.predict(X)
        label    = models.decode_kidney(raw_pred)
        conf     = _proba_of_predicted(models.kidney_model, X)

        clinical_values = {
            "systolic_bp": float(bp),
            "sodium":      float(sodium),
            "potassium":   float(potassium),
            "creatinine":  float(creatinine),
            "bmi":         float(bmi),
        }
        risk = classify_risk("kidney", label, conf or 0.0, clinical_values)

        return {"label": label, "confidence": conf, "risk": risk}

    except Exception as exc:
        msg = f"Prediction error — {exc}"
        return {"label": msg, "confidence": None, "risk": _error_risk(str(exc))}
