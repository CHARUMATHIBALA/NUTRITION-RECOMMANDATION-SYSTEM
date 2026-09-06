"""
backend/services/severity_rules.py
====================================
Single source of truth for disease-severity-based nutritional constraints.

Architecture
------------
Each disease (diabetes, kidney_disease, obesity) has three severity tiers
(mild, moderate, severe).  Each tier carries:

  hard_constraints  – absolute upper/lower bounds used as DataFrame filters.
                      Foods that violate a hard constraint are excluded entirely.
                      NEVER relax hard constraints regardless of pool size.

  soft_constraints  – preferred bounds used to produce a continuous penalty
                      score in [0, 1].  Foods outside the soft bound are NOT
                      excluded but receive a lower suitability score.
                      Soft constraints may be progressively relaxed when the
                      candidate pool is critically small (see MIN_POOL_SIZE).

  scoring_multipliers – per-nutrient coefficients added to the disease term
                        inside _calculate_nutrition_score().  Higher-severity
                        tiers apply larger penalties/rewards to differentiate
                        the same food across severity levels.

  required_minimums – minimum desirable nutrient amounts.  Foods meeting a
                      minimum receive a score bonus.

Threshold sources
-----------------
The values below are calibrated against the project dataset
(food_dataset.csv, 1 014 rows) and standard dietary guidance for South Indian
foods.  They are NOT claimed to be clinically validated thresholds.
Fields marked  # ⚠ REQUIRES DOMAIN VALIDATION  should be reviewed by a
registered dietitian or clinician before deployment in a clinical setting.

Dataset percentile reference used for calibration
--------------------------------------------------
  Calories (kcal) : p25=101   p75=316  p90=529  max=839
  Free Sugar (g)  : p25=1.15  p75=11.8 p90=26.7 max=85.6
  Sodium (mg)     : p25=41.9  p75=196.6 p90=380  max=2000
  Protein (g)     : p25=2.1   p75=6.6  p90=9.5  max=21.6
  Fats (g)        : p25=4.0   p75=17.1 p90=52.9 max=90.5
  Fibre (g)       : p25=0.59  p75=2.35 p90=4.28 max=35.7
  Carbs (g)       : p25=5.6   p75=26.9 p90=45.0 max=86.5

Severity derivation (from RiskResult.final_status)
---------------------------------------------------
  'High Risk'                → 'severe'
  'Moderate Risk'            → 'moderate'
  'Borderline' / 'Model Flag'→ 'mild'
  'Low Risk'                 → None  (no disease restrictions applied)
  None / unknown             → 'moderate'  (safe default, see DEFAULT_SEVERITY)
"""

from __future__ import annotations
from typing import Dict, Optional, Any

# ── Canonical severity level strings ────────────────────────────────────────
SEVERITY_MILD     = "mild"
SEVERITY_MODERATE = "moderate"
SEVERITY_SEVERE   = "severe"

# Ordered list — useful for comparisons
SEVERITY_ORDER = [SEVERITY_MILD, SEVERITY_MODERATE, SEVERITY_SEVERE]

# Default when severity is present but unrecognised
DEFAULT_SEVERITY = SEVERITY_MODERATE

# Minimum candidate pool size before soft constraints are progressively relaxed
MIN_POOL_SIZE = 30

# ── RiskResult.final_status → severity mapping ───────────────────────────────
_FINAL_STATUS_TO_SEVERITY: Dict[str, Optional[str]] = {
    "High Risk":     SEVERITY_SEVERE,
    "Moderate Risk": SEVERITY_MODERATE,
    "Borderline":    SEVERITY_MILD,
    "Model Flag":    SEVERITY_MILD,
    "Low Risk":      None,    # no disease → no disease restrictions
}


def final_status_to_severity(final_status: Optional[str]) -> Optional[str]:
    """
    Convert a RiskResult.final_status string to a severity level string.

    Returns one of 'mild' | 'moderate' | 'severe' | None.
    Returns DEFAULT_SEVERITY when the value is present but unrecognised
    (defensive — avoids silent gaps).
    Returns None when final_status is None or empty (no disease).
    """
    if not final_status:
        return None
    result = _FINAL_STATUS_TO_SEVERITY.get(final_status)
    if result is None and final_status not in _FINAL_STATUS_TO_SEVERITY:
        # Unrecognised status → safe fallback
        return DEFAULT_SEVERITY
    return result


# ────────────────────────────────────────────────────────────────────────────
#  SEVERITY RULES
#
#  Structure per disease per severity tier:
#
#  hard_constraints: {
#      column_name: {"max": float} | {"min": float} | {"max": float, "min": float}
#  }
#  soft_constraints: {
#      column_name: {"max": float} | {"min": float}   (preferred bounds)
#  }
#  scoring_multipliers: {
#      nutrient_key: float    (multiplier applied to penalty/bonus term)
#  }
#  required_minimums: {
#      column_name: float     (below this value → no bonus; at/above → +bonus)
#  }
# ────────────────────────────────────────────────────────────────────────────

SEVERITY_RULES: Dict[str, Dict[str, Dict]] = {

    # ════════════════════════════════════════════════════════════════════
    #  DIABETES
    #  Key nutritional levers: free sugar, carbohydrates, calories, fibre
    #  Goal: lower GI, higher fibre, controlled carbs
    # ════════════════════════════════════════════════════════════════════
    "diabetes": {

        "mild": {
            # Hard: allow slightly more sugar/carbs than moderate
            # Pool size reference: ~425 foods at the existing moderate level
            # Mild uses looser bounds to preserve diet variety
            "hard_constraints": {
                "Free Sugar (g)":   {"max": 8.0},    # ⚠ REQUIRES DOMAIN VALIDATION
                "Calories (kcal)":  {"max": 350.0},  # ⚠ REQUIRES DOMAIN VALIDATION
            },
            "soft_constraints": {
                "Free Sugar (g)":        {"max": 5.0},
                "Carbohydrates (g)":     {"max": 35.0},
                "Calories (kcal)":       {"max": 300.0},
            },
            "scoring_multipliers": {
                "sugar_penalty":   3.0,   # per-gram penalty on Free Sugar
                "carb_penalty":    0.2,   # per-gram penalty on Carbohydrates
                "fiber_bonus":     2.5,   # per-gram bonus on Fibre
            },
            "required_minimums": {
                "Fibre (g)": 1.5,         # foods with ≥1.5 g fibre get bonus
            },
        },

        "moderate": {
            # Hard: current production thresholds (from config.py baseline)
            "hard_constraints": {
                "Free Sugar (g)":   {"max": 5.0},    # ⚠ REQUIRES DOMAIN VALIDATION
                "Calories (kcal)":  {"max": 250.0},  # ⚠ REQUIRES DOMAIN VALIDATION
            },
            "soft_constraints": {
                "Free Sugar (g)":        {"max": 3.0},
                "Carbohydrates (g)":     {"max": 25.0},
                "Calories (kcal)":       {"max": 220.0},
            },
            "scoring_multipliers": {
                "sugar_penalty":   4.0,
                "carb_penalty":    0.3,
                "fiber_bonus":     3.0,
            },
            "required_minimums": {
                "Fibre (g)": 2.0,
            },
        },

        "severe": {
            # Hard: strictest allowable thresholds
            # Designed to keep only low-GI, very-low-sugar foods
            "hard_constraints": {
                "Free Sugar (g)":   {"max": 2.0},    # ⚠ REQUIRES DOMAIN VALIDATION
                "Calories (kcal)":  {"max": 200.0},  # ⚠ REQUIRES DOMAIN VALIDATION
                "Carbohydrates (g)": {"max": 20.0},  # ⚠ REQUIRES DOMAIN VALIDATION
            },
            "soft_constraints": {
                "Free Sugar (g)":        {"max": 1.0},
                "Carbohydrates (g)":     {"max": 15.0},
                "Calories (kcal)":       {"max": 175.0},
            },
            "scoring_multipliers": {
                "sugar_penalty":   6.0,
                "carb_penalty":    0.5,
                "fiber_bonus":     4.0,
            },
            "required_minimums": {
                "Fibre (g)": 2.5,
            },
        },
    },

    # ════════════════════════════════════════════════════════════════════
    #  KIDNEY DISEASE
    #  Key nutritional levers: sodium, protein, potassium (dataset has no
    #  potassium column — handled via sodium + protein)
    #  Goal: reduce kidney workload, manage fluid balance
    # ════════════════════════════════════════════════════════════════════
    "kidney_disease": {

        "mild": {
            "hard_constraints": {
                "Sodium (mg)":  {"max": 150.0},   # ⚠ REQUIRES DOMAIN VALIDATION
                "Protein (g)":  {"max": 15.0},    # ⚠ REQUIRES DOMAIN VALIDATION
            },
            "soft_constraints": {
                "Sodium (mg)":  {"max": 100.0},
                "Protein (g)":  {"max": 12.0},
            },
            "scoring_multipliers": {
                "sodium_penalty":  0.08,   # per-mg penalty on Sodium
                "protein_penalty": 0.4,    # per-g penalty on Protein
            },
            "required_minimums": {},
        },

        "moderate": {
            # Hard: current production thresholds
            "hard_constraints": {
                "Sodium (mg)":  {"max": 100.0},   # ⚠ REQUIRES DOMAIN VALIDATION
                "Protein (g)":  {"max": 12.0},    # ⚠ REQUIRES DOMAIN VALIDATION
            },
            "soft_constraints": {
                "Sodium (mg)":  {"max": 75.0},
                "Protein (g)":  {"max": 9.0},
            },
            "scoring_multipliers": {
                "sodium_penalty":  0.12,
                "protein_penalty": 0.55,
            },
            "required_minimums": {},
        },

        "severe": {
            # Hard: very low sodium, very controlled protein
            "hard_constraints": {
                "Sodium (mg)":  {"max": 60.0},    # ⚠ REQUIRES DOMAIN VALIDATION
                "Protein (g)":  {"max": 8.0},     # ⚠ REQUIRES DOMAIN VALIDATION
            },
            "soft_constraints": {
                "Sodium (mg)":  {"max": 40.0},
                "Protein (g)":  {"max": 6.0},
                "Fats (g)":     {"max": 12.0},    # secondary concern at severe
            },
            "scoring_multipliers": {
                "sodium_penalty":  0.18,
                "protein_penalty": 0.75,
            },
            "required_minimums": {},
        },
    },

    # ════════════════════════════════════════════════════════════════════
    #  OBESITY
    #  Key nutritional levers: calories, fats, fibre, protein
    #  Goal: energy deficit, satiety, preserve lean mass
    # ════════════════════════════════════════════════════════════════════
    "obesity": {

        "mild": {
            "hard_constraints": {
                "Calories (kcal)": {"max": 350.0},   # ⚠ REQUIRES DOMAIN VALIDATION
                "Fats (g)":        {"max": 18.0},    # ⚠ REQUIRES DOMAIN VALIDATION
            },
            "soft_constraints": {
                "Calories (kcal)": {"max": 300.0},
                "Fats (g)":        {"max": 12.0},
                "Free Sugar (g)":  {"max": 8.0},
            },
            "scoring_multipliers": {
                "calorie_penalty": 0.025,  # per-kcal penalty
                "fat_penalty":     0.4,
                "protein_bonus":   1.5,
                "fiber_bonus":     2.0,
            },
            "required_minimums": {
                "Protein (g)": 3.0,
            },
        },

        "moderate": {
            # Hard: current production thresholds
            "hard_constraints": {
                "Calories (kcal)": {"max": 250.0},   # ⚠ REQUIRES DOMAIN VALIDATION
                "Fats (g)":        {"max": 10.0},    # ⚠ REQUIRES DOMAIN VALIDATION
            },
            "soft_constraints": {
                "Calories (kcal)": {"max": 200.0},
                "Fats (g)":        {"max": 7.0},
                "Free Sugar (g)":  {"max": 5.0},
            },
            "scoring_multipliers": {
                "calorie_penalty": 0.04,
                "fat_penalty":     0.6,
                "protein_bonus":   2.0,
                "fiber_bonus":     2.5,
            },
            "required_minimums": {
                "Protein (g)": 4.0,
            },
        },

        "severe": {
            # Hard: very low calorie, very low fat
            "hard_constraints": {
                "Calories (kcal)": {"max": 180.0},   # ⚠ REQUIRES DOMAIN VALIDATION
                "Fats (g)":        {"max": 6.0},     # ⚠ REQUIRES DOMAIN VALIDATION
                "Free Sugar (g)":  {"max": 3.0},     # ⚠ REQUIRES DOMAIN VALIDATION
            },
            "soft_constraints": {
                "Calories (kcal)": {"max": 150.0},
                "Fats (g)":        {"max": 4.0},
                "Free Sugar (g)":  {"max": 2.0},
            },
            "scoring_multipliers": {
                "calorie_penalty": 0.06,
                "fat_penalty":     0.9,
                "protein_bonus":   2.5,
                "fiber_bonus":     3.0,
            },
            "required_minimums": {
                "Protein (g)": 5.0,
            },
        },
    },
}


# ────────────────────────────────────────────────────────────────────────────
#  PUBLIC ACCESSOR FUNCTIONS
# ────────────────────────────────────────────────────────────────────────────

def get_rules(disease_key: str, severity: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Return the rule dict for (disease_key, severity), or None if not found.

    Args:
        disease_key: One of 'diabetes', 'kidney_disease', 'obesity'
                     (case-insensitive, normalised internally).
        severity:    One of 'mild', 'moderate', 'severe', or None.
                     None → no disease-specific rules (healthy user).

    Returns:
        Rule dict with keys 'hard_constraints', 'soft_constraints',
        'scoring_multipliers', 'required_minimums' — or None.
    """
    if severity is None:
        return None

    norm_disease  = _normalise_disease_key(disease_key)
    norm_severity = _normalise_severity(severity)

    disease_block = SEVERITY_RULES.get(norm_disease)
    if disease_block is None:
        return None

    return disease_block.get(norm_severity)


def get_hard_constraints(disease_key: str, severity: Optional[str]) -> Dict[str, Dict]:
    """Return hard_constraints dict for (disease_key, severity)."""
    rules = get_rules(disease_key, severity)
    return rules.get("hard_constraints", {}) if rules else {}


def get_soft_constraints(disease_key: str, severity: Optional[str]) -> Dict[str, Dict]:
    """Return soft_constraints dict for (disease_key, severity)."""
    rules = get_rules(disease_key, severity)
    return rules.get("soft_constraints", {}) if rules else {}


def get_scoring_multipliers(disease_key: str, severity: Optional[str]) -> Dict[str, float]:
    """Return scoring_multipliers dict for (disease_key, severity)."""
    rules = get_rules(disease_key, severity)
    return rules.get("scoring_multipliers", {}) if rules else {}


def get_required_minimums(disease_key: str, severity: Optional[str]) -> Dict[str, float]:
    """Return required_minimums dict for (disease_key, severity)."""
    rules = get_rules(disease_key, severity)
    return rules.get("required_minimums", {}) if rules else {}


def build_severity_dict(
    diabetes_final_status:      Optional[str],
    kidney_final_status:        Optional[str],
    obesity_final_status:       Optional[str],
) -> Dict[str, Optional[str]]:
    """
    Convert the three disease RiskResult.final_status strings into a
    severity dict suitable for storing in user_profile['severity'].

    Args:
        diabetes_final_status: RiskResult.final_status for diabetes prediction
        kidney_final_status:   RiskResult.final_status for kidney prediction
        obesity_final_status:  RiskResult.final_status for obesity prediction

    Returns:
        {'diabetes': severity_str_or_None,
         'kidney_disease': severity_str_or_None,
         'obesity': severity_str_or_None}

    Examples:
        build_severity_dict('High Risk', 'Low Risk', 'Moderate Risk')
        → {'diabetes': 'severe', 'kidney_disease': None, 'obesity': 'moderate'}
    """
    return {
        "diabetes":       final_status_to_severity(diabetes_final_status),
        "kidney_disease": final_status_to_severity(kidney_final_status),
        "obesity":        final_status_to_severity(obesity_final_status),
    }


def get_severity_for_disease_list(
    diseases: list,
    severity_dict: Optional[Dict[str, Optional[str]]],
) -> Dict[str, Optional[str]]:
    """
    Filter the severity dict to only the diseases present in the diseases list.

    Args:
        diseases:      List of active disease strings from app.py
                       (e.g. ['Diabetes', 'Obesity']).
        severity_dict: Full severity dict from build_severity_dict().

    Returns:
        Dict mapping normalised disease key → severity, for active diseases only.
        Returns {} if severity_dict is None.

    Example:
        diseases = ['Diabetes', 'Obesity']
        severity_dict = {'diabetes': 'severe', 'kidney_disease': None, 'obesity': 'moderate'}
        → {'diabetes': 'severe', 'obesity': 'moderate'}
    """
    if not severity_dict:
        return {}

    result: Dict[str, Optional[str]] = {}
    for disease in diseases:
        key = _normalise_disease_key(str(disease))
        if key in severity_dict:
            result[key] = severity_dict[key]
    return result


# ────────────────────────────────────────────────────────────────────────────
#  SEVERITY SUITABILITY SCORE
# ────────────────────────────────────────────────────────────────────────────

def calculate_severity_suitability_score(
    row: "pd.Series",
    disease_key: str,
    severity: Optional[str],
) -> float:
    """
    Calculate a [0, 1] suitability score for a food item given disease + severity.

    Logic:
    1. Hard constraint violation → 0.0  (caller should have already filtered,
       but this acts as a second safety net).
    2. For each soft constraint: compute a partial penalty proportional to
       how far the food exceeds the preferred bound.
    3. For each required_minimum: add a small bonus if met.
    4. Final score = 1.0 − total_penalty + bonus, clamped to [0, 1].

    A food perfectly within all soft bounds scores 1.0.
    A food that just exceeds a soft bound gets a proportional reduction.
    The score never goes negative.

    Returns 0.5 (neutral) when severity is None (healthy user, no penalty).
    """
    import pandas as _pd
    import numpy as _np

    if severity is None:
        return 0.5   # neutral — no disease penalty for healthy users

    rules = get_rules(disease_key, severity)
    if rules is None:
        return 0.5

    # ── Hard constraint check (safety net) ───────────────────────────
    for col, bounds in rules.get("hard_constraints", {}).items():
        try:
            val = float(row.get(col, 0) or 0)
        except (TypeError, ValueError):
            val = 0.0
        if "max" in bounds and val > bounds["max"]:
            return 0.0
        if "min" in bounds and val < bounds["min"]:
            return 0.0

    # ── Soft constraint penalty ───────────────────────────────────────
    total_penalty = 0.0
    num_soft = 0

    for col, bounds in rules.get("soft_constraints", {}).items():
        try:
            val = float(row.get(col, 0) or 0)
        except (TypeError, ValueError):
            val = 0.0
        num_soft += 1

        if "max" in bounds:
            soft_max = bounds["max"]
            if val > soft_max:
                # Penalty proportional to excess, normalised against the bound
                # Cap penalty at 0.5 per nutrient to preserve diversity
                excess_ratio = (val - soft_max) / max(soft_max, 1.0)
                total_penalty += min(0.5, excess_ratio * 0.4)

        if "min" in bounds:
            soft_min = bounds["min"]
            if val < soft_min:
                deficit_ratio = (soft_min - val) / max(soft_min, 1.0)
                total_penalty += min(0.5, deficit_ratio * 0.4)

    # ── Required minimum bonus ────────────────────────────────────────
    bonus = 0.0
    for col, min_val in rules.get("required_minimums", {}).items():
        try:
            val = float(row.get(col, 0) or 0)
        except (TypeError, ValueError):
            val = 0.0
        if val >= min_val:
            bonus += 0.05   # small fixed bonus per satisfied minimum

    # ── Combine ───────────────────────────────────────────────────────
    score = 1.0 - total_penalty + bonus
    return max(0.0, min(1.0, score))


# ────────────────────────────────────────────────────────────────────────────
#  INTERNAL NORMALISERS
# ────────────────────────────────────────────────────────────────────────────

def _normalise_disease_key(raw: str) -> str:
    """
    Map free-form disease strings to canonical SEVERITY_RULES keys.

      'Diabetes'       → 'diabetes'
      'Kidney Disease' → 'kidney_disease'
      'Obesity'        → 'obesity'
      'Overweight'     → 'obesity'

    Returns the lowercase normalised key, or the lowercased input if not
    recognised (caller must handle the None result from get_rules).
    """
    s = str(raw).lower().strip()
    if "diabetes" in s:
        return "diabetes"
    if "kidney" in s:
        return "kidney_disease"
    if "obesity" in s or "overweight" in s or "obese" in s:
        return "obesity"
    return s


def _normalise_severity(raw: str) -> str:
    """
    Normalise a severity string to one of the canonical level strings.

    Accepts:
      'mild', 'Mild', 'MILD'  → 'mild'
      'moderate', ...          → 'moderate'
      'severe', 'critical'     → 'severe'
      anything else            → DEFAULT_SEVERITY
    """
    s = str(raw).lower().strip()
    if s in ("mild", "low", "minor"):
        return SEVERITY_MILD
    if s in ("moderate", "medium", "moderate risk"):
        return SEVERITY_MODERATE
    if s in ("severe", "high", "critical", "high risk"):
        return SEVERITY_SEVERE
    return DEFAULT_SEVERITY


# ────────────────────────────────────────────────────────────────────────────
#  EXPLANATION HELPERS
# ────────────────────────────────────────────────────────────────────────────

def get_severity_explanation(disease_key: str, severity: Optional[str]) -> str:
    """
    Return a plain-language explanation of the severity-level constraints
    applied to a given disease.  Safe to surface in the UI.

    Does NOT make clinical claims — only explains what the system is doing.
    """
    if severity is None:
        return "No disease-specific restrictions applied."

    norm_d = _normalise_disease_key(disease_key)
    norm_s = _normalise_severity(severity)

    _explanations = {
        ("diabetes", "mild"):
            "Mild diabetes risk: foods with free sugar ≤ 8 g and calories ≤ 350 kcal "
            "are included. Higher-fibre options are preferred.",
        ("diabetes", "moderate"):
            "Moderate diabetes risk: stricter sugar limit (≤ 5 g) and calorie cap "
            "(≤ 250 kcal). High-fibre, low-carbohydrate foods are prioritised.",
        ("diabetes", "severe"):
            "Severe diabetes risk: very strict constraints — sugar ≤ 2 g, "
            "carbohydrates ≤ 20 g, calories ≤ 200 kcal. Only the most suitable "
            "low-GI foods are recommended.",

        ("kidney_disease", "mild"):
            "Mild kidney disease risk: sodium limited to ≤ 150 mg and protein to "
            "≤ 15 g per serving.",
        ("kidney_disease", "moderate"):
            "Moderate kidney disease risk: sodium ≤ 100 mg and protein ≤ 12 g. "
            "Lower-sodium foods are strongly preferred.",
        ("kidney_disease", "severe"):
            "Severe kidney disease risk: very low sodium (≤ 60 mg) and protein "
            "(≤ 8 g). Only kidney-friendly foods are recommended.",

        ("obesity", "mild"):
            "Mild obesity risk: calorie cap at ≤ 350 kcal and fat ≤ 18 g. "
            "Higher-protein and higher-fibre foods are preferred.",
        ("obesity", "moderate"):
            "Moderate obesity risk: calories ≤ 250 kcal, fat ≤ 10 g. "
            "Protein-rich, fibre-rich foods are strongly prioritised.",
        ("obesity", "severe"):
            "Severe obesity risk: very strict calorie (≤ 180 kcal), fat (≤ 6 g), "
            "and sugar (≤ 3 g) limits. Only the leanest, most nutrient-dense "
            "options are recommended.",
    }

    return _explanations.get(
        (norm_d, norm_s),
        f"{norm_d.replace('_', ' ').title()} — {norm_s} severity restrictions applied.",
    )


def get_recommendation_reason(
    row: "pd.Series",
    disease_key: str,
    severity: Optional[str],
) -> str:
    """
    Generate a short, food-specific reason string explaining why this food
    was recommended (or given a lower score) for the given disease + severity.

    Safe for UI display.  Does NOT make medical claims.
    """
    if severity is None:
        return "Suitable for a balanced diet."

    norm_d = _normalise_disease_key(disease_key)
    norm_s = _normalise_severity(severity) if severity else DEFAULT_SEVERITY

    reasons = []

    try:
        sugar    = float(row.get("Free Sugar (g)", 0) or 0)
        sodium   = float(row.get("Sodium (mg)", 0) or 0)
        protein  = float(row.get("Protein (g)", 0) or 0)
        calories = float(row.get("Calories (kcal)", 0) or 0)
        fiber    = float(row.get("Fibre (g)", 0) or 0)
        fats     = float(row.get("Fats (g)", 0) or 0)
        carbs    = float(row.get("Carbohydrates (g)", 0) or 0)
    except (TypeError, ValueError):
        return "Suitable based on nutritional profile."

    if norm_d == "diabetes":
        if sugar <= 2:
            reasons.append("Very low free sugar")
        elif sugar <= 5:
            reasons.append("Low free sugar")
        if fiber >= 2.5:
            reasons.append(f"Good fibre content ({fiber:.1f} g)")
        if carbs <= 20:
            reasons.append("Low carbohydrate")

    elif norm_d == "kidney_disease":
        if sodium <= 60:
            reasons.append("Very low sodium")
        elif sodium <= 100:
            reasons.append("Low sodium")
        if protein <= 8:
            reasons.append(f"Controlled protein ({protein:.1f} g)")

    elif norm_d == "obesity":
        if calories <= 150:
            reasons.append("Very low calorie")
        elif calories <= 250:
            reasons.append(f"Low calorie ({calories:.0f} kcal)")
        if protein >= 5:
            reasons.append(f"Good protein content ({protein:.1f} g)")
        if fiber >= 2:
            reasons.append(f"High fibre ({fiber:.1f} g)")
        if fats <= 5:
            reasons.append("Low fat")

    if not reasons:
        reasons.append(f"Meets {norm_s} {norm_d.replace('_', ' ')} dietary guidelines")

    return "; ".join(reasons) + "."
