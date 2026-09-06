# Central configuration for Smart Health Dashboard

# Disease-specific thresholds
DIABETES_MAX_SUGAR = 5          # g, maximum free sugar for diabetes filter
DIABETES_MAX_CALORIES = 250    # kcal, max calories for diabetes
KIDNEY_MAX_SODIUM = 100        # mg, max sodium for kidney disease
KIDNEY_MAX_PROTEIN = 12        # g, hard ceiling for protein in kidney disease
OBESITY_MAX_CALORIES = 250    # kcal, max calories for obesity filter
OBESITY_MAX_FAT = 10           # g, max fat for obesity filter

# General thresholds
FREE_SUGAR_MAX_NORMAL = 15    # g, normal free sugar limit
CALORIES_MAX_NORMAL = 500     # kcal, normal calorie limit
MAX_TIPS = 10                  # maximum number of nutrition tips to return
MAX_WATER_INTAKE_L = 5.0       # liters per day ceiling

# Water intake factors
WATER_INTAKE_FACTOR = 0.035    # liters per kg of body weight
WATER_ACTIVITY_MULTIPLIERS = {
    "Sedentary": 1.0,
    "Light": 1.1,
    "Moderate": 1.2,
    "Active": 1.3,
    "Very Active": 1.4,
}

# Protein per kg mapping (used in nutrition_engine)
PROTEIN_PER_KG = {
    "default": 0.8,
    "kidney": 0.6,
    "obesity": 1.2,
    "diabetes": 1.0,
}

# ── Hybrid scoring weights ──────────────────────────────────────────────────
# These three weights define the target distribution when ALL components are
# available.  They MUST be numeric, non-negative, and sum to 1.0.
#
# NCF_SCORE_WEIGHT is applied ONLY when a trained NCF model is confirmed
# available at runtime.  When NCF is unavailable the two available weights
# are re-normalised dynamically so they still sum to 1.0:
#
#   effective_nutrition_w = NUTRITION_SCORE_WEIGHT
#                           / (NUTRITION_SCORE_WEIGHT + CONTENT_SCORE_WEIGHT)
#   effective_content_w   = CONTENT_SCORE_WEIGHT
#                           / (NUTRITION_SCORE_WEIGHT + CONTENT_SCORE_WEIGHT)
#
# Implemented in EnhancedNutritionRecommender._calculate_hybrid_score().
NUTRITION_SCORE_WEIGHT: float = 0.50   # nutrition-suitability component
CONTENT_SCORE_WEIGHT:   float = 0.30   # content-based (goal / nutritional profile fit)
NCF_SCORE_WEIGHT:       float = 0.20   # NCF predicted-rating (only when model trained)

# Legacy alias kept for backward-compatibility — do not use in new code.
SUITABILITY_SCORE_WEIGHT: float = NUTRITION_SCORE_WEIGHT

# ── Weight validation ────────────────────────────────────────────────────────
def _validate_hybrid_weights() -> None:
    """Raise ValueError if any hybrid weight is invalid."""
    weights = {
        "NUTRITION_SCORE_WEIGHT": NUTRITION_SCORE_WEIGHT,
        "CONTENT_SCORE_WEIGHT":   CONTENT_SCORE_WEIGHT,
        "NCF_SCORE_WEIGHT":       NCF_SCORE_WEIGHT,
    }
    for name, value in weights.items():
        if not isinstance(value, (int, float)):
            raise ValueError(f"config.{name} must be numeric, got {type(value)}")
        if value < 0:
            raise ValueError(f"config.{name} must be non-negative, got {value}")
    total = NUTRITION_SCORE_WEIGHT + CONTENT_SCORE_WEIGHT + NCF_SCORE_WEIGHT
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"Hybrid weights must sum to 1.0, got {total:.6f} "
            f"(NUTRITION={NUTRITION_SCORE_WEIGHT}, CONTENT={CONTENT_SCORE_WEIGHT}, "
            f"NCF={NCF_SCORE_WEIGHT})"
        )

_validate_hybrid_weights()

# ── Disease severity defaults ────────────────────────────────────────────────
# Used by the severity pipeline when severity cannot be determined from
# RiskResult.final_status (e.g. the value is None or unrecognised).
# Must be one of: "mild" | "moderate" | "severe"
# See backend/services/severity_rules.py for full severity rules.
SEVERITY_DEFAULT: str = "moderate"

# Minimum number of candidate foods that must remain after severity-based
# hard filtering.  If the pool drops below this, soft constraints are
# progressively relaxed (hard constraints are NEVER relaxed).
# See EnhancedNutritionRecommender._apply_severity_filters().
SEVERITY_MIN_POOL_SIZE: int = 30

# UI mappings (example placeholders)
LEVEL_CSS_CLASS = {
    "ok": "status-ok",
    "warning": "status-warning",
    "danger": "status-danger",
    "info": "status-info",
}
RISK_BADGE_CLASS = {
    "high": "pred-risk-high",
    "medium": "pred-risk-medium",
    "low": "pred-risk-low",
}
