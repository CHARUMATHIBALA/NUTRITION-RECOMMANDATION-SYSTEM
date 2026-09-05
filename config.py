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

# ── NCF hybrid scoring weights ──────────────────────────────────────────────
# These three weights must sum to 1.0.
# NCF_SCORE_WEIGHT applies only when the trained NCF model is loaded;
# when falling back, NCF_SCORE_WEIGHT is treated as 0 and the remaining
# weight is redistributed to SUITABILITY_SCORE_WEIGHT so the maths still
# sums to 1.  See HybridRecommender._compute_hybrid_score() for usage.
SUITABILITY_SCORE_WEIGHT: float = 0.60   # nutrition-suitability component
CONTENT_SCORE_WEIGHT:     float = 0.20   # content-based component (rating normalised)
NCF_SCORE_WEIGHT:         float = 0.20   # NCF predicted-rating component

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
