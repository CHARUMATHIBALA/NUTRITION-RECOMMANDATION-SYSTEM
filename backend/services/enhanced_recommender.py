"""
Enhanced Nutrition Recommendation Engine
User-specific, severity-aware, calorie-targeted recommendations with real hybrid scoring.

Pipeline
────────
User Profile (incl. severity dict)
  → Disease Filtering        (hard thresholds from config.py, then severity-based)
  → Severity Hard Filtering  (severity_rules.py hard_constraints per disease tier)
  → Nutrition Score          (real arithmetic, severity multipliers applied)
  → Severity Suitability     (soft-constraint penalty from severity_rules.py)
  → Content Score            (nutritional-profile fit, already in [0,1])
  → NCF Score                (only if trained model exists; currently UNAVAILABLE)
  → Hybrid Score             (dynamically normalised weights)
  → Deterministic Ranking
  → Meal Construction
  → UI Display

Scoring components
──────────────────
Nutrition Score  (always available)
    Real arithmetic score from food nutritional values × user disease/profile.
    Severity multipliers from severity_rules.py are applied per disease tier,
    making the same food score differently at Mild vs Moderate vs Severe.
    Normalised to [0, 1] before combining.

Severity Suitability Score  (always available when disease is present)
    A [0, 1] penalty score from severity_rules.calculate_severity_suitability_score().
    Foods outside soft constraints are penalised proportionally.
    Foods violating hard constraints score 0 (already excluded by filtering,
    but this acts as a second safety net).
    Blended into the nutrition score before hybrid combination.

Content Score  (always available)
    Real nutritional-profile fit: calorie fit, protein density,
    fibre density, sugar penalty. In [0, 1].

NCF Score  (UNAVAILABLE — no trained model exists)
    Returns (0.0, False). Weights re-normalised over available components.
    NO fake/random/sine-hash score ever generated.

Weight source
─────────────
All hybrid weights read from config.py:
  config.NUTRITION_SCORE_WEIGHT = 0.50
  config.CONTENT_SCORE_WEIGHT   = 0.30
  config.NCF_SCORE_WEIGHT       = 0.20

Severity derivation
───────────────────
user_profile['severity'] is a dict  {disease_key: severity_level | None}
built by meal_planner.py from RiskResult.final_status values:
  'High Risk'     → 'severe'
  'Moderate Risk' → 'moderate'
  'Borderline' /
  'Model Flag'    → 'mild'
  'Low Risk'      → None   (no restrictions for that disease)
  missing/None    → config.SEVERITY_DEFAULT ('moderate')
"""

from __future__ import annotations

import logging
import hashlib
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from models import food_df
import config
from backend.services.severity_rules import (
    SEVERITY_RULES,
    DEFAULT_SEVERITY,
    MIN_POOL_SIZE,
    get_hard_constraints,
    get_soft_constraints,
    get_scoring_multipliers,
    get_required_minimums,
    calculate_severity_suitability_score,
    get_severity_explanation,
    get_recommendation_reason,
    _normalise_disease_key,
    _normalise_severity,
)

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
#  NCF availability check (unchanged from Step 4)
# ────────────────────────────────────────────────────────────────────────────

def _check_ncf_availability() -> bool:
    """Return True only when a *trained* NCF model is confirmed available."""
    try:
        import os
        from ncf_integration.models.ncf_model import NCFModel  # noqa: F401
        for path in [
            "ncf_integration/models/ncf_model.pt",
            "ncf_integration/models/ncf_model.pkl",
            "ncf_model.pt",
            "ncf_model.pkl",
        ]:
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                logger.info("NCF model found at %s — NCF scoring active.", path)
                return True
        logger.info("NCF model weights not found. NCF scoring is UNAVAILABLE.")
        return False
    except Exception as exc:
        logger.info("NCF package not importable (%s). NCF scoring is UNAVAILABLE.", exc)
        return False


_NCF_AVAILABLE: bool = _check_ncf_availability()


# ────────────────────────────────────────────────────────────────────────────
#  Weight helpers
# ────────────────────────────────────────────────────────────────────────────

def _load_and_validate_weights() -> Tuple[float, float, float]:
    n_w   = config.NUTRITION_SCORE_WEIGHT
    c_w   = config.CONTENT_SCORE_WEIGHT
    ncf_w = config.NCF_SCORE_WEIGHT
    for name, val in [
        ("NUTRITION_SCORE_WEIGHT", n_w),
        ("CONTENT_SCORE_WEIGHT",   c_w),
        ("NCF_SCORE_WEIGHT",       ncf_w),
    ]:
        if not isinstance(val, (int, float)):
            raise ValueError(f"config.{name} must be numeric, got {type(val)}")
        if val < 0:
            raise ValueError(f"config.{name} must be non-negative, got {val}")
    total = n_w + c_w + ncf_w
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Hybrid weights must sum to 1.0, got {total:.6f}")
    return float(n_w), float(c_w), float(ncf_w)


# ────────────────────────────────────────────────────────────────────────────
#  Normalisation utilities
# ────────────────────────────────────────────────────────────────────────────

_NUTRITION_SCORE_MAX: float = 80.0


def _safe_float(value, default: float = 0.0) -> float:
    try:
        v = float(value)
        return default if (np.isnan(v) or np.isinf(v)) else v
    except (TypeError, ValueError):
        return default


def _normalise_to_unit(score: float, max_val: float) -> float:
    if max_val <= 0:
        return 0.0
    return max(0.0, min(1.0, score / max_val))


# ────────────────────────────────────────────────────────────────────────────
#  Severity helpers (internal)
# ────────────────────────────────────────────────────────────────────────────

def _extract_severity_dict(user_profile: Dict) -> Dict[str, Optional[str]]:
    """
    Extract and normalise the severity dict from user_profile.

    user_profile['severity'] is expected to be a dict produced by
    severity_rules.build_severity_dict(), e.g.:
        {'diabetes': 'severe', 'kidney_disease': None, 'obesity': 'moderate'}

    If the key is absent or the value is not a dict, returns {}.
    Individual missing/None entries are left as None (no restrictions).
    """
    raw = user_profile.get('severity')
    if not isinstance(raw, dict):
        return {}
    # normalise keys and values
    result: Dict[str, Optional[str]] = {}
    for k, v in raw.items():
        nk = _normalise_disease_key(str(k))
        result[nk] = _normalise_severity(str(v)) if v is not None else None
    return result


def _get_severity_for_disease(disease_key: str, severity_dict: Dict) -> Optional[str]:
    """
    Return severity for a specific disease key from the severity dict.

    If the key is present but its value is None → return None (healthy/no disease).
    If the key is missing entirely → return config.SEVERITY_DEFAULT (safe fallback,
    never assume Severe for an unknown disease).
    """
    norm_key = _normalise_disease_key(disease_key)
    if norm_key not in severity_dict:
        return config.SEVERITY_DEFAULT
    return severity_dict[norm_key]   # could be None for "no disease"


# ────────────────────────────────────────────────────────────────────────────
#  Main recommender class
# ────────────────────────────────────────────────────────────────────────────

class EnhancedNutritionRecommender:
    """
    Recommendation engine with real hybrid scoring and disease-severity awareness.

    Scoring architecture
    --------------------
    nutrition_score_raw  = base nutrition arithmetic
                           + severity-specific multiplier adjustments
                           + required-minimum bonuses
    nutrition_norm       = clamp(nutrition_score_raw / _NUTRITION_SCORE_MAX, 0, 1)
    severity_suit        = calculate_severity_suitability_score()   ∈ [0, 1]

    # Blend severity suitability into nutrition norm (70/30 split)
    nutrition_blended    = 0.70 × nutrition_norm + 0.30 × severity_suit

    content_score        = real macro/calorie-profile fit            ∈ [0, 1]
    ncf_score            = 0.0 (unavailable)

    hybrid = w_n × nutrition_blended + w_c × content_score + w_ncf × ncf_score

    When NCF is unavailable:
        w_n_eff = w_n / (w_n + w_c)   = 0.6250
        w_c_eff = w_c / (w_n + w_c)   = 0.3750

    All component scores are attached to each recommended food under
    '_score_breakdown' for downstream explanation.
    """

    MEAL_CALORIE_DISTRIBUTION = {
        'Breakfast': 0.25,
        'Lunch':     0.35,
        'Snack':     0.15,
        'Dinner':    0.25,
    }

    def __init__(self):
        self.df = food_df.copy()
        self._w_nutrition, self._w_content, self._w_ncf = _load_and_validate_weights()
        self._ncf_available: bool = _NCF_AVAILABLE
        self._eff_w_nutrition, self._eff_w_content, self._eff_w_ncf = (
            self._compute_effective_weights()
        )
        if not self._ncf_available:
            logger.info(
                "NCF UNAVAILABLE. Effective weights → Nutrition=%.4f, Content=%.4f "
                "(NCF weight %.2f redistributed proportionally).",
                self._eff_w_nutrition, self._eff_w_content, self._w_ncf,
            )

    # ── Weight helpers ────────────────────────────────────────────────────

    def _compute_effective_weights(self) -> Tuple[float, float, float]:
        if self._ncf_available:
            return self._w_nutrition, self._w_content, self._w_ncf
        available = self._w_nutrition + self._w_content
        if available <= 0:
            return 0.5, 0.5, 0.0
        return self._w_nutrition / available, self._w_content / available, 0.0

    # ── Disease condition helpers ─────────────────────────────────────────

    def _get_disease_conditions(self, diseases: List[str]) -> Dict[str, bool]:
        conditions = {
            'has_diabetes':      False,
            'has_kidney_disease': False,
            'has_obesity':       False,
        }
        for d in diseases:
            dl = str(d).lower()
            if 'diabetes' in dl:
                conditions['has_diabetes'] = True
            elif 'kidney' in dl:
                conditions['has_kidney_disease'] = True
            elif 'obesity' in dl or 'overweight' in dl:
                conditions['has_obesity'] = True
        return conditions

    # ── Stage 1: disease-level hard filter (config thresholds) ───────────

    def _apply_disease_filters(
        self, df: pd.DataFrame, diseases: List[str]
    ) -> pd.DataFrame:
        """
        Apply the base disease hard filters defined in config.py.
        These are the same thresholds used before severity was introduced.
        They represent the moderate-severity baseline.
        """
        conditions = self._get_disease_conditions(diseases)
        fdf = df.copy()

        if conditions['has_diabetes']:
            fdf = fdf[
                (fdf["Free Sugar (g)"] <= config.DIABETES_MAX_SUGAR) &
                (fdf["Calories (kcal)"] <= config.DIABETES_MAX_CALORIES)
            ]
        if conditions['has_kidney_disease']:
            fdf = fdf[
                (fdf["Sodium (mg)"]  <= config.KIDNEY_MAX_SODIUM) &
                (fdf["Protein (g)"]  <= config.KIDNEY_MAX_PROTEIN)
            ]
        if conditions['has_obesity']:
            fdf = fdf[
                (fdf["Calories (kcal)"] <= config.OBESITY_MAX_CALORIES) &
                (fdf["Fats (g)"]        <= config.OBESITY_MAX_FAT)
            ]
        if not any(conditions.values()):
            fdf = fdf[
                (fdf["Calories (kcal)"]  <= config.CALORIES_MAX_NORMAL) &
                (fdf["Free Sugar (g)"]   <= config.FREE_SUGAR_MAX_NORMAL)
            ]
        return fdf

    # ── Stage 2: severity-based hard filter ──────────────────────────────

    def _apply_severity_filters(
        self,
        df: pd.DataFrame,
        diseases: List[str],
        severity_dict: Dict[str, Optional[str]],
    ) -> pd.DataFrame:
        """
        Apply per-disease severity hard constraints on top of the base filters.

        For each active disease, looks up severity_rules.get_hard_constraints()
        and filters the DataFrame row-by-row.

        Soft-constraint progressive relaxation
        ──────────────────────────────────────
        If after applying ALL hard constraints the pool drops below
        config.SEVERITY_MIN_POOL_SIZE, the method does NOT relax hard
        constraints.  Instead it logs a warning and returns whatever pool
        is left — even if small — to preserve safety.

        If NO diseases are active (healthy user), df is returned unchanged.
        """
        if df.empty:
            return df

        conditions = self._get_disease_conditions(diseases)
        if not any(conditions.values()):
            return df   # healthy user — no severity filtering

        fdf = df.copy()
        active_disease_keys = []

        if conditions['has_diabetes']:
            active_disease_keys.append('diabetes')
        if conditions['has_kidney_disease']:
            active_disease_keys.append('kidney_disease')
        if conditions['has_obesity']:
            active_disease_keys.append('obesity')

        for disease_key in active_disease_keys:
            severity = _get_severity_for_disease(disease_key, severity_dict)
            if severity is None:
                # disease predicted 'Low Risk' → no additional severity filtering
                continue

            hard = get_hard_constraints(disease_key, severity)
            if not hard:
                continue

            before = len(fdf)
            for col, bounds in hard.items():
                if col not in fdf.columns:
                    logger.warning(
                        "Severity hard constraint column '%s' not in dataset.", col
                    )
                    continue
                if "max" in bounds:
                    fdf = fdf[fdf[col] <= bounds["max"]]
                if "min" in bounds:
                    fdf = fdf[fdf[col] >= bounds["min"]]

            after = len(fdf)
            logger.debug(
                "Severity hard filter [%s/%s]: %d → %d foods",
                disease_key, severity, before, after,
            )

        if len(fdf) < config.SEVERITY_MIN_POOL_SIZE:
            logger.warning(
                "Severity hard filtering left only %d foods (min=%d). "
                "Hard constraints are preserved — no soft relaxation applied.",
                len(fdf), config.SEVERITY_MIN_POOL_SIZE,
            )

        return fdf

    # ── Nutrition score (severity-aware) ──────────────────────────────────

    def _calculate_nutrition_score(
        self, row: pd.Series, user_profile: Dict
    ) -> float:
        """
        Severity-aware arithmetic nutrition score.

        Base penalties/bonuses are identical to the pre-severity version.
        Severity multipliers from severity_rules.py are then applied on top,
        making the same food score differently at Mild vs Moderate vs Severe.

        Example (diabetes, sugar=4g):
          Mild:    score contribution = 4 × 3.0 (mild sugar_penalty) = -12
          Moderate: score contribution = 4 × 4.0 (mod  sugar_penalty) = -16
          Severe:  score contribution = 4 × 6.0 (sev  sugar_penalty) = -24

        Returns a non-negative raw float (clamped to ≥ 0).
        """
        try:
            score: float = 0.0
            diseases   = user_profile.get('diseases', [])
            conditions = self._get_disease_conditions(diseases)
            sev_dict   = _extract_severity_dict(user_profile)

            # ── Nutritional values ────────────────────────────────────
            fiber    = _safe_float(row.get('Fibre (g)',          0))
            protein  = _safe_float(row.get('Protein (g)',        0))
            sugar    = _safe_float(row.get('Free Sugar (g)',     0))
            calories = _safe_float(row.get('Calories (kcal)',    0))
            sodium   = _safe_float(row.get('Sodium (mg)',        0))
            carbs    = _safe_float(row.get('Carbohydrates (g)',  0))
            fats     = _safe_float(row.get('Fats (g)',           0))
            vit_c    = _safe_float(row.get('Vitamin C (mg)',     0))
            calcium  = _safe_float(row.get('Calcium (mg)',       0))
            iron     = _safe_float(row.get('Iron (mg)',          0))

            # ── Demographic adjustments (unchanged from Step 4) ───────
            age = int(_safe_float(user_profile.get('age', 30)))
            if age > 65:
                score += protein * 1.2
                score += calcium * 0.01
            elif age < 30:
                score += min(calories / 100.0, 5.0) * 0.5

            gender = str(user_profile.get('gender', 'Male')).lower()
            if gender == 'male':
                score += protein * 1.1
            else:
                score += iron * 2.0

            bmi = _safe_float(user_profile.get('bmi', 22.0))
            if bmi > 25.0:
                score -= calories * 0.03
                score += fiber   * 2.0
            elif bmi < 18.5:
                score += calories * 0.02
                score += protein  * 1.5

            activity = str(user_profile.get('activity_level', 'Sedentary'))
            act_mult = _safe_float(
                config.WATER_ACTIVITY_MULTIPLIERS.get(activity, 1.0), 1.0
            )
            if act_mult > 1.2:
                score += calories * 0.015
                score += carbs    * 0.1

            # ── Disease + severity scoring ────────────────────────────
            # For each active disease, load the severity-specific multipliers.
            # If severity is None (Low Risk), fall back to base multipliers.

            if conditions['has_diabetes']:
                sev = _get_severity_for_disease('diabetes', sev_dict)
                mults = get_scoring_multipliers('diabetes', sev)

                sugar_pen  = mults.get('sugar_penalty',  4.0)
                carb_pen   = mults.get('carb_penalty',   0.3)
                fiber_bon  = mults.get('fiber_bonus',    3.0)

                score -= sugar  * sugar_pen
                score += fiber  * fiber_bon
                score -= carbs  * carb_pen

                # Required minimum bonus
                for col, min_val in get_required_minimums('diabetes', sev).items():
                    col_val = _safe_float(row.get(col, 0))
                    if col_val >= min_val:
                        score += 2.0   # explicit bonus for meeting minimum

            if conditions['has_kidney_disease']:
                sev = _get_severity_for_disease('kidney_disease', sev_dict)
                mults = get_scoring_multipliers('kidney_disease', sev)

                sodium_pen  = mults.get('sodium_penalty',  0.15)
                protein_pen = mults.get('protein_penalty', 0.6)

                score -= sodium   * sodium_pen
                score -= protein  * protein_pen

            if conditions['has_obesity']:
                sev = _get_severity_for_disease('obesity', sev_dict)
                mults = get_scoring_multipliers('obesity', sev)

                cal_pen    = mults.get('calorie_penalty', 0.04)
                fat_pen    = mults.get('fat_penalty',     0.6)
                prot_bon   = mults.get('protein_bonus',   2.0)
                fib_bon    = mults.get('fiber_bonus',     2.5)

                score -= calories * cal_pen
                score -= fats     * fat_pen
                score += protein  * prot_bon
                score += fiber    * fib_bon

                for col, min_val in get_required_minimums('obesity', sev).items():
                    col_val = _safe_float(row.get(col, 0))
                    if col_val >= min_val:
                        score += 2.0

            # ── General nutrition benefits (always applied) ───────────
            score += fiber   * 1.5
            score += protein * 1.2
            score -= sugar   * 1.5

            # ── Micronutrient bonuses ─────────────────────────────────
            score += vit_c   * 0.1
            score += calcium * 0.01
            score += iron    * 0.5

            return max(0.0, score)
        except Exception:
            logger.exception(
                "Nutrition score calculation failed for %s", row.get('Dish Name')
            )
            return 0.0

    # ── Content score (unchanged from Step 4) ────────────────────────────

    def _calculate_content_score(
        self, row: pd.Series, user_profile: Dict
    ) -> float:
        """
        Real content-based nutritional-profile fit score (Step 4 implementation).
        Four sub-scores: calorie fit (40%), protein density (25%),
        fibre density (20%), sugar inverse (15%).
        Returns float in [0, 1].
        """
        try:
            calories   = _safe_float(row.get('Calories (kcal)',  0))
            protein    = _safe_float(row.get('Protein (g)',      0))
            fiber      = _safe_float(row.get('Fibre (g)',        0))
            free_sugar = _safe_float(row.get('Free Sugar (g)',   0))

            daily_cal   = _safe_float(user_profile.get('daily_calories', 2000), 2000)
            meal_target = daily_cal * 0.25
            if meal_target > 0 and calories > 0:
                ratio = calories / meal_target
                calorie_fit = max(0.0, 1.0 - abs(ratio - 1.0))
            else:
                calorie_fit = 0.5

            protein_density = (
                _normalise_to_unit(protein / calories, 0.12)
                if calories > 0 else 0.0
            )
            fibre_density = (
                _normalise_to_unit(fiber / calories, 0.06)
                if calories > 0 else 0.0
            )

            sugar_ceiling = _safe_float(config.FREE_SUGAR_MAX_NORMAL, 15.0)
            sugar_score = (
                max(0.0, 1.0 - (free_sugar / sugar_ceiling))
                if sugar_ceiling > 0 else 1.0
            )

            return max(0.0, min(1.0,
                calorie_fit     * 0.40 +
                protein_density * 0.25 +
                fibre_density   * 0.20 +
                sugar_score     * 0.15
            ))
        except Exception:
            logger.exception(
                "Content score calculation failed for %s", row.get('Dish Name')
            )
            return 0.0

    # ── NCF score (unchanged from Step 4 — unavailable) ──────────────────

    def _calculate_ncf_score(
        self, row: pd.Series, user_profile: Dict
    ) -> Tuple[float, bool]:
        """Returns (0.0, False) — NCF is currently unavailable."""
        if not self._ncf_available:
            return 0.0, False
        try:
            from ncf_integration.models.ncf_model import NCFModel
            dish_name = str(row.get('Dish Name', ''))
            matches   = self.df[self.df['Dish Name'] == dish_name]
            food_idx  = int(matches.index[0]) if not matches.empty else 0
            user_id   = int(_safe_float(user_profile.get('user_id', 0)))
            model     = NCFModel.__new__(NCFModel)
            raw_scores = model.predict(user_id, [food_idx])
            raw        = _safe_float(raw_scores[0] if raw_scores else 0.0)
            return _normalise_to_unit(raw, 5.0), True
        except Exception as exc:
            logger.warning("NCF prediction failed: %s.", exc)
            return 0.0, False

    # ── Hybrid score (severity-aware) ─────────────────────────────────────

    def _calculate_hybrid_score(
        self, row: pd.Series, user_profile: Dict
    ) -> Tuple[float, Dict]:
        """
        Compute the final hybrid score with severity-aware components.

        Pipeline
        ────────
        1. nutrition_raw   = _calculate_nutrition_score (severity multipliers applied)
        2. nutrition_norm  = clamp(nutrition_raw / 80, 0, 1)
        3. severity_suit   = calculate_severity_suitability_score()  per active disease
           combined_suit   = min over all active diseases (strictest wins)
        4. nutrition_blended = 0.70 × nutrition_norm + 0.30 × combined_suit
        5. content_score   = _calculate_content_score()
        6. hybrid = w_n × nutrition_blended + w_c × content_score + w_ncf × ncf

        Score breakdown keys returned
        ──────────────────────────────
        nutrition_raw, nutrition_norm, severity_suitability, severity_dict,
        nutrition_blended, content_score, ncf_score, ncf_available,
        w_nutrition, w_content, w_ncf, hybrid_score,
        severity_reason (plain-text explanation per disease)
        """
        sev_dict   = _extract_severity_dict(user_profile)
        diseases   = user_profile.get('diseases', [])
        if isinstance(diseases, str):
            diseases = [diseases]
        conditions = self._get_disease_conditions(diseases)

        # ── 1. Nutrition score (severity-multiplied) ──────────────────
        nutrition_raw  = self._calculate_nutrition_score(row, user_profile)
        nutrition_norm = _normalise_to_unit(nutrition_raw, _NUTRITION_SCORE_MAX)

        # ── 2. Severity suitability score ─────────────────────────────
        # Compute per active disease, then take the minimum (strictest wins
        # when multiple diseases are active — same food must satisfy all).
        suitability_scores: List[float] = []
        severity_reasons:   Dict[str, str] = {}

        active_disease_map = {
            'diabetes':       conditions['has_diabetes'],
            'kidney_disease': conditions['has_kidney_disease'],
            'obesity':        conditions['has_obesity'],
        }

        for dk, is_active in active_disease_map.items():
            if not is_active:
                continue
            sev = _get_severity_for_disease(dk, sev_dict)
            suit = calculate_severity_suitability_score(row, dk, sev)
            suitability_scores.append(suit)
            severity_reasons[dk] = get_recommendation_reason(row, dk, sev)

        if suitability_scores:
            combined_suit = min(suitability_scores)  # strictest active disease wins
        else:
            combined_suit = 0.5   # healthy user — neutral

        # ── 3. Blend nutrition + severity suitability ─────────────────
        nutrition_blended = 0.70 * nutrition_norm + 0.30 * combined_suit

        # ── 4. Content + NCF ──────────────────────────────────────────
        content_score          = self._calculate_content_score(row, user_profile)
        ncf_score, ncf_used    = self._calculate_ncf_score(row, user_profile)

        # ── 5. Hybrid combination ─────────────────────────────────────
        w_n   = self._eff_w_nutrition
        w_c   = self._eff_w_content
        w_ncf = self._eff_w_ncf

        hybrid = max(0.0, min(1.0,
            nutrition_blended * w_n +
            content_score     * w_c +
            ncf_score         * w_ncf
        ))

        # ── Severity explanation strings ──────────────────────────────
        sev_explanations: Dict[str, str] = {}
        for dk, is_active in active_disease_map.items():
            if not is_active:
                continue
            sev = _get_severity_for_disease(dk, sev_dict)
            sev_explanations[dk] = get_severity_explanation(dk, sev)

        breakdown: Dict = {
            'nutrition_raw':         round(nutrition_raw,   4),
            'nutrition_norm':        round(nutrition_norm,  4),
            'severity_suitability':  round(combined_suit,   4),
            'severity_dict':         {k: v for k, v in sev_dict.items()},
            'nutrition_blended':     round(nutrition_blended, 4),
            'content_score':         round(content_score,   4),
            'ncf_score':             round(ncf_score, 4) if ncf_used else None,
            'ncf_available':         ncf_used,
            'w_nutrition':           round(w_n,   4),
            'w_content':             round(w_c,   4),
            'w_ncf':                 round(w_ncf, 4),
            'hybrid_score':          round(hybrid, 4),
            'severity_reasons':      severity_reasons,
            'severity_explanations': sev_explanations,
        }
        return hybrid, breakdown

    # ── Diversity penalty ─────────────────────────────────────────────────

    def _apply_diversity_penalty(
        self, df: pd.DataFrame, selected_foods: Set[str]
    ) -> pd.DataFrame:
        df = df.copy()
        df['diversity_penalty'] = df['Dish Name'].apply(
            lambda x: 0.3 if x in selected_foods else 0.0
        )
        df['adjusted_score'] = df['hybrid_score'] - df['diversity_penalty']
        return df.sort_values('adjusted_score', ascending=False)

    # ── Top-N selection (deterministic) ──────────────────────────────────

    def _select_top_foods(self, df: pd.DataFrame, top_n: int) -> List[pd.Series]:
        if df.empty:
            return []
        score_col = 'adjusted_score' if 'adjusted_score' in df.columns else 'hybrid_score'
        top = df.sort_values(score_col, ascending=False).head(top_n)
        return [top.loc[idx] for idx in top.index]

    # ── Score breakdown attachment ────────────────────────────────────────

    @staticmethod
    def _attach_breakdown(food: pd.Series, breakdown: Dict) -> pd.Series:
        food = food.copy()
        food['_score_breakdown'] = breakdown
        return food

    # ── User profile hash ─────────────────────────────────────────────────

    def _get_user_hash(self, user_profile: Dict) -> str:
        profile_str = str([(k, str(v)) for k, v in sorted(user_profile.items())])
        return hashlib.md5(profile_str.encode()).hexdigest()

    # ── Main pipeline ─────────────────────────────────────────────────────

    def generate_meal_plan(self, user_profile: Dict) -> Dict:
        """
        Generate a complete severity-aware meal plan.

        Steps
        ─────
        1. Normalise disease list.
        2. Apply base disease hard filters (config.py thresholds).
        3. Apply severity hard filters (severity_rules.py per disease/tier).
           If pool is too small, log warning but keep remaining foods.
        4. Compute hybrid score (with severity) for every candidate.
        5. For each meal slot:
             a. Filter by MealType.
             b. Apply diversity penalty.
             c. Select top-N deterministically.
             d. Attach score breakdowns.
        6. Aggregate nutrition summary, severity summary, meal explanations.

        Returns the standard dict + 'severity_info' key with per-disease
        severity levels and explanations.
        """
        diseases = user_profile.get('diseases', [])
        if isinstance(diseases, str):
            diseases = [diseases]

        sev_dict = _extract_severity_dict(user_profile)

        # ── Stage 1: base disease filters ────────────────────────────
        filtered_df = self._apply_disease_filters(self.df, diseases)

        if filtered_df.empty:
            logger.warning(
                "No foods survived base disease filters for diseases=%s", diseases
            )
            return self._create_fallback_meal_plan(user_profile, sev_dict)

        # ── Stage 2: severity hard filters ───────────────────────────
        filtered_df = self._apply_severity_filters(filtered_df, diseases, sev_dict)

        if filtered_df.empty:
            logger.warning(
                "No foods survived severity filters "
                "(diseases=%s, severity=%s)", diseases, sev_dict
            )
            return self._create_fallback_meal_plan(user_profile, sev_dict)

        # ── Stage 3: score all candidates ────────────────────────────
        filtered_df = filtered_df.copy()
        hybrid_scores, breakdowns = [], []
        for _, row in filtered_df.iterrows():
            h, bd = self._calculate_hybrid_score(row, user_profile)
            hybrid_scores.append(h)
            breakdowns.append(bd)

        filtered_df['hybrid_score']     = hybrid_scores
        filtered_df['_score_breakdown'] = breakdowns

        # ── Stage 4: build meal plan ──────────────────────────────────
        meal_plan:      Dict[str, List] = {}
        selected_foods: Set[str]        = set()
        total_calories: float           = 0.0
        daily_target = _safe_float(user_profile.get('daily_calories', 2000), 2000)

        for meal_type in ['Breakfast', 'Lunch', 'Snack', 'Dinner']:
            meal_lower = meal_type.lower()
            meal_df = filtered_df[
                filtered_df["MealType"].str.lower().str.contains(
                    meal_lower, na=False
                )
            ]
            if meal_df.empty:
                meal_plan[meal_type] = []
                continue

            meal_df  = self._apply_diversity_penalty(meal_df, selected_foods)
            num_foods = 3 if meal_type in ['Lunch', 'Dinner'] else 2
            selected  = self._select_top_foods(meal_df, num_foods)

            annotated = []
            for food in selected:
                bd   = food.get('_score_breakdown', {})
                food = self._attach_breakdown(food, bd)
                annotated.append(food)
                selected_foods.add(food['Dish Name'])
                total_calories += _safe_float(food.get('Calories (kcal)', 0))

            meal_plan[meal_type] = annotated

        nutrition_summary = self._calculate_nutrition_summary(meal_plan)
        meal_explanations = self._generate_meal_explanations(
            meal_plan, user_profile, sev_dict
        )
        severity_info = self._build_severity_info(diseases, sev_dict)

        return {
            'meal_plan':          meal_plan,
            'total_calories':     round(total_calories, 2),
            'target_calories':    daily_target,
            'calorie_difference': round(total_calories - daily_target, 2),
            'nutrition_summary':  nutrition_summary,
            'meal_explanations':  meal_explanations,
            'severity_info':      severity_info,
            'scoring_info': {
                'ncf_available':    self._ncf_available,
                'w_nutrition':      round(self._eff_w_nutrition, 4),
                'w_content':        round(self._eff_w_content,   4),
                'w_ncf':            round(self._eff_w_ncf,       4),
                'config_weights': {
                    'NUTRITION_SCORE_WEIGHT': self._w_nutrition,
                    'CONTENT_SCORE_WEIGHT':   self._w_content,
                    'NCF_SCORE_WEIGHT':       self._w_ncf,
                },
            },
            'debug_info': {
                'filtered_foods_count': len(filtered_df),
                'selected_foods':       list(selected_foods),
                'user_hash':            self._get_user_hash(user_profile),
            },
        }

    # ── Severity info summary ─────────────────────────────────────────────

    def _build_severity_info(
        self, diseases: List[str], sev_dict: Dict
    ) -> Dict:
        """
        Build the 'severity_info' block attached to every meal plan result.

        Contains per-disease severity level and plain-language explanation.
        """
        conditions = self._get_disease_conditions(diseases)
        info: Dict = {}

        disease_map = {
            'diabetes':       conditions['has_diabetes'],
            'kidney_disease': conditions['has_kidney_disease'],
            'obesity':        conditions['has_obesity'],
        }

        for dk, is_active in disease_map.items():
            if not is_active:
                continue
            sev = _get_severity_for_disease(dk, sev_dict)
            info[dk] = {
                'severity':    sev,
                'explanation': get_severity_explanation(dk, sev),
            }

        if not info:
            info['status'] = 'No active diseases — general dietary recommendations applied.'

        return info

    # ── Nutrition summary ─────────────────────────────────────────────────

    def _calculate_nutrition_summary(self, meal_plan: Dict) -> Dict:
        totals = {'protein': 0.0, 'carbohydrates': 0.0, 'fat': 0.0, 'fiber': 0.0}
        for foods in meal_plan.values():
            for food in foods:
                if isinstance(food, pd.Series):
                    totals['protein']       += _safe_float(food.get('Protein (g)',       0))
                    totals['carbohydrates'] += _safe_float(food.get('Carbohydrates (g)', 0))
                    totals['fat']           += _safe_float(food.get('Fats (g)',          0))
                    totals['fiber']         += _safe_float(food.get('Fibre (g)',         0))
        return {k: round(v, 1) for k, v in totals.items()}

    # ── Meal explanations (severity-aware) ───────────────────────────────

    def _generate_meal_explanations(
        self, meal_plan: Dict, user_profile: Dict, sev_dict: Dict
    ) -> Dict:
        """Generate plain-language explanations for each meal slot."""
        explanations = {}
        diseases   = user_profile.get('diseases', [])
        conditions = self._get_disease_conditions(diseases)

        for meal_type, foods in meal_plan.items():
            if not foods:
                explanations[meal_type] = "No suitable foods available."
                continue

            reasons: List[str] = []
            total_fiber   = sum(
                _safe_float(f.get('Fibre (g)',   0)) for f in foods
                if isinstance(f, pd.Series)
            )
            total_protein = sum(
                _safe_float(f.get('Protein (g)', 0)) for f in foods
                if isinstance(f, pd.Series)
            )

            if total_fiber   >= 5:  reasons.append("High-fibre meal")
            if total_protein >= 10: reasons.append("Protein-rich")

            # Severity-specific labels
            if conditions['has_diabetes']:
                sev = _get_severity_for_disease('diabetes', sev_dict)
                if sev == 'severe':
                    reasons.append("Very-low-sugar (severe diabetes)")
                elif sev == 'moderate':
                    reasons.append("Controlled-carbohydrate")
                else:
                    reasons.append("Low-sugar")

            if conditions['has_obesity']:
                sev = _get_severity_for_disease('obesity', sev_dict)
                if sev == 'severe':
                    reasons.append("Very-low-calorie (severe obesity)")
                elif sev == 'moderate':
                    reasons.append("Calorie-conscious")
                else:
                    reasons.append("Calorie-managed")

            if conditions['has_kidney_disease']:
                sev = _get_severity_for_disease('kidney_disease', sev_dict)
                if sev == 'severe':
                    reasons.append("Very-low-sodium (severe kidney disease)")
                elif sev == 'moderate':
                    reasons.append("Kidney-friendly")
                else:
                    reasons.append("Low-sodium")

            if not reasons:
                reasons.append("Balanced nutrition")

            explanations[meal_type] = (
                f"{', '.join(reasons)} suitable for your health profile."
            )

        return explanations

    # ── Fallback meal plan ────────────────────────────────────────────────

    def _create_fallback_meal_plan(
        self, user_profile: Dict, sev_dict: Optional[Dict] = None
    ) -> Dict:
        """Return an empty meal plan with severity info when filters leave no foods."""
        empty_meals = ['Breakfast', 'Lunch', 'Snack', 'Dinner']
        diseases = user_profile.get('diseases', [])
        if isinstance(diseases, str):
            diseases = [diseases]
        sev_dict = sev_dict or _extract_severity_dict(user_profile)
        severity_info = self._build_severity_info(diseases, sev_dict)

        return {
            'meal_plan':          {m: [] for m in empty_meals},
            'total_calories':     0.0,
            'target_calories':    _safe_float(user_profile.get('daily_calories', 2000), 2000),
            'calorie_difference': -_safe_float(user_profile.get('daily_calories', 2000), 2000),
            'nutrition_summary':  {'protein': 0, 'carbohydrates': 0, 'fat': 0, 'fiber': 0},
            'meal_explanations':  {m: "No suitable foods available" for m in empty_meals},
            'severity_info':      severity_info,
            'scoring_info': {
                'ncf_available': self._ncf_available,
                'w_nutrition':   round(self._eff_w_nutrition, 4),
                'w_content':     round(self._eff_w_content,   4),
                'w_ncf':         round(self._eff_w_ncf,       4),
            },
            'debug_info': {'error': 'No foods passed disease + severity filters'},
        }

    # ── Swap alternatives ─────────────────────────────────────────────────

    def get_swap_alternatives(
        self,
        current_food: str,
        meal_type: str,
        user_profile: Dict,
        limit: int = 3,
    ) -> List[pd.Series]:
        """
        Return the top *limit* alternative foods for swapping *current_food*.
        Uses the full severity-aware scoring pipeline.
        """
        diseases = user_profile.get('diseases', [])
        if isinstance(diseases, str):
            diseases = [diseases]
        sev_dict = _extract_severity_dict(user_profile)

        filtered_df = self._apply_disease_filters(self.df, diseases)
        filtered_df = self._apply_severity_filters(filtered_df, diseases, sev_dict)

        meal_lower = meal_type.lower()
        meal_df = filtered_df[
            filtered_df["MealType"].str.lower().str.contains(meal_lower, na=False)
        ]
        meal_df = meal_df[meal_df['Dish Name'] != current_food]

        if meal_df.empty:
            return []

        meal_df = meal_df.copy()
        hybrid_scores, breakdowns = [], []
        for _, row in meal_df.iterrows():
            h, bd = self._calculate_hybrid_score(row, user_profile)
            hybrid_scores.append(h)
            breakdowns.append(bd)

        meal_df['hybrid_score']     = hybrid_scores
        meal_df['_score_breakdown'] = breakdowns
        meal_df['adjusted_score']   = meal_df['hybrid_score']

        return self._select_top_foods(meal_df, limit)


# ────────────────────────────────────────────────────────────────────────────
#  Convenience entry-point (backward-compatible)
# ────────────────────────────────────────────────────────────────────────────

def generate_enhanced_recommendations(user_profile: Dict) -> Dict:
    """
    Convenience wrapper used by meal_planner.py and legacy callers.
    Passes user_profile (including 'severity' key if present) directly to
    EnhancedNutritionRecommender.generate_meal_plan().
    """
    recommender = EnhancedNutritionRecommender()
    return recommender.generate_meal_plan(user_profile)
