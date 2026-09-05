"""
Regression tests for the fiber-scoring logic in
IntelligentNutritionRecommender._calculate_nutritional_score().

These tests guard against re-introduction of the duplicate-fiber bug where
Diabetes + Obesity was accidentally producing a fiber contribution of 25
(= 5g * 3 + 5g * 2) instead of the correct 15 (= max(3, 2) * 5g).

Run with:
    python test_fiber_scoring_regression.py

Style follows test_risk_regression.py: custom check() helper, printed
PASS / FAIL per assertion, final summary with exit code.
"""

import sys
import pandas as pd

# ---------------------------------------------------------------------------
# Minimal fixture — only the columns that _calculate_nutritional_score reads
# ---------------------------------------------------------------------------
_ZERO_ROW = pd.Series({
    "Free Sugar (g)":    0.0,
    "Carbohydrates (g)": 0.0,
    "Sodium (mg)":       0.0,
    "Protein (g)":       0.0,
    "Calories (kcal)":   0.0,
    "Fats (g)":          0.0,
    "Fibre (g)":         5.0,   # the one nutrient under test
})

FIBRE_G = 5.0   # kept as a named constant so the intent is obvious

# ---------------------------------------------------------------------------
# Shared helper (same pattern as test_risk_regression.py)
# ---------------------------------------------------------------------------
PASS_COUNT = 0
FAIL_COUNT = 0
FAILURES:  list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS_COUNT, FAIL_COUNT
    if condition:
        print(f"  [PASS] {name}")
        PASS_COUNT += 1
    else:
        msg = f"  [FAIL] {name}"
        if detail:
            msg += f"\n         {detail}"
        print(msg)
        FAIL_COUNT += 1
        FAILURES.append(name)


SEP = "=" * 65

# ---------------------------------------------------------------------------
# Set up the recommender once — avoids reloading the food DataFrame per test
# ---------------------------------------------------------------------------
from recommendation import IntelligentNutritionRecommender
recommender = IntelligentNutritionRecommender()


def _score(conditions: dict) -> float:
    """Return the nutritional score for _ZERO_ROW under the given conditions."""
    return recommender._calculate_nutritional_score(_ZERO_ROW, conditions)


# ---------------------------------------------------------------------------
# TEST 1 — Diabetes only: fiber weight 3  →  contribution = 5 * 3 = 15
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("  TEST 1 — Diabetes only  |  Fibre = 5 g  |  Expected contribution = 15")
print(SEP)

conds_diabetes = {"has_diabetes": True, "has_kidney_disease": False, "has_obesity": False}
score_diabetes = _score(conds_diabetes)

check(
    "Diabetes only: total score == 15.0",
    score_diabetes == 15.0,
    f"got {score_diabetes}",
)
check(
    "Diabetes only: score is positive (fiber rewarded)",
    score_diabetes > 0,
    f"got {score_diabetes}",
)

# ---------------------------------------------------------------------------
# TEST 2 — Obesity only: fiber weight 2  →  contribution = 5 * 2 = 10
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("  TEST 2 — Obesity only  |  Fibre = 5 g  |  Expected contribution = 10")
print(SEP)

conds_obesity = {"has_diabetes": False, "has_kidney_disease": False, "has_obesity": True}
score_obesity = _score(conds_obesity)

check(
    "Obesity only: total score == 10.0",
    score_obesity == 10.0,
    f"got {score_obesity}",
)
check(
    "Obesity only: score is less than Diabetes-only score (weight 2 < weight 3)",
    score_obesity < score_diabetes,
    f"obesity={score_obesity}, diabetes={score_diabetes}",
)

# ---------------------------------------------------------------------------
# TEST 3 — Diabetes + Obesity: max(3, 2) = 3  →  contribution = 5 * 3 = 15
#          Must NOT be 25 (old bug: 5*3 + 5*2 = 25)
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("  TEST 3 — Diabetes + Obesity  |  Fibre = 5 g  |  Expected = 15  (NOT 25)")
print(SEP)

conds_both = {"has_diabetes": True, "has_kidney_disease": False, "has_obesity": True}
score_both = _score(conds_both)

check(
    "Diabetes + Obesity: total score == 15.0  (max-weight, not additive)",
    score_both == 15.0,
    f"got {score_both}",
)
check(
    "Diabetes + Obesity: score is NOT 25.0  (duplicate-fiber bug absent)",
    score_both != 25.0,
    f"got {score_both} — duplicate-fiber bug has been re-introduced",
)
check(
    "Diabetes + Obesity: score equals Diabetes-only score (max wins)",
    score_both == score_diabetes,
    f"both={score_both}, diabetes_only={score_diabetes}",
)

# ---------------------------------------------------------------------------
# TEST 4 — Kidney disease only: no fiber term in current scoring logic
#          contribution = 0
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("  TEST 4 — Kidney only  |  Fibre = 5 g  |  Expected fiber contribution = 0")
print(SEP)

conds_kidney = {"has_diabetes": False, "has_kidney_disease": True, "has_obesity": False}
score_kidney = _score(conds_kidney)

check(
    "Kidney only: total score == 0.0  (no fiber weight in kidney path)",
    score_kidney == 0.0,
    f"got {score_kidney}",
)
check(
    "Kidney only: score is not positive from fiber",
    score_kidney <= 0.0,
    f"got {score_kidney}",
)

# ---------------------------------------------------------------------------
# TEST 5 — Normal diet (no disease): weight 1.5  →  contribution = 5 * 1.5 = 7.5
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("  TEST 5 — Normal diet  |  Fibre = 5 g  |  Expected contribution = 7.5")
print(SEP)

conds_normal = {"has_diabetes": False, "has_kidney_disease": False, "has_obesity": False}
score_normal = _score(conds_normal)

check(
    "Normal diet: total score == 7.5  (fiber weight 1.5)",
    score_normal == 7.5,
    f"got {score_normal}",
)
check(
    "Normal diet: score is less than Diabetes score (1.5 < 3)",
    score_normal < score_diabetes,
    f"normal={score_normal}, diabetes={score_diabetes}",
)
check(
    "Normal diet: score is less than Obesity score (1.5 < 2)",
    score_normal < score_obesity,
    f"normal={score_normal}, obesity={score_obesity}",
)

# ---------------------------------------------------------------------------
# TEST 6 — Cross-check: fiber contribution ordering
#          Expected: Diabetes(15) == Diabetes+Obesity(15) > Obesity(10) > Normal(7.5) > Kidney(0)
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("  TEST 6 — Ordering: fiber contributions must follow the correct hierarchy")
print(SEP)

check(
    "Ordering: Diabetes(15) >= Diabetes+Obesity(15)",
    score_diabetes >= score_both,
    f"diabetes={score_diabetes}, both={score_both}",
)
check(
    "Ordering: Diabetes+Obesity(15) > Obesity(10)",
    score_both > score_obesity,
    f"both={score_both}, obesity={score_obesity}",
)
check(
    "Ordering: Obesity(10) > Normal(7.5)",
    score_obesity > score_normal,
    f"obesity={score_obesity}, normal={score_normal}",
)
check(
    "Ordering: Normal(7.5) > Kidney(0)",
    score_normal > score_kidney,
    f"normal={score_normal}, kidney={score_kidney}",
)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print(f"  RESULT: {PASS_COUNT} passed,  {FAIL_COUNT} failed")
if FAILURES:
    print(f"  Failed tests: {FAILURES}")
print(SEP)

sys.exit(0 if FAIL_COUNT == 0 else 1)
