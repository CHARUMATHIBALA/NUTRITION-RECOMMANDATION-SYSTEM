"""
Hybrid Scoring Validation Suite
================================
Tests all four required scenarios:

  Scenario 1 – Nutrition + Content available, NCF unavailable
  Scenario 2 – Only Nutrition score available (content forced to zero)
  Scenario 3 – Invalid / missing values (NaN, None) do not crash
  Scenario 4 – Changing configured weights changes ranking

Run from the project root:
    python validate_hybrid_scoring.py
"""

import sys
import os
import traceback
import types
import math

# ── make project root importable ────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "backend", "services"))

# ── results tracker ──────────────────────────────────────────────────────────
PASS = 0
FAIL = 0
RESULTS = []


def ok(name: str, detail: str = ""):
    global PASS
    PASS += 1
    RESULTS.append(("PASS", name, detail))
    print(f"  ✓  {name}" + (f"  [{detail}]" if detail else ""))


def fail(name: str, detail: str = ""):
    global FAIL
    FAIL += 1
    RESULTS.append(("FAIL", name, detail))
    print(f"  ✗  {name}  ← {detail}")


def section(title: str):
    print(f"\n{'─'*65}")
    print(f"  {title}")
    print(f"{'─'*65}")


# ═══════════════════════════════════════════════════════════════════════
#  Import modules under test
# ═══════════════════════════════════════════════════════════════════════
section("IMPORT CHECK")
try:
    import config
    ok("config imported")
except Exception as e:
    fail("config import", str(e))
    sys.exit(1)

try:
    from backend.services.enhanced_recommender import (
        EnhancedNutritionRecommender,
        _check_ncf_availability,
        _NCF_AVAILABLE,
        _safe_float,
        _normalise_to_unit,
        generate_enhanced_recommendations,
    )
    ok("enhanced_recommender imported")
except Exception as e:
    fail("enhanced_recommender import", str(e))
    traceback.print_exc()
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════
#  Config weight validation
# ═══════════════════════════════════════════════════════════════════════
section("CONFIG WEIGHT VALIDATION")

try:
    n_w = config.NUTRITION_SCORE_WEIGHT
    c_w = config.CONTENT_SCORE_WEIGHT
    ncf_w = config.NCF_SCORE_WEIGHT

    if abs(n_w - 0.50) < 1e-9:
        ok("NUTRITION_SCORE_WEIGHT = 0.50", f"got {n_w}")
    else:
        fail("NUTRITION_SCORE_WEIGHT", f"expected 0.50, got {n_w}")

    if abs(c_w - 0.30) < 1e-9:
        ok("CONTENT_SCORE_WEIGHT = 0.30", f"got {c_w}")
    else:
        fail("CONTENT_SCORE_WEIGHT", f"expected 0.30, got {c_w}")

    if abs(ncf_w - 0.20) < 1e-9:
        ok("NCF_SCORE_WEIGHT = 0.20", f"got {ncf_w}")
    else:
        fail("NCF_SCORE_WEIGHT", f"expected 0.20, got {ncf_w}")

    total = n_w + c_w + ncf_w
    if abs(total - 1.0) < 1e-9:
        ok("Weights sum to 1.0", f"sum={total}")
    else:
        fail("Weights sum to 1.0", f"sum={total}")

    # Validate that invalid weights raise ValueError
    orig_n = config.NUTRITION_SCORE_WEIGHT
    config.NUTRITION_SCORE_WEIGHT = -0.1
    try:
        config._validate_hybrid_weights()
        fail("Negative weight should raise ValueError")
    except ValueError:
        ok("Negative weight raises ValueError")
    finally:
        config.NUTRITION_SCORE_WEIGHT = orig_n

    # Wrong sum
    orig_ncf = config.NCF_SCORE_WEIGHT
    config.NCF_SCORE_WEIGHT = 0.99
    try:
        config._validate_hybrid_weights()
        fail("Wrong-sum weights should raise ValueError")
    except ValueError:
        ok("Wrong-sum weights raise ValueError")
    finally:
        config.NCF_SCORE_WEIGHT = orig_ncf

    # Non-numeric
    orig_c = config.CONTENT_SCORE_WEIGHT
    config.CONTENT_SCORE_WEIGHT = "bad"
    try:
        config._validate_hybrid_weights()
        fail("Non-numeric weight should raise ValueError")
    except ValueError:
        ok("Non-numeric weight raises ValueError")
    finally:
        config.CONTENT_SCORE_WEIGHT = orig_c

except Exception as e:
    fail("Config weight validation block", str(e))
    traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════
#  Build a minimal pandas Series to test scoring functions in isolation
# ═══════════════════════════════════════════════════════════════════════
import pandas as pd
import numpy as np

GOOD_FOOD = pd.Series({
    "Dish Name":          "Test Idli",
    "Calories (kcal)":    150.0,
    "Protein (g)":        5.0,
    "Fibre (g)":          3.0,
    "Free Sugar (g)":     1.0,
    "Carbohydrates (g)":  28.0,
    "Fats (g)":           1.5,
    "Sodium (mg)":        80.0,
    "Iron (mg)":          1.2,
    "Calcium (mg)":       60.0,
    "Vitamin C (mg)":     5.0,
    "MealType":           "Breakfast",
})

HIGH_SUGAR_FOOD = pd.Series({
    "Dish Name":          "Sugary Cereal",
    "Calories (kcal)":    420.0,
    "Protein (g)":        3.0,
    "Fibre (g)":          0.5,
    "Free Sugar (g)":     22.0,
    "Carbohydrates (g)":  80.0,
    "Fats (g)":           8.0,
    "Sodium (mg)":        300.0,
    "Iron (mg)":          0.2,
    "Calcium (mg)":       10.0,
    "Vitamin C (mg)":     0.0,
    "MealType":           "Breakfast",
})

USER_DIABETES = {
    "diseases":       ["diabetes"],
    "age":            45,
    "gender":         "Female",
    "bmi":            27.0,
    "activity_level": "Sedentary",
    "daily_calories": 1800,
    "goal":           "Weight Loss",
    "region":         "South India",
}

USER_HEALTHY = {
    "diseases":       [],
    "age":            30,
    "gender":         "Male",
    "bmi":            22.0,
    "activity_level": "Moderate",
    "daily_calories": 2200,
    "goal":           "Maintain Weight",
    "region":         "North India",
}


# ═══════════════════════════════════════════════════════════════════════
#  Scenario 1: Nutrition + Content available, NCF unavailable
# ═══════════════════════════════════════════════════════════════════════
section("SCENARIO 1 — Nutrition + Content available, NCF unavailable")

rec = EnhancedNutritionRecommender()

try:
    # 1a. NCF must be unavailable (no trained model on disk)
    if not _NCF_AVAILABLE:
        ok("NCF unavailable confirmed (_NCF_AVAILABLE=False)")
    else:
        fail("NCF should be unavailable", "_NCF_AVAILABLE=True unexpectedly")

    # 1b. Effective weights re-normalised correctly
    expected_w_n   = 0.50 / (0.50 + 0.30)   # 0.625
    expected_w_c   = 0.30 / (0.50 + 0.30)   # 0.375
    expected_w_ncf = 0.0

    if abs(rec._eff_w_nutrition - expected_w_n) < 1e-6:
        ok("Effective nutrition weight = 0.6250", f"got {rec._eff_w_nutrition:.6f}")
    else:
        fail("Effective nutrition weight", f"expected {expected_w_n:.6f}, got {rec._eff_w_nutrition:.6f}")

    if abs(rec._eff_w_content - expected_w_c) < 1e-6:
        ok("Effective content weight = 0.3750", f"got {rec._eff_w_content:.6f}")
    else:
        fail("Effective content weight", f"expected {expected_w_c:.6f}, got {rec._eff_w_content:.6f}")

    if abs(rec._eff_w_ncf - expected_w_ncf) < 1e-9:
        ok("Effective NCF weight = 0.0")
    else:
        fail("Effective NCF weight", f"expected 0.0, got {rec._eff_w_ncf}")

    # 1c. NCF score returns (0.0, False) — no fake value
    ncf_score, ncf_used = rec._calculate_ncf_score(GOOD_FOOD, USER_HEALTHY)
    if ncf_score == 0.0 and ncf_used is False:
        ok("NCF score = 0.0, ncf_used = False (no fake score)")
    else:
        fail("NCF fake score check", f"score={ncf_score}, used={ncf_used}")

    # 1d. Hybrid score uses only nutrition + content
    h, bd = rec._calculate_hybrid_score(GOOD_FOOD, USER_HEALTHY)

    if bd['ncf_available'] is False:
        ok("Breakdown ncf_available=False")
    else:
        fail("Breakdown ncf_available", f"got {bd['ncf_available']}")

    if bd['ncf_score'] is None:
        ok("Breakdown ncf_score=None (not 0, not fake)")
    else:
        fail("Breakdown ncf_score should be None", f"got {bd['ncf_score']}")

    if 0.0 <= h <= 1.0:
        ok("Hybrid score in [0,1]", f"h={h:.4f}")
    else:
        fail("Hybrid score range", f"h={h}")

    # 1e. Manual verification of formula
    n_norm = bd['nutrition_norm']
    c_s    = bd['content_score']
    expected_h = n_norm * expected_w_n + c_s * expected_w_c
    expected_h = max(0.0, min(1.0, expected_h))
    if abs(h - expected_h) < 1e-6:
        ok("Hybrid score formula verified",
           f"h={h:.4f} = {n_norm:.4f}×{expected_w_n:.4f} + {c_s:.4f}×{expected_w_c:.4f}")
    else:
        fail("Hybrid score formula mismatch", f"got {h:.6f}, expected {expected_h:.6f}")

    # 1f. Recommendations are generated (end-to-end)
    result = generate_enhanced_recommendations(USER_DIABETES)
    total_foods = sum(len(v) for v in result['meal_plan'].values())
    if total_foods > 0:
        ok("End-to-end recommendations generated", f"{total_foods} foods across meals")
    else:
        fail("End-to-end: no foods generated")

    # 1g. scoring_info present and correct
    si = result.get('scoring_info', {})
    if si.get('ncf_available') is False:
        ok("scoring_info.ncf_available=False in output")
    else:
        fail("scoring_info.ncf_available", f"got {si.get('ncf_available')}")

    eff_sum = si.get('w_nutrition', 0) + si.get('w_content', 0) + si.get('w_ncf', 0)
    if abs(eff_sum - 1.0) < 1e-4:
        ok("Effective weights sum to 1.0 in scoring_info", f"sum={eff_sum:.6f}")
    else:
        fail("scoring_info weights sum", f"got {eff_sum}")

except Exception as e:
    fail("Scenario 1 block", str(e))
    traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════
#  Scenario 2: Only Nutrition available (content zeroed out)
# ═══════════════════════════════════════════════════════════════════════
section("SCENARIO 2 — Only Nutrition score available (content forced to 0)")

try:
    # Monkey-patch content score to return 0 for this scenario
    original_content = rec._calculate_content_score
    rec._calculate_content_score = lambda row, profile: 0.0

    h2, bd2 = rec._calculate_hybrid_score(GOOD_FOOD, USER_HEALTHY)

    if 0.0 <= h2 <= 1.0:
        ok("Hybrid score still in [0,1] with content=0", f"h={h2:.4f}")
    else:
        fail("Hybrid score range with content=0", f"h={h2}")

    # When content=0, hybrid = nutrition_norm × w_n_eff
    n_norm2 = bd2['nutrition_norm']
    expected_h2 = n_norm2 * expected_w_n   # w_n_eff = 0.625
    expected_h2 = max(0.0, min(1.0, expected_h2))
    if abs(h2 - expected_h2) < 1e-5:
        ok("Hybrid score = nutrition_norm × w_n when content=0",
           f"{h2:.4f} = {n_norm2:.4f} × {expected_w_n:.4f}")
    else:
        fail("Hybrid formula with content=0",
             f"got {h2:.6f}, expected {expected_h2:.6f}")

    if not math.isnan(h2):
        ok("No NaN when content=0")
    else:
        fail("NaN produced when content=0")

    # Restore
    rec._calculate_content_score = original_content

    # Full pipeline still works
    result2 = generate_enhanced_recommendations(USER_HEALTHY)
    total2 = sum(len(v) for v in result2['meal_plan'].values())
    if total2 > 0:
        ok("Pipeline works with healthy user (no disease)", f"{total2} foods")
    else:
        fail("Pipeline produced no foods for healthy user")

except Exception as e:
    fail("Scenario 2 block", str(e))
    traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════
#  Scenario 3: NaN / None values do not crash
# ═══════════════════════════════════════════════════════════════════════
section("SCENARIO 3 — NaN / None values handled safely")

try:
    # 3a. _safe_float helper
    assert _safe_float(None)       == 0.0, "_safe_float(None) != 0.0"
    assert _safe_float(float('nan')) == 0.0, "_safe_float(nan) != 0.0"
    assert _safe_float(float('inf')) == 0.0, "_safe_float(inf) != 0.0"
    assert _safe_float("bad")      == 0.0, "_safe_float('bad') != 0.0"
    assert abs(_safe_float(3.14) - 3.14) < 1e-9, "_safe_float(3.14) failed"
    ok("_safe_float handles None/NaN/Inf/str/float correctly")

    # 3b. _normalise_to_unit with edge cases
    assert _normalise_to_unit(0.0, 0.0)   == 0.0
    assert _normalise_to_unit(100.0, 0.0) == 0.0
    assert _normalise_to_unit(5.0, 10.0)  == 0.5
    assert _normalise_to_unit(15.0, 10.0) == 1.0   # clamped
    assert _normalise_to_unit(-1.0, 10.0) == 0.0   # clamped
    ok("_normalise_to_unit handles zero/negative/over-range")

    # 3c. All-NaN food row
    nan_food = pd.Series({
        "Dish Name":          "NaN Food",
        "Calories (kcal)":    float('nan'),
        "Protein (g)":        None,
        "Fibre (g)":          float('nan'),
        "Free Sugar (g)":     None,
        "Carbohydrates (g)":  float('nan'),
        "Fats (g)":           None,
        "Sodium (mg)":        float('nan'),
        "Iron (mg)":          None,
        "Calcium (mg)":       float('nan'),
        "Vitamin C (mg)":     None,
        "MealType":           "Breakfast",
    })
    try:
        n_s = rec._calculate_nutrition_score(nan_food, USER_HEALTHY)
        assert not math.isnan(n_s), f"nutrition score is NaN: {n_s}"
        ok("NaN food: nutrition score does not crash/NaN", f"score={n_s:.4f}")
    except Exception as ex:
        fail("NaN food nutrition score crashed", str(ex))

    try:
        c_s = rec._calculate_content_score(nan_food, USER_HEALTHY)
        assert not math.isnan(c_s), f"content score is NaN: {c_s}"
        assert 0.0 <= c_s <= 1.0, f"content score out of range: {c_s}"
        ok("NaN food: content score does not crash/NaN", f"score={c_s:.4f}")
    except Exception as ex:
        fail("NaN food content score crashed", str(ex))

    try:
        ncf_s, ncf_u = rec._calculate_ncf_score(nan_food, USER_HEALTHY)
        assert ncf_s == 0.0 and ncf_u is False
        ok("NaN food: NCF score returns (0.0, False) safely")
    except Exception as ex:
        fail("NaN food NCF score crashed", str(ex))

    try:
        h3, bd3 = rec._calculate_hybrid_score(nan_food, USER_HEALTHY)
        assert not math.isnan(h3), f"hybrid score is NaN: {h3}"
        assert 0.0 <= h3 <= 1.0,   f"hybrid score out of range: {h3}"
        ok("NaN food: hybrid score does not crash/NaN", f"h={h3:.4f}")
    except Exception as ex:
        fail("NaN food hybrid score crashed", str(ex))

    # 3d. Missing MealType column
    no_mealtype_food = GOOD_FOOD.copy()
    no_mealtype_food = no_mealtype_food.drop('MealType') if 'MealType' in no_mealtype_food.index else no_mealtype_food
    try:
        h4, bd4 = rec._calculate_hybrid_score(no_mealtype_food, USER_HEALTHY)
        assert not math.isnan(h4)
        ok("Missing MealType: hybrid score does not crash", f"h={h4:.4f}")
    except Exception as ex:
        fail("Missing MealType crashed", str(ex))

    # 3e. Empty diseases list
    user_no_disease = USER_HEALTHY.copy()
    user_no_disease['diseases'] = []
    try:
        h5, bd5 = rec._calculate_hybrid_score(GOOD_FOOD, user_no_disease)
        assert not math.isnan(h5)
        ok("Empty diseases list: hybrid score safe", f"h={h5:.4f}")
    except Exception as ex:
        fail("Empty diseases list crashed", str(ex))

    # 3f. Missing user_profile keys
    try:
        h6, bd6 = rec._calculate_hybrid_score(GOOD_FOOD, {})
        assert not math.isnan(h6)
        ok("Empty user_profile dict: hybrid score safe", f"h={h6:.4f}")
    except Exception as ex:
        fail("Empty user_profile crashed", str(ex))

except Exception as e:
    fail("Scenario 3 block", str(e))
    traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════
#  Scenario 4: Changing weights changes ranking
# ═══════════════════════════════════════════════════════════════════════
section("SCENARIO 4 — Weight changes affect ranking")

try:
    # Build two foods with very different nutrition vs content profiles:
    #   food_high_nutrition: high protein/fibre, moderate calories → high nutrition score
    #   food_high_content:   calories close to meal target → high content/calorie-fit score
    food_high_nutrition = pd.Series({
        "Dish Name":          "High Nutrition Food",
        "Calories (kcal)":    180.0,
        "Protein (g)":        20.0,   # very high protein → high nutrition score
        "Fibre (g)":          8.0,    # very high fibre
        "Free Sugar (g)":     0.5,
        "Carbohydrates (g)":  10.0,
        "Fats (g)":           2.0,
        "Sodium (mg)":        50.0,
        "Iron (mg)":          3.0,
        "Calcium (mg)":       80.0,
        "Vitamin C (mg)":     20.0,
        "MealType":           "Lunch",
    })

    food_good_calorie_fit = pd.Series({
        "Dish Name":          "Good Calorie Fit Food",
        "Calories (kcal)":    550.0,  # close to 2200×0.25=550 → perfect calorie fit
        "Protein (g)":        3.0,    # low protein → lower nutrition score
        "Fibre (g)":          1.0,    # low fibre
        "Free Sugar (g)":     2.0,
        "Carbohydrates (g)":  90.0,
        "Fats (g)":           15.0,
        "Sodium (mg)":        200.0,
        "Iron (mg)":          0.3,
        "Calcium (mg)":       20.0,
        "Vitamin C (mg)":     1.0,
        "MealType":           "Lunch",
    })

    # ── Baseline: w_n=0.625 (fallback), w_c=0.375 ─────────────────────
    h_n_base, bd_n_base = rec._calculate_hybrid_score(food_high_nutrition,  USER_HEALTHY)
    h_c_base, bd_c_base = rec._calculate_hybrid_score(food_good_calorie_fit, USER_HEALTHY)

    print(f"\n     Baseline (w_n=0.6250, w_c=0.3750):")
    print(f"       High-nutrition food  → hybrid={h_n_base:.4f}  "
          f"(nutr_norm={bd_n_base['nutrition_norm']:.4f}, content={bd_n_base['content_score']:.4f})")
    print(f"       Good-calorie food    → hybrid={h_c_base:.4f}  "
          f"(nutr_norm={bd_c_base['nutrition_norm']:.4f}, content={bd_c_base['content_score']:.4f})")

    # ── Shift: strongly boost content weight ──────────────────────────
    # Save original effective weights
    orig_eff_n   = rec._eff_w_nutrition
    orig_eff_c   = rec._eff_w_content
    orig_eff_ncf = rec._eff_w_ncf

    # Override effective weights directly (content-dominant)
    rec._eff_w_nutrition = 0.10
    rec._eff_w_content   = 0.90
    rec._eff_w_ncf       = 0.00

    h_n_shifted, bd_n_s = rec._calculate_hybrid_score(food_high_nutrition,  USER_HEALTHY)
    h_c_shifted, bd_c_s = rec._calculate_hybrid_score(food_good_calorie_fit, USER_HEALTHY)

    print(f"\n     Shifted (w_n=0.10, w_c=0.90):")
    print(f"       High-nutrition food  → hybrid={h_n_shifted:.4f}")
    print(f"       Good-calorie food    → hybrid={h_c_shifted:.4f}")

    # Restore
    rec._eff_w_nutrition = orig_eff_n
    rec._eff_w_content   = orig_eff_c
    rec._eff_w_ncf       = orig_eff_ncf

    # Assertion: at baseline (nutrition-dominant) the high-nutrition food
    # should score higher; after the content-dominant shift the good-calorie
    # food should score higher (or at least the high-nutrition food should
    # drop relative to the baseline gap).
    #
    # We test the directional property: the *gap* between the two foods
    # decreases (or reverses) when we shift weight toward content.

    baseline_gap = h_n_base - h_c_base         # positive → nutrition food winning
    shifted_gap  = h_n_shifted - h_c_shifted   # should be smaller (or negative)

    if shifted_gap < baseline_gap:
        ok("Weight shift changes ranking direction",
           f"baseline_gap={baseline_gap:+.4f}  shifted_gap={shifted_gap:+.4f}")
    else:
        # Not a hard failure — depends on dataset values — but flag it
        fail("Weight shift did not produce expected ranking shift",
             f"baseline_gap={baseline_gap:+.4f}  shifted_gap={shifted_gap:+.4f}")

    # Verify w_ncf=0 in both cases (NCF still absent)
    for bd, label in [(bd_n_base, "nutrition-food"), (bd_c_base, "content-food")]:
        if bd['w_ncf'] == 0.0 and bd['ncf_available'] is False:
            ok(f"NCF still absent in weight-shift test ({label})")
        else:
            fail(f"NCF check in weight-shift test ({label})",
                 f"w_ncf={bd['w_ncf']}, ncf_available={bd['ncf_available']}")

    # Verify effective weights sum to 1 after restoration
    restored_sum = rec._eff_w_nutrition + rec._eff_w_content + rec._eff_w_ncf
    if abs(restored_sum - 1.0) < 1e-6:
        ok("Effective weights restored and sum to 1.0", f"sum={restored_sum:.6f}")
    else:
        fail("Effective weights sum after restore", f"sum={restored_sum}")

except Exception as e:
    fail("Scenario 4 block", str(e))
    traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════
#  Score breakdown spot-check
# ═══════════════════════════════════════════════════════════════════════
section("SCORE BREAKDOWN — Example food spot-check")

try:
    h_ex, bd_ex = rec._calculate_hybrid_score(GOOD_FOOD, USER_DIABETES)
    print(f"\n     Food: {GOOD_FOOD['Dish Name']}   (User: diabetes, F45, BMI 27)")
    print(f"       nutrition_raw   = {bd_ex['nutrition_raw']}")
    print(f"       nutrition_norm  = {bd_ex['nutrition_norm']}")
    print(f"       content_score   = {bd_ex['content_score']}")
    print(f"       ncf_score       = {bd_ex['ncf_score']}  (None → not used)")
    print(f"       ncf_available   = {bd_ex['ncf_available']}")
    print(f"       w_nutrition     = {bd_ex['w_nutrition']}")
    print(f"       w_content       = {bd_ex['w_content']}")
    print(f"       w_ncf           = {bd_ex['w_ncf']}")
    print(f"       hybrid_score    = {bd_ex['hybrid_score']}")

    required_keys = [
        'nutrition_raw', 'nutrition_norm', 'content_score',
        'ncf_score', 'ncf_available', 'w_nutrition', 'w_content',
        'w_ncf', 'hybrid_score',
    ]
    missing = [k for k in required_keys if k not in bd_ex]
    if not missing:
        ok("All breakdown keys present")
    else:
        fail("Missing breakdown keys", str(missing))

    if bd_ex['hybrid_score'] == round(h_ex, 4):
        ok("breakdown.hybrid_score matches returned value")
    else:
        fail("breakdown.hybrid_score mismatch",
             f"{bd_ex['hybrid_score']} vs {round(h_ex, 4)}")

    # Check good food scores higher than high-sugar food for diabetic user
    h_good,  _ = rec._calculate_hybrid_score(GOOD_FOOD,       USER_DIABETES)
    h_sugar, _ = rec._calculate_hybrid_score(HIGH_SUGAR_FOOD, USER_DIABETES)
    print(f"\n     Diabetic user ranking check:")
    print(f"       GOOD_FOOD  ({GOOD_FOOD['Dish Name']})       → {h_good:.4f}")
    print(f"       HIGH_SUGAR ({HIGH_SUGAR_FOOD['Dish Name']}) → {h_sugar:.4f}")
    if h_good > h_sugar:
        ok("Good food ranks above high-sugar food for diabetic user")
    else:
        fail("Ranking: high-sugar food ranked above good food for diabetes",
             f"good={h_good:.4f}, sugar={h_sugar:.4f}")

except Exception as e:
    fail("Breakdown spot-check", str(e))
    traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════
#  Project-wide duplicate / stale weight search
# ═══════════════════════════════════════════════════════════════════════
section("PROJECT-WIDE STALE WEIGHT CHECK")

import re

WEIGHT_PATTERNS = [
    r"NCF_SCORE_WEIGHT",
    r"CONTENT_SCORE_WEIGHT",
    r"SUITABILITY_SCORE_WEIGHT",
    r"NUTRITION_SCORE_WEIGHT",
]

PY_EXTENSIONS = ('.py',)
SKIP_DIRS = {'.git', '__pycache__', '.pytest_cache', 'experiments'}

matches_by_file = {}
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
    for fname in filenames:
        if not fname.endswith(PY_EXTENSIONS):
            continue
        fpath = os.path.join(dirpath, fname)
        try:
            with open(fpath, encoding='utf-8', errors='ignore') as fh:
                for lineno, line in enumerate(fh, 1):
                    for pat in WEIGHT_PATTERNS:
                        if re.search(pat, line):
                            matches_by_file.setdefault(fpath, []).append(
                                (lineno, line.rstrip())
                            )
        except Exception:
            pass

print()
for fpath, hits in sorted(matches_by_file.items()):
    rel = os.path.relpath(fpath, ROOT)
    print(f"  {rel}:")
    for lineno, line in hits:
        print(f"      L{lineno:4d}: {line}")

# Authoritative definitions must only live in config.py
non_config = [f for f in matches_by_file if not f.endswith("config.py")]
# enhanced_recommender.py legitimately *reads* them — that's fine
legit_readers = {"enhanced_recommender.py", "validate_hybrid_scoring.py"}
true_duplicates = [
    f for f in non_config
    if os.path.basename(f) not in legit_readers
]

if not true_duplicates:
    ok("No duplicate weight definitions outside config.py + readers")
else:
    for f in true_duplicates:
        fail("Potential duplicate weight definition", os.path.relpath(f, ROOT))


# ═══════════════════════════════════════════════════════════════════════
#  Final summary
# ═══════════════════════════════════════════════════════════════════════
section("SUMMARY")
total = PASS + FAIL
print(f"\n  Total: {total}   PASS: {PASS}   FAIL: {FAIL}\n")

if FAIL > 0:
    print("  Failed checks:")
    for status, name, detail in RESULTS:
        if status == "FAIL":
            print(f"    ✗  {name}: {detail}")
    print()
    sys.exit(1)
else:
    print("  All checks passed.\n")
    sys.exit(0)
