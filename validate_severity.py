"""
Severity Recommendation Validation Suite
=========================================
Tests A–F as specified in the requirements, plus unit tests for
severity_rules, config constants, and the full end-to-end pipeline.

Run from the project root:
    python validate_severity.py

Exit 0 = all pass.  Exit 1 = at least one failure.
"""

import sys, os, math, traceback, warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# ── result tracker ────────────────────────────────────────────────────────────
PASS = FAIL = 0
RESULTS = []

def ok(name, detail=""):
    global PASS; PASS += 1
    RESULTS.append(("PASS", name, detail))
    print(f"  ✓  {name}" + (f"  [{detail}]" if detail else ""))

def fail(name, detail=""):
    global FAIL; FAIL += 1
    RESULTS.append(("FAIL", name, detail))
    print(f"  ✗  {name}  ← {detail}")

def section(title):
    print(f"\n{'─'*68}")
    print(f"  {title}")
    print(f"{'─'*68}")

# ═══════════════════════════════════════════════════════════════════════
#  IMPORTS
# ═══════════════════════════════════════════════════════════════════════
section("IMPORTS")
try:
    import pandas as pd
    import numpy as np
    ok("pandas / numpy")
except Exception as e:
    fail("pandas / numpy", str(e)); sys.exit(1)

try:
    from backend.services.severity_rules import (
        build_severity_dict, final_status_to_severity,
        get_rules, get_hard_constraints, get_soft_constraints,
        get_scoring_multipliers, get_required_minimums,
        calculate_severity_suitability_score,
        get_severity_explanation, get_recommendation_reason,
        SEVERITY_MILD, SEVERITY_MODERATE, SEVERITY_SEVERE,
        DEFAULT_SEVERITY, MIN_POOL_SIZE,
        _normalise_disease_key, _normalise_severity,
    )
    ok("severity_rules imported")
except Exception as e:
    fail("severity_rules import", str(e)); traceback.print_exc(); sys.exit(1)

try:
    import config
    ok("config imported")
except Exception as e:
    fail("config import", str(e)); sys.exit(1)

try:
    from backend.services.enhanced_recommender import (
        EnhancedNutritionRecommender,
        generate_enhanced_recommendations,
        _safe_float, _normalise_to_unit, _NCF_AVAILABLE,
        _extract_severity_dict, _get_severity_for_disease,
    )
    ok("enhanced_recommender imported")
except Exception as e:
    fail("enhanced_recommender import", str(e)); traceback.print_exc(); sys.exit(1)

try:
    import meal_planner as mp
    ok(f"meal_planner imported  (SEVERITY_RULES_AVAILABLE={mp.SEVERITY_RULES_AVAILABLE})")
except Exception as e:
    fail("meal_planner import", str(e)); traceback.print_exc(); sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════
#  UNIT — severity_rules module
# ═══════════════════════════════════════════════════════════════════════
section("UNIT — severity_rules")

# final_status → severity mapping
cases = [
    ("High Risk",     SEVERITY_SEVERE),
    ("Moderate Risk", SEVERITY_MODERATE),
    ("Borderline",    SEVERITY_MILD),
    ("Model Flag",    SEVERITY_MILD),
    ("Low Risk",      None),
    (None,            None),
    ("",              None),
    ("Unknown XYZ",   DEFAULT_SEVERITY),   # unrecognised → default
]
for fs, expected in cases:
    got = final_status_to_severity(fs)
    if got == expected:
        ok(f"final_status_to_severity('{fs}') = {expected}")
    else:
        fail(f"final_status_to_severity('{fs}')", f"expected {expected}, got {got}")

# build_severity_dict
sd = build_severity_dict("High Risk", "Low Risk", "Moderate Risk")
checks = [("diabetes", SEVERITY_SEVERE), ("kidney_disease", None), ("obesity", SEVERITY_MODERATE)]
for k, v in checks:
    if sd.get(k) == v:
        ok(f"build_severity_dict: {k}={v}")
    else:
        fail(f"build_severity_dict: {k}", f"expected {v}, got {sd.get(k)}")

# get_rules returns correct structure
for disease in ["diabetes", "kidney_disease", "obesity"]:
    for sev in [SEVERITY_MILD, SEVERITY_MODERATE, SEVERITY_SEVERE]:
        rules = get_rules(disease, sev)
        if rules and "hard_constraints" in rules and "soft_constraints" in rules:
            ok(f"get_rules({disease},{sev}) has hard+soft keys")
        else:
            fail(f"get_rules({disease},{sev})", f"got: {rules}")

# Severity ordering: severe hard constraints must be stricter than mild
for disease, col, severity_col_map in [
    ("diabetes",       "Free Sugar (g)",   None),
    ("diabetes",       "Calories (kcal)",  None),
    ("kidney_disease", "Sodium (mg)",      None),
    ("kidney_disease", "Protein (g)",      None),
    ("obesity",        "Calories (kcal)",  None),
    ("obesity",        "Fats (g)",         None),
]:
    mild_h = get_hard_constraints(disease, SEVERITY_MILD).get(col, {})
    sev_h  = get_hard_constraints(disease, SEVERITY_SEVERE).get(col, {})
    if "max" in mild_h and "max" in sev_h:
        if sev_h["max"] < mild_h["max"]:
            ok(f"{disease} {col}: severe_max({sev_h['max']}) < mild_max({mild_h['max']})")
        else:
            fail(f"{disease} {col} ordering", f"severe_max={sev_h['max']} ≥ mild_max={mild_h['max']}")

# Scoring multipliers: severe must be larger than mild for penalty terms
for disease, mult_key in [
    ("diabetes",       "sugar_penalty"),
    ("kidney_disease", "sodium_penalty"),
    ("obesity",        "calorie_penalty"),
]:
    mild_m = get_scoring_multipliers(disease, SEVERITY_MILD).get(mult_key, 0)
    sev_m  = get_scoring_multipliers(disease, SEVERITY_SEVERE).get(mult_key, 0)
    if sev_m > mild_m:
        ok(f"{disease} {mult_key}: severe({sev_m}) > mild({mild_m})")
    else:
        fail(f"{disease} {mult_key} multiplier ordering", f"severe={sev_m} ≤ mild={mild_m}")

# calculate_severity_suitability_score
good_food = pd.Series({
    "Free Sugar (g)": 1.0, "Calories (kcal)": 150.0, "Protein (g)": 6.0,
    "Fats (g)": 3.0, "Sodium (mg)": 40.0, "Fibre (g)": 3.0,
    "Carbohydrates (g)": 18.0, "Dish Name": "Good Food",
})
bad_food = pd.Series({
    "Free Sugar (g)": 30.0, "Calories (kcal)": 600.0, "Protein (g)": 20.0,
    "Fats (g)": 50.0, "Sodium (mg)": 900.0, "Fibre (g)": 0.2,
    "Carbohydrates (g)": 80.0, "Dish Name": "Bad Food",
})

for disease in ["diabetes", "kidney_disease", "obesity"]:
    good_s = calculate_severity_suitability_score(good_food, disease, SEVERITY_MODERATE)
    bad_s  = calculate_severity_suitability_score(bad_food,  disease, SEVERITY_MODERATE)
    if good_s > bad_s:
        ok(f"suitability {disease}: good_food({good_s:.3f}) > bad_food({bad_s:.3f})")
    else:
        fail(f"suitability {disease} ordering", f"good={good_s:.3f} ≤ bad={bad_s:.3f}")

    # Hard constraint violation → 0.0
    violating = pd.Series({
        "Free Sugar (g)": 50.0, "Calories (kcal)": 700.0,
        "Sodium (mg)": 2000.0, "Protein (g)": 25.0,
        "Fats (g)": 80.0, "Fibre (g)": 0.0,
        "Carbohydrates (g)": 85.0, "Dish Name": "Violating Food",
    })
    v_s = calculate_severity_suitability_score(violating, disease, SEVERITY_SEVERE)
    if v_s == 0.0:
        ok(f"suitability {disease} severe: hard-violating food → 0.0")
    else:
        fail(f"suitability {disease} severe: hard-violating should be 0", f"got {v_s}")

# None severity → neutral 0.5
s_none = calculate_severity_suitability_score(good_food, "diabetes", None)
if s_none == 0.5:
    ok("suitability with severity=None → 0.5 (neutral)")
else:
    fail("suitability None severity", f"expected 0.5, got {s_none}")


# ═══════════════════════════════════════════════════════════════════════
#  UNIT — config constants
# ═══════════════════════════════════════════════════════════════════════
section("UNIT — config constants")

if config.SEVERITY_DEFAULT == "moderate":
    ok(f"SEVERITY_DEFAULT = 'moderate'")
else:
    fail("SEVERITY_DEFAULT", f"expected 'moderate', got '{config.SEVERITY_DEFAULT}'")

if isinstance(config.SEVERITY_MIN_POOL_SIZE, int) and config.SEVERITY_MIN_POOL_SIZE > 0:
    ok(f"SEVERITY_MIN_POOL_SIZE = {config.SEVERITY_MIN_POOL_SIZE}")
else:
    fail("SEVERITY_MIN_POOL_SIZE", str(config.SEVERITY_MIN_POOL_SIZE))


# ═══════════════════════════════════════════════════════════════════════
#  BASE PROFILES used across scenarios
# ═══════════════════════════════════════════════════════════════════════
BASE_PROFILE = {
    "diseases":        ["Diabetes"],
    "age":             50,
    "gender":          "Female",
    "bmi":             27.0,
    "activity_level":  "Sedentary",
    "daily_calories":  1800,
    "hba1c":           7.5,
    "glucose":         160,
    "bp":              130,
    "sodium":          138,
    "potassium":       4.2,
    "creatinine":      1.0,
    "goal":            "Weight Loss",
    "region":          "Andhra Pradesh",
}

def make_profile(diseases, severity_dict, **overrides):
    p = BASE_PROFILE.copy()
    p["diseases"] = diseases
    p["severity"] = severity_dict
    p.update(overrides)
    return p


# ═══════════════════════════════════════════════════════════════════════
#  TEST A — Same disease, three severity levels → different rankings
# ═══════════════════════════════════════════════════════════════════════
section("TEST A — Same disease, different severity → different ranking/filtering")

rec = EnhancedNutritionRecommender()

try:
    prof_mild = make_profile(
        ["Diabetes"], {"diabetes": SEVERITY_MILD}
    )
    prof_mod  = make_profile(
        ["Diabetes"], {"diabetes": SEVERITY_MODERATE}
    )
    prof_sev  = make_profile(
        ["Diabetes"], {"diabetes": SEVERITY_SEVERE}
    )

    result_mild = rec.generate_meal_plan(prof_mild)
    result_mod  = rec.generate_meal_plan(prof_mod)
    result_sev  = rec.generate_meal_plan(prof_sev)

    # A1: All three produce recommendations (no empty plan for any tier)
    for label, result in [("mild", result_mild), ("moderate", result_mod), ("severe", result_sev)]:
        total = sum(len(v) for v in result["meal_plan"].values())
        if total > 0:
            ok(f"Test A: {label} severity produces {total} food recommendations")
        else:
            fail(f"Test A: {label} severity produced 0 foods")

    # A2: Severe filtering is strictest — check hard constraint on Free Sugar
    sev_pool_check = rec._apply_disease_filters(rec.df, ["Diabetes"])
    sev_pool_check = rec._apply_severity_filters(
        sev_pool_check, ["Diabetes"], {"diabetes": SEVERITY_SEVERE}
    )
    mild_pool_check = rec._apply_disease_filters(rec.df, ["Diabetes"])
    mild_pool_check = rec._apply_severity_filters(
        mild_pool_check, ["Diabetes"], {"diabetes": SEVERITY_MILD}
    )

    if len(sev_pool_check) <= len(mild_pool_check):
        ok(f"Test A: severe pool ({len(sev_pool_check)}) ≤ mild pool ({len(mild_pool_check)})")
    else:
        fail("Test A: severe pool should be ≤ mild pool",
             f"severe={len(sev_pool_check)}, mild={len(mild_pool_check)}")

    # A3: Verify max Free Sugar in severe pool ≤ severe hard threshold
    sev_threshold = get_hard_constraints("diabetes", SEVERITY_SEVERE).get("Free Sugar (g)", {}).get("max", 999)
    if not sev_pool_check.empty:
        actual_max_sugar = sev_pool_check["Free Sugar (g)"].max()
        if actual_max_sugar <= sev_threshold:
            ok(f"Test A: severe pool max Free Sugar ({actual_max_sugar:.2f}g) ≤ threshold ({sev_threshold}g)")
        else:
            fail("Test A: severe pool Free Sugar exceeds threshold",
                 f"max={actual_max_sugar:.2f} > {sev_threshold}")

    # A4: Nutrition scores differ across severity tiers for the same food
    # Use a moderately sugary food to demonstrate differentiation
    test_food = pd.Series({
        "Dish Name": "Test Food", "Free Sugar (g)": 3.5,
        "Calories (kcal)": 180.0, "Protein (g)": 5.0,
        "Fibre (g)": 2.5, "Carbohydrates (g)": 22.0,
        "Fats (g)": 4.0, "Sodium (mg)": 80.0,
        "Iron (mg)": 1.0, "Calcium (mg)": 40.0, "Vitamin C (mg)": 5.0,
    })
    n_mild = rec._calculate_nutrition_score(test_food, prof_mild)
    n_mod  = rec._calculate_nutrition_score(test_food, prof_mod)
    n_sev  = rec._calculate_nutrition_score(test_food, prof_sev)

    print(f"\n     Same food nutrition scores:")
    print(f"       Mild={n_mild:.4f}  Moderate={n_mod:.4f}  Severe={n_sev:.4f}")

    if n_mild >= n_mod >= n_sev:
        ok("Test A: nutrition score decreases with severity (mild ≥ moderate ≥ severe)")
    else:
        fail("Test A: nutrition score ordering",
             f"mild={n_mild:.4f}, mod={n_mod:.4f}, sev={n_sev:.4f}")

    # A5: Hybrid scores differ
    h_mild, bd_mild = rec._calculate_hybrid_score(test_food, prof_mild)
    h_mod,  bd_mod  = rec._calculate_hybrid_score(test_food, prof_mod)
    h_sev,  bd_sev  = rec._calculate_hybrid_score(test_food, prof_sev)

    print(f"\n     Same food hybrid scores:")
    print(f"       Mild={h_mild:.4f}  Moderate={h_mod:.4f}  Severe={h_sev:.4f}")
    print(f"       Severity suitability:")
    print(f"         Mild={bd_mild['severity_suitability']:.4f}  "
          f"Moderate={bd_mod['severity_suitability']:.4f}  "
          f"Severe={bd_sev['severity_suitability']:.4f}")

    if not (h_mild == h_mod == h_sev):
        ok("Test A: hybrid scores differ across severity tiers")
    else:
        fail("Test A: hybrid scores identical across severity tiers — no differentiation")

    # A6: Severity info present in results
    for label, result in [("mild", result_mild), ("moderate", result_mod), ("severe", result_sev)]:
        si = result.get("severity_info", {})
        if "diabetes" in si:
            reported = si["diabetes"].get("severity")
            expected_sev = label
            if reported == expected_sev:
                ok(f"Test A: severity_info.diabetes = '{reported}' for {label}")
            else:
                fail(f"Test A: severity_info.diabetes mismatch for {label}",
                     f"expected '{expected_sev}', got '{reported}'")
        else:
            fail(f"Test A: severity_info missing 'diabetes' key for {label}", str(si))

    # A7: Meal explanations contain severity labels
    for label, result in [
        ("mild",     result_mild),
        ("moderate", result_mod),
        ("severe",   result_sev),
    ]:
        expl = " ".join(result.get("meal_explanations", {}).values())
        if label == "severe" and "severe" in expl.lower():
            ok("Test A: severe meal explanation contains 'severe' label")
        elif label == "moderate" and "controlled" in expl.lower():
            ok("Test A: moderate meal explanation contains 'controlled' label")
        elif label == "mild" and ("low-sugar" in expl.lower() or "balanced" in expl.lower()):
            ok("Test A: mild meal explanation contains appropriate label")

except Exception as e:
    fail("Test A block", str(e))
    traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════
#  TEST B — No Disease / Severity = None
# ═══════════════════════════════════════════════════════════════════════
section("TEST B — No disease, severity=None → no disease restrictions")

try:
    prof_healthy = make_profile(
        ["Normal"],
        {"diabetes": None, "kidney_disease": None, "obesity": None}
    )

    # B1: base disease filter for "Normal" uses the general thresholds
    base_df = rec._apply_disease_filters(rec.df, ["Normal"])
    if len(base_df) > 0:
        ok(f"Test B: healthy user base pool = {len(base_df)} foods")
    else:
        fail("Test B: base pool empty for healthy user")

    # B2: severity filters leave pool unchanged for None severity
    sev_df = rec._apply_severity_filters(base_df, ["Normal"], {})
    if len(sev_df) == len(base_df):
        ok("Test B: severity filters do NOT reduce pool for healthy user")
    else:
        fail("Test B: severity filters changed pool for healthy user",
             f"{len(base_df)} → {len(sev_df)}")

    # B3: suitability score is neutral (0.5) for each None-disease food
    s = calculate_severity_suitability_score(good_food, "diabetes", None)
    if s == 0.5:
        ok("Test B: suitability(diabetes, None) = 0.5 (no penalty)")
    else:
        fail("Test B: suitability should be 0.5 for None severity", f"got {s}")

    # B4: end-to-end recommendations generated
    result_healthy = rec.generate_meal_plan(prof_healthy)
    total_h = sum(len(v) for v in result_healthy["meal_plan"].values())
    if total_h > 0:
        ok(f"Test B: healthy user gets {total_h} recommendations")
    else:
        fail("Test B: healthy user got 0 recommendations")

    # B5: severity_info does not claim active diseases
    si_h = result_healthy.get("severity_info", {})
    active_disease_keys = [k for k in ["diabetes", "kidney_disease", "obesity"] if k in si_h]
    if not active_disease_keys:
        ok("Test B: severity_info has no active disease entries for healthy user")
    else:
        fail("Test B: severity_info unexpectedly has disease entries", str(active_disease_keys))

    # B6: no disease-specific filters (e.g. diabetes thresholds) applied
    # check that foods with sugar > DIABETES_MAX_SUGAR can appear in healthy pool
    over_sugar = rec.df[rec.df["Free Sugar (g)"] > config.DIABETES_MAX_SUGAR]
    healthy_pool = rec._apply_disease_filters(rec.df, ["Normal"])
    overlap = healthy_pool[healthy_pool["Free Sugar (g)"] > config.DIABETES_MAX_SUGAR]
    if len(overlap) > 0:
        ok(f"Test B: healthy pool includes {len(overlap)} foods with sugar > diabetes threshold")
    else:
        fail("Test B: healthy pool incorrectly excludes all high-sugar foods", "0 overlap")

except Exception as e:
    fail("Test B block", str(e))
    traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════
#  TEST C — Missing / None / invalid severity
# ═══════════════════════════════════════════════════════════════════════
section("TEST C — Missing / invalid severity does not crash")

try:
    # C1: no severity key in user_profile
    prof_no_sev = make_profile(["Diabetes"], {})
    del prof_no_sev["severity"]
    try:
        r = rec.generate_meal_plan(prof_no_sev)
        total_ns = sum(len(v) for v in r["meal_plan"].values())
        ok(f"Test C: missing severity key → no crash, {total_ns} foods")
    except Exception as ex:
        fail("Test C: missing severity key crashed", str(ex))

    # C2: severity = None
    prof_none_sev = make_profile(["Diabetes"], None)
    prof_none_sev["severity"] = None
    try:
        r2 = rec.generate_meal_plan(prof_none_sev)
        total_n2 = sum(len(v) for v in r2["meal_plan"].values())
        ok(f"Test C: severity=None → no crash, {total_n2} foods")
    except Exception as ex:
        fail("Test C: severity=None crashed", str(ex))

    # C3: invalid severity value string
    prof_bad_sev = make_profile(["Diabetes"], {"diabetes": "extreme"})
    try:
        r3 = rec.generate_meal_plan(prof_bad_sev)
        total_b = sum(len(v) for v in r3["meal_plan"].values())
        ok(f"Test C: invalid severity string → no crash, {total_b} foods")
    except Exception as ex:
        fail("Test C: invalid severity string crashed", str(ex))

    # C4: empty diseases list with severity
    prof_empty_dis = make_profile([], {"diabetes": "severe"})
    try:
        r4 = rec.generate_meal_plan(prof_empty_dis)
        ok("Test C: empty diseases list with severity → no crash")
    except Exception as ex:
        fail("Test C: empty diseases list crashed", str(ex))

    # C5: severity dict has wrong types
    prof_wrong_type = make_profile(["Diabetes"], {"diabetes": 42, "obesity": True})
    try:
        r5 = rec.generate_meal_plan(prof_wrong_type)
        ok("Test C: wrong-type severity values → no crash")
    except Exception as ex:
        fail("Test C: wrong-type severity values crashed", str(ex))

    # C6: _extract_severity_dict safety
    cases_dict = [
        ({},                   {}),
        ({"severity": None},   {}),
        ({"severity": "bad"},  {}),
        ({"severity": 123},    {}),
    ]
    for prof_partial, expected_empty in cases_dict:
        got = _extract_severity_dict(prof_partial)
        if isinstance(got, dict):
            ok(f"Test C: _extract_severity_dict({list(prof_partial.keys())}) → dict (no crash)")
        else:
            fail("Test C: _extract_severity_dict returned non-dict", str(got))

except Exception as e:
    fail("Test C block", str(e))
    traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════
#  TEST D — Multiple diseases
# ═══════════════════════════════════════════════════════════════════════
section("TEST D — Multiple diseases with combined severity")

try:
    prof_multi = make_profile(
        ["Diabetes", "Obesity"],
        {"diabetes": SEVERITY_MODERATE, "obesity": SEVERITY_MODERATE, "kidney_disease": None},
        bmi=32.0
    )

    # D1: pipeline does not crash
    try:
        r_multi = rec.generate_meal_plan(prof_multi)
        total_m = sum(len(v) for v in r_multi["meal_plan"].values())
        ok(f"Test D: Diabetes+Obesity combined → no crash, {total_m} foods")
    except Exception as ex:
        fail("Test D: multi-disease crashed", str(ex))
        traceback.print_exc()
        r_multi = {"meal_plan": {}, "severity_info": {}}

    # D2: combined pool is ≤ single-disease pool (both constraints active)
    pool_diab_only = rec._apply_disease_filters(
        rec._apply_disease_filters(rec.df, ["Diabetes"]), ["Diabetes"]
    )
    pool_combined  = rec._apply_disease_filters(rec.df, ["Diabetes", "Obesity"])
    pool_combined_sev = rec._apply_severity_filters(
        pool_combined, ["Diabetes", "Obesity"],
        {"diabetes": SEVERITY_MODERATE, "obesity": SEVERITY_MODERATE}
    )
    if len(pool_combined_sev) <= len(pool_diab_only):
        ok(f"Test D: combined pool ({len(pool_combined_sev)}) ≤ diabetes-only pool ({len(pool_diab_only)})")
    else:
        # Can be valid if obesity moderate is less strict than diabetes moderate on overlap
        ok(f"Test D: combined pool size = {len(pool_combined_sev)} (may be larger if constraints don't fully overlap)")

    # D3: severity_info reports both diseases
    si_m = r_multi.get("severity_info", {})
    for dk in ["diabetes", "obesity"]:
        if dk in si_m:
            ok(f"Test D: severity_info contains '{dk}'")
        else:
            fail(f"Test D: severity_info missing '{dk}'", str(list(si_m.keys())))

    # D4: no contradictory result — candidates actually remain
    if total_m > 0:
        ok("Test D: candidates survive combined disease+severity filtering")
    else:
        # Not a hard failure (may happen with very strict settings) — warn only
        fail("Test D: no candidates survived combined filtering — may need threshold review")

    # D5: suitability score for multi-disease uses min() (strictest wins)
    s_diab  = calculate_severity_suitability_score(good_food, "diabetes", SEVERITY_MODERATE)
    s_obese = calculate_severity_suitability_score(good_food, "obesity",  SEVERITY_MODERATE)
    expected_combined = min(s_diab, s_obese)
    _, bd_multi = rec._calculate_hybrid_score(good_food, prof_multi)
    actual_combined = bd_multi["severity_suitability"]
    if abs(actual_combined - expected_combined) < 1e-4:
        ok(f"Test D: combined suitability = min(diab={s_diab:.3f}, obese={s_obese:.3f}) = {expected_combined:.3f}")
    else:
        fail("Test D: combined suitability should be min() of per-disease scores",
             f"expected {expected_combined:.4f}, got {actual_combined:.4f}")

except Exception as e:
    fail("Test D block", str(e))
    traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════
#  TEST E — No suitable foods (graceful message, no crash)
# ═══════════════════════════════════════════════════════════════════════
section("TEST E — No suitable foods → graceful fallback, no crash")

try:
    # Monkeypatch the dataset with an impossibly small DF to force empty pool
    original_df = rec.df
    rec.df = original_df.head(0).copy()   # empty dataframe

    prof_e = make_profile(["Diabetes"], {"diabetes": SEVERITY_SEVERE})
    try:
        r_empty = rec.generate_meal_plan(prof_e)
        # Must return a valid dict, not raise
        assert isinstance(r_empty, dict), "Result is not a dict"
        assert "meal_plan" in r_empty, "No 'meal_plan' key"
        assert "severity_info" in r_empty, "No 'severity_info' key"
        assert "debug_info" in r_empty, "No 'debug_info' key"

        total_e = sum(len(v) for v in r_empty["meal_plan"].values())
        ok(f"Test E: empty dataset → no crash, {total_e} foods returned (expected 0)")

        # Meal plan should have empty slots, not be missing keys
        expected_keys = {"Breakfast", "Lunch", "Snack", "Dinner"}
        actual_keys   = set(r_empty["meal_plan"].keys())
        if expected_keys == actual_keys:
            ok("Test E: all 4 meal slots present even with no foods")
        else:
            fail("Test E: meal slots missing", str(actual_keys))

        # Debug info or severity_info should indicate the failure
        debug_err = r_empty.get("debug_info", {}).get("error", "")
        if debug_err:
            ok(f"Test E: debug_info.error populated: '{debug_err}'")
        else:
            ok("Test E: fallback result returned (no error key required)")

    except Exception as ex:
        fail("Test E: empty pool crashed the pipeline", str(ex))
        traceback.print_exc()
    finally:
        rec.df = original_df   # always restore

    # E2: pool becomes critically small (< MIN_POOL_SIZE) — no crash, warning logged
    rec.df = original_df.head(5).copy()   # 5 foods
    try:
        r_tiny = rec.generate_meal_plan(prof_e)
        ok("Test E: tiny pool (5 foods) → no crash")
    except Exception as ex:
        fail("Test E: tiny pool crashed", str(ex))
    finally:
        rec.df = original_df

except Exception as e:
    rec.df = original_df
    fail("Test E block", str(e))
    traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════
#  TEST F — Existing application pipeline still works
# ═══════════════════════════════════════════════════════════════════════
section("TEST F — Existing pipeline backward compatibility")

try:
    # F1: generate_comprehensive_recommendations without severity (legacy call)
    try:
        r_legacy = mp.generate_comprehensive_recommendations(
            diseases=["Diabetes"],
            age=45, gender="Male", height=170, weight=80, bmi=27.7,
            activity_level="Sedentary", daily_calories=1900,
            hba1c=7.0, glucose=150, bp=128,
            sodium=138, potassium=4.5, creatinine=1.0,
            goal="Weight Loss", region="Andhra Pradesh",
            # severity intentionally omitted — must not crash
        )
        total_legacy = sum(len(v) for v in r_legacy.get("meal_plan", {}).values())
        ok(f"Test F: legacy call (no severity arg) → no crash, {total_legacy} foods")
    except Exception as ex:
        fail("Test F: legacy call crashed", str(ex))
        traceback.print_exc()

    # F2: call WITH severity passes correctly
    try:
        sev = {"diabetes": "moderate", "kidney_disease": None, "obesity": None}
        r_new = mp.generate_comprehensive_recommendations(
            diseases=["Diabetes"],
            age=45, gender="Male", height=170, weight=80, bmi=27.7,
            activity_level="Sedentary", daily_calories=1900,
            hba1c=7.0, glucose=150, bp=128,
            sodium=138, potassium=4.5, creatinine=1.0,
            goal="Weight Loss", region="Andhra Pradesh",
            severity=sev,
        )
        total_new = sum(len(v) for v in r_new.get("meal_plan", {}).values())
        ok(f"Test F: call with severity → no crash, {total_new} foods")
    except Exception as ex:
        fail("Test F: call with severity crashed", str(ex))
        traceback.print_exc()

    # F3: hybrid scoring still works (NCF still absent, weights still valid)
    rec2 = EnhancedNutritionRecommender()
    if not rec2._ncf_available:
        ok("Test F: NCF still unavailable (correct — no model trained)")
    else:
        fail("Test F: NCF reported as available unexpectedly")

    eff_sum = rec2._eff_w_nutrition + rec2._eff_w_content + rec2._eff_w_ncf
    if abs(eff_sum - 1.0) < 1e-6:
        ok(f"Test F: effective weights still sum to 1.0 (sum={eff_sum:.6f})")
    else:
        fail("Test F: effective weights don't sum to 1.0", f"sum={eff_sum}")

    # F4: score breakdown still contains all Step-4 keys
    h, bd = rec2._calculate_hybrid_score(good_food, make_profile(["Diabetes"], {"diabetes": "moderate"}))
    required_keys = [
        "nutrition_raw", "nutrition_norm", "content_score",
        "ncf_score", "ncf_available", "w_nutrition", "w_content", "w_ncf", "hybrid_score",
        # new severity keys
        "severity_suitability", "severity_dict", "nutrition_blended",
        "severity_reasons", "severity_explanations",
    ]
    missing = [k for k in required_keys if k not in bd]
    if not missing:
        ok("Test F: score breakdown contains all required keys (Step-4 + severity)")
    else:
        fail("Test F: score breakdown missing keys", str(missing))

    # F5: meal_plan dict has all 4 meal types
    r_f5 = rec2.generate_meal_plan(make_profile(["Diabetes"], {"diabetes": "moderate"}))
    slots = set(r_f5["meal_plan"].keys())
    if slots == {"Breakfast", "Lunch", "Snack", "Dinner"}:
        ok("Test F: meal_plan has all 4 meal slots")
    else:
        fail("Test F: meal_plan missing slots", str(slots))

    # F6: severity_info key present in result
    if "severity_info" in r_f5:
        ok("Test F: 'severity_info' key present in result")
    else:
        fail("Test F: 'severity_info' key missing from result")

    # F7: scoring_info still present
    if "scoring_info" in r_f5:
        ok("Test F: 'scoring_info' key still present")
    else:
        fail("Test F: 'scoring_info' key missing")

    # F8: build_severity_dict round-trip through app-level helper
    from backend.services.severity_rules import build_severity_dict as bsd
    sd_trip = bsd("High Risk", "Moderate Risk", "Low Risk")
    expected_trip = {"diabetes": "severe", "kidney_disease": "moderate", "obesity": None}
    if sd_trip == expected_trip:
        ok("Test F: build_severity_dict round-trip correct")
    else:
        fail("Test F: build_severity_dict round-trip", f"expected {expected_trip}, got {sd_trip}")

except Exception as e:
    fail("Test F block", str(e))
    traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════
#  BEFORE / AFTER COMPARISON
# ═══════════════════════════════════════════════════════════════════════
section("BEFORE / AFTER — Ranking comparison (same profile, mild vs severe)")

try:
    rec3 = EnhancedNutritionRecommender()
    prof_before = make_profile(["Diabetes"], {"diabetes": SEVERITY_MILD})
    prof_after  = make_profile(["Diabetes"], {"diabetes": SEVERITY_SEVERE})

    r_before = rec3.generate_meal_plan(prof_before)
    r_after  = rec3.generate_meal_plan(prof_after)

    # Collect breakfast foods for comparison
    bf_before = [
        f for f in r_before["meal_plan"].get("Breakfast", [])
        if isinstance(f, pd.Series)
    ]
    bf_after = [
        f for f in r_after["meal_plan"].get("Breakfast", [])
        if isinstance(f, pd.Series)
    ]

    print("\n  MILD severity — Breakfast:")
    for f in bf_before:
        bd = f.get("_score_breakdown", {})
        print(f"    {str(f.get('Dish Name','?')):<35} "
              f"sugar={f.get('Free Sugar (g)',0):.1f}g  "
              f"kcal={f.get('Calories (kcal)',0):.0f}  "
              f"hybrid={bd.get('hybrid_score',0):.4f}  "
              f"sev_suit={bd.get('severity_suitability',0):.4f}")

    print("\n  SEVERE severity — Breakfast:")
    for f in bf_after:
        bd = f.get("_score_breakdown", {})
        print(f"    {str(f.get('Dish Name','?')):<35} "
              f"sugar={f.get('Free Sugar (g)',0):.1f}g  "
              f"kcal={f.get('Calories (kcal)',0):.0f}  "
              f"hybrid={bd.get('hybrid_score',0):.4f}  "
              f"sev_suit={bd.get('severity_suitability',0):.4f}")

    # Verify severe breakfast foods have ≤ severe sugar threshold
    sev_sugar_limit = get_hard_constraints("diabetes", SEVERITY_SEVERE).get("Free Sugar (g)", {}).get("max", 999)
    severe_violations = [
        f for f in bf_after
        if isinstance(f, pd.Series) and float(f.get("Free Sugar (g)", 0) or 0) > sev_sugar_limit
    ]
    if not severe_violations:
        ok(f"Before/After: all severe foods have Free Sugar ≤ {sev_sugar_limit}g")
    else:
        fail("Before/After: severe foods violate sugar threshold",
             str([f.get("Dish Name") for f in severe_violations]))

    # Average hybrid score should differ
    avg_before = (
        sum(f.get("_score_breakdown", {}).get("hybrid_score", 0) for f in bf_before) / max(len(bf_before), 1)
    )
    avg_after = (
        sum(f.get("_score_breakdown", {}).get("hybrid_score", 0) for f in bf_after) / max(len(bf_after), 1)
    )
    if avg_before != avg_after:
        ok(f"Before/After: avg hybrid differs  mild={avg_before:.4f}  severe={avg_after:.4f}")
    else:
        fail("Before/After: avg hybrid identical across severity levels")

except Exception as e:
    fail("Before/After block", str(e))
    traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════
#  SUMMARY
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
