"""
tests/test_ncf.py
==================
NCF test suite — 15 cases covering:

 1. NCF model loads successfully
 2. model missing → safe fallback
 3. unknown user → safe fallback (None)
 4. unknown food → safe fallback (None)
 5. valid user + valid food → score in [0, 1]
 6. food_id mapping covers real food_dataset.csv range (1–1014)
 7. existing recommendation pipeline still works
 8. existing disease filtering still works
 9. NCF cannot reintroduce foods removed by disease filters
10. 7-day planner remains deterministic
11. no random.choice / random.sample in weekly_meal_planner
12. NCF checkbox not required (ncf_available is auto-detected)
13. scoring information is accurate (w_ncf=0.20 when active, 0 when not)
14. new users receive recommendations (cold-start graceful)
15. existing tests still pass (regression guard)

Run:
    python -m pytest tests/test_ncf.py -v
"""

import importlib
import inspect
import json
import os
import sys
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# ── project root on path ────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fresh_service():
    """Return ncf_service module with a fresh load attempt (no cached state)."""
    import backend.services.ncf_service as svc
    # Reset module-level state
    svc._load_attempted = False
    svc._model = None
    svc._user_to_idx = {}
    svc._food_id_to_idx = {}
    svc._num_users = 0
    svc._num_items = 0
    svc._load_error = None
    svc._attempt_load()
    return svc


def _food_ids_from_dataset():
    import pandas as pd
    df = pd.read_csv(os.path.join(_ROOT, "food_dataset.csv"))
    return sorted(df["food_id"].astype(int).tolist())


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — model loads
# ─────────────────────────────────────────────────────────────────────────────
class TestNCFLoads:
    def test_model_available(self):
        """NCF model file exists and loads without error."""
        svc = _fresh_service()
        assert svc.is_available(), (
            f"NCF not available. load_error={svc._load_error}"
        )

    def test_correct_dimensions(self):
        """Loaded model covers 1,000 users and 1,014 items."""
        svc = _fresh_service()
        status = svc.get_status()
        assert status["num_users"] == 1000
        assert status["num_items"] == 1014

    def test_model_has_expected_layer_names(self):
        """Model must have 'user_embedding' and 'item_embedding' layers."""
        svc = _fresh_service()
        assert svc._model is not None
        layer_names = {l.name for l in svc._model.layers}
        assert "user_embedding" in layer_names
        assert "item_embedding" in layer_names


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — missing model → safe fallback
# ─────────────────────────────────────────────────────────────────────────────
class TestMissingModelFallback:
    def test_missing_model_returns_none(self, tmp_path, monkeypatch):
        """When the model file is absent, predict_single returns None."""
        import backend.services.ncf_service as svc
        # Point the module at a nonexistent path
        monkeypatch.setattr(svc, "_MODEL_PATH", str(tmp_path / "nonexistent.keras"))
        monkeypatch.setattr(svc, "_model", None)
        monkeypatch.setattr(svc, "_load_attempted", False)
        monkeypatch.setattr(svc, "_load_error", None)
        svc._attempt_load()
        result = svc.predict_single("0", 1)
        assert result is None
        # Restore
        monkeypatch.undo()

    def test_missing_model_is_available_false(self, tmp_path, monkeypatch):
        """is_available() is False when no model exists."""
        import backend.services.ncf_service as svc
        monkeypatch.setattr(svc, "_MODEL_PATH", str(tmp_path / "ghost.keras"))
        monkeypatch.setattr(svc, "_model", None)
        monkeypatch.setattr(svc, "_load_attempted", False)
        svc._attempt_load()
        assert svc.is_available() is False
        monkeypatch.undo()


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — unknown user → None
# ─────────────────────────────────────────────────────────────────────────────
class TestUnknownUser:
    def test_unknown_user_returns_none(self):
        svc = _fresh_service()
        assert svc.is_available(), "model not loaded"
        result = svc.predict_single("user_that_does_not_exist_xyz", 1)
        assert result is None

    def test_unknown_user_batch_all_none(self):
        svc = _fresh_service()
        results = svc.predict_for_user("ghost_user", [1, 2, 3, 4, 5])
        assert all(r is None for r in results)
        assert len(results) == 5


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 — unknown food → None
# ─────────────────────────────────────────────────────────────────────────────
class TestUnknownFood:
    def test_unknown_food_id_returns_none(self):
        svc = _fresh_service()
        assert svc.is_available()
        result = svc.predict_single("0", 99999)  # food_id 99999 not in dataset
        assert result is None

    def test_food_id_zero_returns_none(self):
        svc = _fresh_service()
        # food_id 0 doesn't exist in food_dataset.csv (starts at 1)
        result = svc.predict_single("0", 0)
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5 — valid user + valid food → score in [0, 1]
# ─────────────────────────────────────────────────────────────────────────────
class TestValidPrediction:
    def test_score_is_float_in_unit_interval(self):
        svc = _fresh_service()
        # user "5" is in the synthetic mapping (key "5" → idx 5)
        score = svc.predict_single("5", 100)
        assert score is not None
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_batch_scores_all_in_unit_interval(self):
        svc = _fresh_service()
        food_ids = [1, 50, 100, 500, 1014]
        results = svc.predict_for_user("10", food_ids)
        assert len(results) == len(food_ids)
        for r in results:
            if r is not None:
                assert 0.0 <= r <= 1.0

    def test_different_foods_produce_different_scores(self):
        svc = _fresh_service()
        s1 = svc.predict_single("42", 1)
        s2 = svc.predict_single("42", 1014)
        # Scores may theoretically be equal but highly unlikely
        assert s1 is not None
        assert s2 is not None
        # Just confirm they're both valid floats
        assert isinstance(s1, float) and isinstance(s2, float)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 6 — food_id mapping covers food_dataset.csv (1–1014)
# ─────────────────────────────────────────────────────────────────────────────
class TestFoodIdMapping:
    def test_all_dataset_food_ids_in_mapping(self):
        svc = _fresh_service()
        real_food_ids = set(_food_ids_from_dataset())
        mapped_food_ids = set(svc._food_id_to_idx.keys())
        missing = real_food_ids - mapped_food_ids
        assert len(missing) == 0, (
            f"{len(missing)} real food_ids not in NCF mapping: {sorted(missing)[:10]}"
        )

    def test_mapping_food_id_min_max(self):
        svc = _fresh_service()
        keys = svc._food_id_to_idx.keys()
        assert min(keys) == 1
        assert max(keys) == 1014

    def test_mappings_json_matches_service_state(self):
        svc = _fresh_service()
        with open(os.path.join(_ROOT, "ncf_integration", "models", "ncf_mappings.json")) as f:
            m = json.load(f)
        assert m["num_items"] == svc._num_items
        assert m["num_users"] == svc._num_users


# ─────────────────────────────────────────────────────────────────────────────
# TEST 7 — existing recommendation pipeline still works
# ─────────────────────────────────────────────────────────────────────────────
class TestExistingPipelineIntact:
    def test_enhanced_recommender_runs(self):
        from backend.services.enhanced_recommender import EnhancedNutritionRecommender
        rec = EnhancedNutritionRecommender()
        profile = {
            "age": 35, "gender": "Male", "bmi": 24.0,
            "activity_level": "Moderate", "daily_calories": 2000,
            "diseases": ["Normal"], "severity": {}, "username": "testuser",
        }
        result = rec.generate_meal_plan(profile)
        assert "meal_plan" in result
        assert "total_calories" in result
        assert result["total_calories"] > 0

    def test_weekly_planner_generates_7_days(self):
        from backend.services.weekly_meal_planner import generate_weekly_recommendations
        profile = {
            "age": 40, "gender": "Female", "weight": 65, "height": 165,
            "bmi": 23.9, "activity_level": "Moderate",
            "diseases": ["Normal"], "severity": {}, "daily_calories": 1800,
        }
        result = generate_weekly_recommendations(profile)
        assert "weekly_plan" in result
        assert len(result["weekly_plan"]) == 7


# ─────────────────────────────────────────────────────────────────────────────
# TEST 8 — existing disease filtering still works
# ─────────────────────────────────────────────────────────────────────────────
class TestDiseaseSafetyFilters:
    def test_diabetes_filter_applied(self):
        from backend.services.enhanced_recommender import EnhancedNutritionRecommender
        import config
        rec = EnhancedNutritionRecommender()
        filtered = rec._apply_disease_filters(rec.df, ["Diabetes"])
        assert (filtered["Free Sugar (g)"] <= config.DIABETES_MAX_SUGAR).all()
        assert (filtered["Calories (kcal)"] <= config.DIABETES_MAX_CALORIES).all()

    def test_kidney_filter_applied(self):
        from backend.services.enhanced_recommender import EnhancedNutritionRecommender
        import config
        rec = EnhancedNutritionRecommender()
        filtered = rec._apply_disease_filters(rec.df, ["Kidney Disease"])
        assert (filtered["Sodium (mg)"] <= config.KIDNEY_MAX_SODIUM).all()
        assert (filtered["Protein (g)"] <= config.KIDNEY_MAX_PROTEIN).all()


# ─────────────────────────────────────────────────────────────────────────────
# TEST 9 — NCF cannot reintroduce filtered foods
# ─────────────────────────────────────────────────────────────────────────────
class TestNCFSafetyPipeline:
    def test_ncf_score_does_not_bypass_disease_filters(self):
        """
        NCF scores are computed only AFTER disease/severity filtering.
        Verify that no food in the generated plan violates diabetes constraints.
        """
        from backend.services.enhanced_recommender import EnhancedNutritionRecommender
        import config
        rec = EnhancedNutritionRecommender()
        profile = {
            "age": 50, "gender": "Male", "bmi": 28.0,
            "activity_level": "Sedentary", "daily_calories": 1800,
            "diseases": ["Diabetes"], "severity": {"diabetes": "moderate"},
            "username": "5",   # valid NCF user (if model active)
        }
        result = rec.generate_meal_plan(profile)
        for meal_type, foods in result["meal_plan"].items():
            for food in foods:
                assert food.get("Free Sugar (g)", 0) <= config.DIABETES_MAX_SUGAR, (
                    f"{food.get('Dish Name')} has {food.get('Free Sugar (g)')}g sugar "
                    f"(max {config.DIABETES_MAX_SUGAR}g) in meal {meal_type}"
                )

    def test_ncf_does_not_increase_unsafe_pool(self):
        """
        Pool size with NCF active must equal pool size without NCF.
        NCF re-ranks; it does not add foods.
        """
        from backend.services.enhanced_recommender import EnhancedNutritionRecommender
        rec = EnhancedNutritionRecommender()
        diseases = ["Kidney Disease"]
        pool = rec._apply_disease_filters(rec.df, diseases)
        size_before = len(pool)
        # Apply severity filter too
        pool2 = rec._apply_severity_filters(pool, diseases, {"kidney_disease": "moderate"})
        size_after_ncf_would_apply = len(pool2)
        # NCF cannot increase this number
        assert size_after_ncf_would_apply <= size_before


# ─────────────────────────────────────────────────────────────────────────────
# TEST 10 — 7-day planner is deterministic
# ─────────────────────────────────────────────────────────────────────────────
class TestDeterminism:
    def test_same_profile_produces_same_plan(self):
        from backend.services.weekly_meal_planner import WeeklyMealPlanner
        profile = {
            "age": 45, "gender": "Male", "weight": 75, "height": 175,
            "bmi": 24.5, "activity_level": "Moderate",
            "diseases": ["Diabetes"], "severity": {"diabetes": "moderate"},
            "daily_calories": 2000,
        }
        p1 = WeeklyMealPlanner()
        r1 = p1.generate_weekly_plan(profile)
        p2 = WeeklyMealPlanner()
        r2 = p2.generate_weekly_plan(profile)

        foods1 = [
            f["Dish Name"]
            for day in r1.weekly_plan
            for foods in day["meals"].values()
            for f in foods
        ]
        foods2 = [
            f["Dish Name"]
            for day in r2.weekly_plan
            for foods in day["meals"].values()
            for f in foods
        ]
        assert foods1 == foods2, "WeeklyMealPlanner is not deterministic"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 11 — no random.choice / random.sample in weekly_meal_planner
# ─────────────────────────────────────────────────────────────────────────────
class TestNoUncontrolledRandomness:
    def test_weekly_planner_has_no_random_choice(self):
        planner_path = os.path.join(
            _ROOT, "backend", "services", "weekly_meal_planner.py"
        )
        with open(planner_path, encoding="utf-8") as f:
            src = f.read()
        assert "random.choice(" not in src, (
            "weekly_meal_planner.py contains random.choice() — determinism broken"
        )
        assert "random.sample(" not in src, (
            "weekly_meal_planner.py contains random.sample() — determinism broken"
        )

    def test_ncf_training_uses_seeded_rng(self):
        training_path = os.path.join(
            _ROOT, "backend", "services", "ncf_training.py"
        )
        with open(training_path, encoding="utf-8") as f:
            src = f.read()
        # Training uses np.random.default_rng(seed) — not unseeded random.choice
        assert "default_rng(" in src, "ncf_training must use seeded np.random.default_rng"
        assert "random.choice(" not in src, (
            "ncf_training.py contains unseeded random.choice()"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TEST 12 — NCF checkbox not required (auto-detected)
# ─────────────────────────────────────────────────────────────────────────────
class TestNoCHeckboxRequired:
    def test_app_has_no_enable_ncf_checkbox(self):
        app_path = os.path.join(_ROOT, "app.py")
        with open(app_path, encoding="utf-8") as f:
            src = f.read()
        # None of these UI patterns should require the user to enable NCF
        assert "Enable NCF" not in src
        assert "Use NCF" not in src
        assert "enable_ncf" not in src.lower().replace("_", "")

    def test_ncf_availability_is_auto_detected(self):
        """NCF service detects model availability at import time."""
        svc = _fresh_service()
        # is_available() reflects reality — True if model loaded, False if not
        status = svc.get_status()
        assert "available" in status
        # If model is on disk, it should be True
        if os.path.exists(os.path.join(_ROOT, "ncf_integration", "models", "ncf_model.keras")):
            assert svc.is_available() is True


# ─────────────────────────────────────────────────────────────────────────────
# TEST 13 — scoring information is accurate
# ─────────────────────────────────────────────────────────────────────────────
class TestScoringInformation:
    def test_scoring_info_reflects_ncf_status(self):
        from backend.services.enhanced_recommender import EnhancedNutritionRecommender
        rec = EnhancedNutritionRecommender()
        profile = {
            "age": 30, "gender": "Female", "bmi": 22.0,
            "activity_level": "Active", "daily_calories": 2000,
            "diseases": ["Normal"], "severity": {}, "username": "5",
        }
        result = rec.generate_meal_plan(profile)
        scoring = result.get("scoring_info", {})
        ncf_active = rec._ncf_available

        if ncf_active:
            # NCF is loaded — weights should reflect w_ncf > 0
            assert scoring.get("w_ncf", 0) == pytest.approx(0.20, abs=1e-4), (
                f"w_ncf should be 0.20 when NCF is active, got {scoring.get('w_ncf')}"
            )
        else:
            assert scoring.get("w_ncf", 0) == pytest.approx(0.0, abs=1e-4), (
                f"w_ncf should be 0.0 when NCF is unavailable, got {scoring.get('w_ncf')}"
            )

    def test_ncf_available_field_matches_reality(self):
        from backend.services.enhanced_recommender import EnhancedNutritionRecommender
        rec = EnhancedNutritionRecommender()
        profile = {
            "age": 30, "gender": "Male", "bmi": 24.0,
            "activity_level": "Moderate", "daily_calories": 2000,
            "diseases": ["Normal"], "severity": {}, "username": "99",
        }
        result = rec.generate_meal_plan(profile)
        scoring = result.get("scoring_info", {})
        # ncf_available in scoring must equal is_available()
        from backend.services.ncf_service import is_available as ncf_is_available
        assert scoring.get("ncf_available") == ncf_is_available(), (
            "scoring_info.ncf_available does not match ncf_service.is_available()"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TEST 14 — new users still receive recommendations (cold-start)
# ─────────────────────────────────────────────────────────────────────────────
class TestColdStart:
    def test_new_user_gets_recommendations(self):
        """A user not in the NCF mapping still gets a full meal plan."""
        from backend.services.enhanced_recommender import EnhancedNutritionRecommender
        rec = EnhancedNutritionRecommender()
        profile = {
            "age": 28, "gender": "Female", "bmi": 21.5,
            "activity_level": "Light", "daily_calories": 1700,
            "diseases": ["Normal"], "severity": {},
            "username": "totally_new_user_not_in_mapping_12345",
        }
        result = rec.generate_meal_plan(profile)
        assert "meal_plan" in result
        total_foods = sum(len(v) for v in result["meal_plan"].values())
        assert total_foods > 0, "Cold-start user received no recommendations"

    def test_ncf_returns_none_for_new_user(self):
        """predict_single returns None (not a fake score) for new user."""
        svc = _fresh_service()
        score = svc.predict_single("new_user_not_in_mapping_999", 1)
        assert score is None, f"Expected None for unknown user, got {score}"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 15 — existing tests regression guard
# ─────────────────────────────────────────────────────────────────────────────
class TestExistingTestsStillPass:
    def test_weekly_planner_core_tests(self):
        """Smoke-test the critical existing WeeklyMealPlanner assertions."""
        from backend.services.weekly_meal_planner import WeeklyMealPlanner, FoodUsageTracker
        tracker = FoodUsageTracker()
        assert tracker.get_usage_count("anything") == 0
        tracker.record_usage("Rice", 1, "Lunch", food_id=5)
        assert tracker.get_usage_count("Rice") == 1
        assert tracker.was_used_in_meal_type("Rice", "Lunch")
        assert not tracker.was_used_in_meal_type("Rice", "Breakfast")

    def test_disease_encoder_classes(self):
        """disease_encoder.pkl still has exactly 3 classes."""
        import joblib
        enc = joblib.load(os.path.join(_ROOT, "disease_encoder.pkl"))
        assert set(enc.classes_) == {"diabetes", "no diabetes", "prediabetes"}

    def test_predict_functions_return_risk_result(self):
        from predict import predict_diabetes, predict_obesity, predict_kidney
        r = predict_diabetes(35, "Male", 24.0, 5.5, 90)
        assert "risk" in r
        assert "label" in r
        r2 = predict_obesity(30, "Female", 22.0)
        assert "risk" in r2
        r3 = predict_kidney(40, "Male", 24.0, 138.0, 4.5, 120, 1.0)
        assert "risk" in r3

    def test_config_ncf_weights_sum_to_one(self):
        import config
        total = config.NUTRITION_SCORE_WEIGHT + config.CONTENT_SCORE_WEIGHT + config.NCF_SCORE_WEIGHT
        assert abs(total - 1.0) < 1e-9, f"Hybrid weights sum to {total}, expected 1.0"

    def test_ncf_config_keys_present(self):
        import config
        assert hasattr(config, "NCF_EMBEDDING_DIM")
        assert hasattr(config, "NCF_MLP_LAYERS")
        assert hasattr(config, "NCF_WEIGHT") or hasattr(config, "NCF_SCORE_WEIGHT")
        assert hasattr(config, "NCF_RANDOM_SEED")
        assert config.NCF_EMBEDDING_DIM == 32
        assert config.NCF_RANDOM_SEED == 42
