import os
import random
import logging
from typing import List, Dict, Any
import pandas as pd

from .nutrition_engine import generate_nutrition_recommendations

logger = logging.getLogger(__name__)

# Path to the Indian food dataset (CSV). Adjust if needed.
DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "indian_food.csv")


def _load_food_dataset() -> pd.DataFrame:
    """Load the Indian food dataset.
    Expected columns: food_name, calories, protein, carbs, fat, fiber, sodium, cuisine
    """
    if not os.path.exists(DATASET_PATH):
        logger.error(f"Food dataset not found at {DATASET_PATH}")
        raise FileNotFoundError(f"Food dataset not found at {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    # Ensure proper column names (case‑insensitive handling)
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"food_name", "calories", "protein", "carbs", "fat", "fiber", "sodium", "cuisine"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    return df


def _filter_by_region(df: pd.DataFrame, region: str) -> pd.DataFrame:
    """Filter foods based on the specified Indian region.

    The function uses a mapping of region names to possible cuisine tags that may appear
    in the dataset (e.g., "tamil", "tamil nadu", "kerala", etc.). It always includes
    generic "indian" and "all" tags to keep nationwide items.
    """
    region = region.lower()
    # Map user‑friendly region names to cuisine identifiers used in the dataset.
    region_map = {
        "tamil nadu": ["tamil", "tamil nadu"],
        "kerala": ["kerala"],
        "karnataka": ["karnataka", "karnataka cuisine"],
        "andhra pradesh": ["andhra", "andhra pradesh"],
        "telangana": ["telangana"],
        "punjab": ["punjab"],
        "gujarat": ["gujarat"],
        "maharashtra": ["maharashtra"],
        "west bengal": ["bengal", "west bengal", "bangla"],
        "north east": ["north east", "northeast", "assam", "naga", "sikkim"],
    }
    # Retrieve synonyms for the requested region; fall back to the raw region string.
    cuisine_tags = region_map.get(region, [region])
    # Always allow generic Indian foods.
    allowed = set(["indian", "all"] + cuisine_tags)
    return df[df["cuisine"].str.lower().isin(allowed)].reset_index(drop=True)


def _filter_by_restrictions(df: pd.DataFrame, avoid_foods: List[str]) -> pd.DataFrame:
    """Remove foods that appear in the avoidance list (case‑insensitive)."""
    avoid_set = {f.lower() for f in avoid_foods}
    mask = ~df["food_name"].str.lower().isin(avoid_set)
    return df[mask].reset_index(drop=True)


def _select_foods_for_target(df: pd.DataFrame, target_cal: float, max_items: int = 4) -> List[Dict[str, Any]]:
    """Randomly select foods whose summed calories are close to *target_cal*.
    The algorithm picks up to *max_items* foods, shuffling the list until the
    cumulative calories exceed the target (or the list is exhausted).
    """
    if df.empty:
        raise ValueError("No foods available after filtering.")
    candidates = df.sample(frac=1).reset_index(drop=True)  # shuffled view
    selected = []
    total = 0.0
    for _, row in candidates.iterrows():
        if len(selected) >= max_items:
            break
        if total + row["calories"] > target_cal * 1.2:  # allow 20 % overshoot
            continue
        selected.append({
            "food_name": row["food_name"],
            "calories": row["calories"],
            "protein": row["protein"],
            "carbs": row["carbs"],
            "fat": row["fat"],
            "fiber": row["fiber"],
            "sodium": row["sodium"],
        })
        total += row["calories"]
        if total >= target_cal:
            break
    # If we couldn't reach the target, pad with the smallest‑calorie food
    if total < target_cal:
        smallest = df.nsmallest(1, "calories").iloc[0]
        selected.append({
            "food_name": smallest["food_name"],
            "calories": smallest["calories"],
            "protein": smallest["protein"],
            "carbs": smallest["carbs"],
            "fat": smallest["fat"],
            "fiber": smallest["fiber"],
            "sodium": smallest["sodium"],
        })
    return selected


def _aggregate_daily_nutrition(meals: Dict[str, List[Dict[str, Any]]]) -> Dict[str, float]:
    totals = {"calories": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0, "fiber": 0.0, "sodium": 0.0}
    for items in meals.values():
        for food in items:
            totals["calories"] += food.get("calories", 0)
            totals["protein"] += food.get("protein", 0)
            totals["carbs"] += food.get("carbs", 0)
            totals["fat"] += food.get("fat", 0)
            totals["fiber"] += food.get("fiber", 0)
            totals["sodium"] += food.get("sodium", 0)
    return {k: round(v, 2) for k, v in totals.items()}


def generate_weekly_meal_plan(
    diseases: List[str],
    severity: Dict[str, str],
    bmi: float,
    target_calories: float,
    region: str,
    age: int,
    gender: str,
    activity_level: str,
    medical_params: Dict[str, Any],
) -> Dict[str, Any]:
    """Create a professional weekly meal plan.

    The function respects disease‑specific food avoidances (via the nutrition
    engine), prefers foods from the requested Indian region, and aims to meet the
    macro‑nutrient targets based on *target_calories*.

    Returns a mapping of weekdays to meals and a daily nutrition summary.
    """
    # 1️⃣ Get disease‑specific recommendations (we only need the avoid list).
    nutrition = generate_nutrition_recommendations(
        diseases=diseases,
        severity=severity,
        bmi=bmi,
        calories=target_calories,
        region=region,
        age=age,
        gender=gender,
        activity_level=activity_level,
        medical_params=medical_params,
    )
    avoid_foods = nutrition.get("foods_to_avoid", [])

    # 2️⃣ Load and filter the food dataset.
    df = _load_food_dataset()
    df = _filter_by_region(df, region)
    df = _filter_by_restrictions(df, avoid_foods)

    # 3️⃣ Define calorie distribution per meal.
    distribution = {
        "Breakfast": 0.25,
        "Morning Snack": 0.075,
        "Lunch": 0.30,
        "Evening Snack": 0.075,
        "Dinner": 0.30,
    }

    week_plan = {}
    for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
        daily_meals: Dict[str, List[Dict[str, Any]]] = {}
        for meal, pct in distribution.items():
            target = target_calories * pct
            try:
                foods = _select_foods_for_target(df, target)
            except Exception as e:
                logger.exception(f"Failed to select foods for {day} {meal}: {e}")
                foods = []
            daily_meals[meal] = foods
        # Aggregate daily nutrition for verification.
        daily_totals = _aggregate_daily_nutrition(daily_meals)
        week_plan[day] = {
            "meals": daily_meals,
            "daily_totals": daily_totals,
        }
    return week_plan
