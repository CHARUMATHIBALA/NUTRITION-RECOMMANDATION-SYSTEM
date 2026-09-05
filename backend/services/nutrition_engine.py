import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

import config
def _calculate_water_intake(weight_kg: float) -> float:
    """Simple heuristic: factor per kg of body weight, capped at max intake."""
    intake = weight_kg * config.WATER_INTAKE_FACTOR
    # Apply maximum water intake ceiling (liters)
    intake = min(intake, config.MAX_WATER_INTAKE_L)
    return round(intake, 2)


def _calculate_protein_requirement(weight_kg: float) -> float:
    """Baseline protein: 0.8 g per kg of body weight, capped at max protein limit (e.g., kidney disease)."""
    protein = weight_kg * config.PROTEIN_PER_KG.get("default", 0.8)
    # Apply hard ceiling for protein intake (e.g., kidney disease limit)
    protein = min(protein, config.KIDNEY_MAX_PROTEIN)
    return round(protein, 2)


def _macro_distribution(calories: float, protein_g: float) -> Dict[str, float]:
    """Distribute calories into carbs, protein, fat.
    Protein is fixed, remaining calories split 50 % carbs / 50 % fat.
    Returns grams of each macronutrient.
    """
    protein_cal = protein_g * 4
    remaining_cal = max(calories - protein_cal, 0)
    carbs_cal = remaining_cal * 0.5
    fat_cal = remaining_cal * 0.5
    return {
        "carbs_g": round(carbs_cal / 4, 2),
        "protein_g": protein_g,
        "fat_g": round(fat_cal / 9, 2),
    }


def _select_foods(diseases: List[str], region: str) -> Dict[str, List[str]]:
    """Very naive food selector based on disease and region.
    In a real system this would query a nutritional database.
    """
    base_foods = {
        "Diabetes": ["Whole grains", "Leafy greens", "Berries", "Nuts"],
        "Obesity": ["High‑protein foods", "Fiber‑rich vegetables", "Low‑fat dairy"],
        "Kidney Disease": ["Low‑potassium fruits", "White rice", "Egg whites"],
        "Normal": ["Balanced meals", "Seasonal fruits", "Lean proteins"],
    }
    foods = []
    for d in diseases:
        foods.extend(base_foods.get(d, []))
    # Simple region filter – placeholder
    if region.lower() in ["tamil nadu", "kerala"]:
        foods.append("Coconut water")
    elif region.lower() in ["maharashtra", "gujarat"]:
        foods.append("Buttermilk")
    return {"recommended": list(set(foods))}


def _avoid_foods(diseases: List[str]) -> List[str]:
    avoid_map = {
        "Diabetes": ["Sugary drinks", "Refined carbs", "Sweets"],
        "Obesity": ["Fried foods", "Processed snacks", "Sugary beverages"],
        "Kidney Disease": ["High‑potassium fruits", "Processed meats", "Dairy (high‑phosphate)"],
    }
    avoids = []
    for d in diseases:
        avoids.extend(avoid_map.get(d, []))
    return list(set(avoids))


def _vitamin_mineral_suggestions(diseases: List[str]) -> Dict[str, List[str]]:
    suggestions = {"vitamins": [], "minerals": []}
    for d in diseases:
        if d == "Diabetes":
            suggestions["vitamins"].extend(["Vitamin D", "Vitamin B12"])
            suggestions["minerals"].extend(["Magnesium", "Chromium"])
        if d == "Kidney Disease":
            suggestions["vitamins"].extend(["Vitamin B6", "Folate"])
            suggestions["minerals"].extend(["Iron", "Zinc"])
    # De‑duplicate
    suggestions["vitamins"] = list(set(suggestions["vitamins"]))
    suggestions["minerals"] = list(set(suggestions["minerals"]))
    return suggestions


def _nutrition_tips(diseases: List[str]) -> List[str]:
    tips = []
    for d in diseases:
        if d == "Diabetes":
            tips.append("Include low‑glycemic index foods.")
            tips.append("Space carbohydrate intake throughout the day.")
        if d == "Obesity":
            tips.append("Prioritize protein at each meal to increase satiety.")
            tips.append("Practice mindful eating and portion control.")
        if d == "Kidney Disease":
            tips.append("Limit sodium intake to manage blood pressure.")
            tips.append("Stay hydrated but monitor potassium levels.")
    if not tips:
        tips.append("Maintain a balanced diet with varied foods.")
    return tips


def _score_recommendations(food_list: List[str], diseases: List[str]) -> List[Dict[str, Any]]:
    """Assign a simple nutritional score (0‑100) based on disease relevance.
    This placeholder scores foods that appear in disease‑specific lists higher.
    """
    scores = []
    disease_food_map = {
        "Diabetes": {"Whole grains", "Leafy greens", "Berries", "Nuts"},
        "Obesity": {"High‑protein foods", "Fiber‑rich vegetables", "Low‑fat dairy"},
        "Kidney Disease": {"Low‑potassium fruits", "White rice", "Egg whites"},
    }
    for food in food_list:
        relevance = 0
        for d in diseases:
            if food in disease_food_map.get(d, set()):
                relevance += 1
        score = min(100, (relevance / max(len(diseases), 1)) * 100)
        scores.append({"food": food, "score": round(score, 1)})
    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores


def generate_nutrition_recommendations(
    diseases: List[str],
    severity: Dict[str, str],
    bmi: float,
    calories: float,
    region: str,
    age: int,
    gender: str,
    activity_level: str,
    medical_params: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate a structured nutrition recommendation.

    Parameters
    ----------
    diseases: List of disease names (e.g., ["Diabetes", "Obesity"]).
    severity: Mapping from disease name to severity level (e.g., {"Diabetes": "moderate"}).
    bmi: Body‑Mass Index.
    calories: Target daily caloric intake.
    region: Patient's geographical region for cultural food suggestions.
    age, gender, activity_level: Basic demographic/behavioral data.
    medical_params: Dictionary of additional medical values (e.g., HbA1c, blood pressure).

    Returns
    -------
    dict with keys:
        * personalized_foods – list of recommended foods
        * foods_to_avoid – list of foods to avoid
        * daily_calories – float (same as input for now)
        * macronutrients – dict with carb/protein/fat grams
        * water_intake – liters per day
        * vitamin_suggestions – list of vitamins
        * mineral_suggestions – list of minerals
        * nutrition_tips – list of tip strings
        * ranked_recommendations – list of dicts {food, score}
    """
    try:
        weight_kg = medical_params.get("weight_kg", 70)
        water = _calculate_water_intake(weight_kg)
        protein = _calculate_protein_requirement(weight_kg)
        macros = _macro_distribution(calories, protein)
        foods_dict = _select_foods(diseases, region)
        recommended = foods_dict["recommended"]
        avoid = _avoid_foods(diseases)
        vitamin_mineral = _vitamin_mineral_suggestions(diseases)
        tips = _nutrition_tips(diseases)
        ranked = _score_recommendations(recommended, diseases)
        return {
            "personalized_foods": recommended,
            "foods_to_avoid": avoid,
            "daily_calories": round(calories, 2),
            "macronutrients": macros,
            "water_intake": water,
            "vitamin_suggestions": vitamin_mineral["vitamins"],
            "mineral_suggestions": vitamin_mineral["minerals"],
            "nutrition_tips": tips,
            "ranked_recommendations": ranked,
        }
    except Exception as e:
        logger.exception("Nutrition recommendation generation failed: %s", e)
        raise
