from recommendation import IntelligentNutritionRecommender


def generate_comprehensive_recommendations(
    diseases,
    age,
    gender,
    height,
    weight,
    bmi,
    activity_level,
    daily_calories,
    hba1c,
    glucose,
    bp,
    sodium,
    potassium,
    creatinine
):
    """
    Generate comprehensive nutrition recommendations including:
    - Meal plans for Breakfast, Lunch, Snack, Dinner
    - Foods to avoid
    - Daily water intake
    - Daily protein requirement
    - Nutrition tips
    """
    recommender = IntelligentNutritionRecommender()
    
    # Handle single disease string
    if isinstance(diseases, str):
        diseases = [diseases]
    
    # Generate meal plans
    meal_types = ["Breakfast", "Lunch", "Snack", "Dinner"]
    meal_plan = {}
    
    for meal in meal_types:
        foods = recommender.recommend_food(
            diseases=diseases,
            meal_type=meal,
            top_n=5
        )
        meal_plan[meal] = foods
    
    # Get foods to avoid
    foods_to_avoid = recommender.get_foods_to_avoid(diseases)
    
    # Calculate water intake
    water_intake = recommender.calculate_water_intake(weight, activity_level)
    
    # Calculate protein requirement
    protein_requirement = recommender.calculate_protein_requirement(
        weight, gender, age, diseases
    )
    
    # Get nutrition tips
    nutrition_tips = recommender.get_nutrition_tips(diseases, bmi, activity_level)
    
    return {
        'meal_plan': meal_plan,
        'foods_to_avoid': foods_to_avoid,
        'water_intake': water_intake,
        'protein_requirement': protein_requirement,
        'nutrition_tips': nutrition_tips
    }


def generate_meal_plan(disease):
    """
    Legacy function for backward compatibility.
    Use generate_comprehensive_recommendations for full functionality.
    """
    recommender = IntelligentNutritionRecommender()
    
    meal_types = ["Breakfast", "Lunch", "Snack", "Dinner"]
    meal_plan = {}
    
    for meal in meal_types:
        foods = recommender.recommend_food(
            diseases=disease,
            meal_type=meal,
            top_n=7
        )
        meal_plan[meal] = foods
    
    return meal_plan