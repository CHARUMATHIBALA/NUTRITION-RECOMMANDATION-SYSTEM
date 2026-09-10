from recommendation import IntelligentNutritionRecommender
import sys
import os
import pandas as pd

# Add backend services to path for improved meal planner
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'services'))

try:
    from backend.services.severity_rules import build_severity_dict
    SEVERITY_RULES_AVAILABLE = True
except ImportError:
    SEVERITY_RULES_AVAILABLE = False

try:
    from improved_meal_planner import ImprovedMealPlanner
    IMPROVED_PLANNER_AVAILABLE = True
except ImportError:
    IMPROVED_PLANNER_AVAILABLE = False

try:
    from enhanced_recommender import EnhancedNutritionRecommender, generate_enhanced_recommendations
    ENHANCED_RECOMMENDER_AVAILABLE = True
except ImportError:
    ENHANCED_RECOMMENDER_AVAILABLE = False

try:
    from recommendation_service import get_recommendation_service
    UNIFIED_SERVICE_AVAILABLE = True
except ImportError:
    UNIFIED_SERVICE_AVAILABLE = False


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
    creatinine,
    goal=None,
    region=None,
    severity=None,          # dict: {disease_key: severity_level | None}
                            # built from RiskResult.final_status by app.py
                            # e.g. {'diabetes': 'severe', 'kidney_disease': None, 'obesity': 'moderate'}
):
    """
    Generate comprehensive nutrition recommendations including:
    - Meal plans for Breakfast, Lunch, Snack, Dinner
    - Foods to avoid
    - Daily water intake
    - Daily protein requirement
    - Nutrition tips

    The `severity` parameter (optional dict) enables severity-aware filtering
    and scoring in the enhanced recommender pipeline.  When absent the
    enhanced recommender falls back to config.SEVERITY_DEFAULT per disease.
    """
    # Handle single disease string
    if isinstance(diseases, str):
        diseases = [diseases]
    
    # ── Primary: Enhanced recommender (hybrid_score pipeline) ───────────
    # This path MUST run first because it is the only one that computes
    # hybrid_score (nutrition × severity-suitability blend + content score)
    # and ranks candidates by that score before top-N selection.
    # The unified RecommendationService below uses a simpler score that
    # ignores content and severity suitability, so it must only be used
    # as a fallback when the enhanced recommender is unavailable or fails.
    if ENHANCED_RECOMMENDER_AVAILABLE:
        try:
            print("Using enhanced recommender (hybrid scoring)...")
            user_profile = {
                'diseases': diseases,
                'age': age,
                'gender': gender,
                'height': height,
                'weight': weight,
                'bmi': bmi,
                'activity_level': activity_level,
                'daily_calories': daily_calories,
                'hba1c': hba1c,
                'glucose': glucose,
                'bp': bp,
                'sodium': sodium,
                'potassium': potassium,
                'creatinine': creatinine,
                'goal': goal or 'Weight Loss',
                'region': region or 'Andhra Pradesh',
                'severity': severity or {},  # pass severity dict into recommender
            }
            result = generate_enhanced_recommendations(user_profile)

            # Convert pd.Series food items to plain dicts for the UI
            meal_plan = {}
            for meal_type, foods in result['meal_plan'].items():
                meal_plan[meal_type] = [f.to_dict() if isinstance(f, pd.Series) else f for f in foods]

            # ── Compute foods_to_avoid / water / protein / tips ───────────
            # These are not produced by the enhanced recommender itself.
            # Use the existing IntelligentNutritionRecommender helpers which
            # are already tested and contain disease-specific logic.
            try:
                from recommendation import IntelligentNutritionRecommender as _INR
                _rec = _INR()
                _foods_to_avoid      = _rec.get_foods_to_avoid(diseases)
                _water_intake        = _rec.calculate_water_intake(weight, activity_level)
                _protein_requirement = _rec.calculate_protein_requirement(weight, gender, age, diseases)
                _nutrition_tips      = _rec.get_nutrition_tips(diseases, bmi, activity_level)
            except Exception:
                _foods_to_avoid      = []
                _water_intake        = round(weight * 0.033, 2)
                _protein_requirement = round(weight * 0.8,   1)
                _nutrition_tips      = []

            enhanced_result = {
                'meal_plan': meal_plan,
                'foods_to_avoid':      _foods_to_avoid,
                'water_intake':        _water_intake,
                'protein_requirement': _protein_requirement,
                'nutrition_tips':      _nutrition_tips,
                'meal_validation': {'is_valid': True, 'warnings': [], 'errors': []},
                'use_enhanced_recommender': True,
                'total_calories': result.get('total_calories', 0),
                'target_calories': result.get('target_calories', daily_calories),
                'calorie_difference': result.get('calorie_difference', 0),
                'nutrition_summary': result.get('nutrition_summary', {}),
                'meal_explanations': result.get('meal_explanations', {}),
                'severity_info': result.get('severity_info', {}),
                'scoring_info': result.get('scoring_info', {}),
                'debug_info': result.get('debug_info', {})
            }
            print(f"Enhanced recommender result: meal plan with {len(enhanced_result['meal_plan'])} meals")
            return enhanced_result
        except Exception as e:
            print(f"Enhanced recommender failed: {e}, falling back to unified service")
            import traceback
            traceback.print_exc()

    # ── Fallback: unified recommendation service (simple nutrition score) ─
    if UNIFIED_SERVICE_AVAILABLE:
        try:
            print("Using unified recommendation service (fallback)...")
            service = get_recommendation_service()
            result = service.generate_recommendations(
                diseases=diseases,
                age=age,
                gender=gender,
                height=height,
                weight=weight,
                bmi=bmi,
                activity_level=activity_level,
                daily_calories=daily_calories,
                hba1c=hba1c,
                glucose=glucose,
                bp=bp,
                sodium=sodium,
                potassium=potassium,
                creatinine=creatinine,
                goal=goal,
                region=region
            )
            print(f"Unified service result: meal plan with {len(result['meal_plan'])} meals")
            return result
        except Exception as e:
            print(f"Unified service failed: {e}, using fallback")
            import traceback
            traceback.print_exc()
    
    # Fallback to original system
    recommender = IntelligentNutritionRecommender()
    
    if IMPROVED_PLANNER_AVAILABLE:
        try:
            planner = ImprovedMealPlanner()
            result = planner.generate_daily_meal_plan(diseases, daily_calories)
            
            meal_plan = {}
            for meal_type, meal_data in result['meal_plan'].items():
                foods_list = []
                if meal_data['main'] is not None:
                    foods_list.append(meal_data['main'].to_dict())
                if meal_data['protein'] is not None:
                    foods_list.append(meal_data['protein'].to_dict())
                if meal_data['vegetable'] is not None:
                    foods_list.append(meal_data['vegetable'].to_dict())
                meal_plan[meal_type] = foods_list if foods_list else []
            
            validation = result['validation']
        except Exception as e:
            print(f"Improved meal planner failed: {e}, using fallback")
            meal_plan = _generate_fallback_meal_plan(recommender, diseases)
            validation = {'is_valid': True, 'warnings': [], 'errors': []}
    else:
        meal_plan = _generate_fallback_meal_plan(recommender, diseases)
        validation = {'is_valid': True, 'warnings': [], 'errors': []}
    
    foods_to_avoid = recommender.get_foods_to_avoid(diseases)
    water_intake = recommender.calculate_water_intake(weight, activity_level)
    protein_requirement = recommender.calculate_protein_requirement(weight, gender, age, diseases)
    nutrition_tips = recommender.get_nutrition_tips(diseases, bmi, activity_level)
    
    return {
        'meal_plan': meal_plan,
        'foods_to_avoid': foods_to_avoid,
        'water_intake': water_intake,
        'protein_requirement': protein_requirement,
        'nutrition_tips': nutrition_tips,
        'meal_validation': validation,
        'use_improved_planner': IMPROVED_PLANNER_AVAILABLE,
        'use_enhanced_recommender': False,
        'use_unified_service': False
    }


def _generate_fallback_meal_plan(recommender, diseases):
    """Generate meal plan using original system as fallback."""
    meal_types = ["Breakfast", "Lunch", "Snack", "Dinner"]
    meal_plan = {}
    selected_foods = set()  # Track foods already selected to avoid repetition
    
    for meal in meal_types:
        foods = recommender.recommend_food(
            diseases=diseases,
            meal_type=meal,
            top_n=10  # Get more candidates then filter out duplicates
        )
        # Convert DataFrame to list of dicts for consistency
        if not foods.empty:
            foods_list = foods.to_dict('records')
            # Filter out already selected foods
            filtered_foods = [f for f in foods_list if f['Dish Name'] not in selected_foods]
            # Take top 5 unique foods
            meal_plan[meal] = filtered_foods[:5]
            # Add selected foods to the set
            for f in meal_plan[meal]:
                selected_foods.add(f['Dish Name'])
        else:
            meal_plan[meal] = []
    
    return meal_plan


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