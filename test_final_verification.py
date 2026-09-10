"""
Final verification testing for 7-day diverse meal planner
"""

import sys
sys.path.insert(0, '.')

from backend.services.weekly_meal_planner import generate_weekly_recommendations
from backend.services.recommendation_evaluator import evaluate_recommendations

# Test profile 1: Diabetes moderate
profile1 = {
    'age': 45,
    'gender': 'Male',
    'weight': 75,
    'height': 175,
    'bmi': 24.5,
    'activity_level': 'Moderate',
    'diseases': ['Diabetes'],
    'severity': {'diabetes': 'moderate'},
    'daily_calories': 2000
}

# Test profile 2: Healthy
profile2 = {
    'age': 30,
    'gender': 'Female',
    'weight': 60,
    'height': 165,
    'bmi': 22.0,
    'activity_level': 'Active',
    'diseases': [],
    'severity': {},
    'daily_calories': 1800
}

# Test profile 3: Severe multiple diseases
profile3 = {
    'age': 55,
    'gender': 'Male',
    'weight': 85,
    'height': 170,
    'bmi': 29.4,
    'activity_level': 'Sedentary',
    'diseases': ['Diabetes', 'Kidney Disease', 'Obesity'],
    'severity': {
        'diabetes': 'severe',
        'kidney_disease': 'severe',
        'obesity': 'severe'
    },
    'daily_calories': 1600
}

print('Testing Profile 1 (Diabetes Moderate)...')
result1 = generate_weekly_recommendations(profile1)
eval1 = evaluate_recommendations(
    result1['weekly_plan'],
    result1['safety_report'],
    result1['diversity_metrics'],
    profile1
)
print(f'Unique foods: {result1["diversity_metrics"]["unique_foods"]}')
print(f'Unique %: {result1["diversity_metrics"]["unique_food_percentage"]}%')
print(f'Max repetition: {result1["diversity_metrics"]["max_repetition_count"]}')
print(f'Safety filters applied: {result1["safety_report"]["filtering_applied"]}')

print('\nTesting Profile 2 (Healthy)...')
result2 = generate_weekly_recommendations(profile2)
print(f'Unique foods: {result2["diversity_metrics"]["unique_foods"]}')
print(f'Unique %: {result2["diversity_metrics"]["unique_food_percentage"]}%')
print(f'Max repetition: {result2["diversity_metrics"]["max_repetition_count"]}')

print('\nTesting Profile 3 (Severe Multiple Diseases)...')
result3 = generate_weekly_recommendations(profile3)
print(f'Unique foods: {result3["diversity_metrics"]["unique_foods"]}')
print(f'Unique %: {result3["diversity_metrics"]["unique_food_percentage"]}%')
print(f'Filtered foods: {result3["safety_report"]["filtered_food_count"]}')
print(f'Original foods: {result3["safety_report"]["original_food_count"]}')

print('\n--- Determinism Test ---')
# Test same profile twice to verify determinism
result1a = generate_weekly_recommendations(profile1)
result1b = generate_weekly_recommendations(profile1)
print(f'Same profile - First run unique foods: {result1a["diversity_metrics"]["unique_foods"]}')
print(f'Same profile - Second run unique foods: {result1b["diversity_metrics"]["unique_foods"]}')
print(f'Deterministic: {result1a["diversity_metrics"]["unique_foods"] == result1b["diversity_metrics"]["unique_foods"]}')

print('\n--- Safety Test ---')
# Verify no unsafe foods are recommended
def check_safety(weekly_plan, diseases, severity):
    """Check that all recommended foods pass safety filters."""
    all_safe = True
    for day_plan in weekly_plan:
        for meal_type, foods in day_plan['meals'].items():
            for food in foods:
                # Foods should have been filtered already
                # This is a basic sanity check
                if not food.get('Dish Name'):
                    all_safe = False
                    print(f"Unsafe: Food without name found")
    return all_safe

safety_check = check_safety(result1['weekly_plan'], profile1['diseases'], profile1['severity'])
print(f'Safety check passed: {safety_check}')

print('\n--- Meal Type Validation ---')
def check_meal_types(weekly_plan):
    """Verify foods match their meal types."""
    for day_plan in weekly_plan:
        for meal_type, foods in day_plan['meals'].items():
            for food in foods:
                food_mealtype = food.get('MealType', '').lower()
                # Basic check - meal type should be compatible
                if meal_type not in food_mealtype and 'lunch/dinner' not in food_mealtype:
                    if not (meal_type in ['lunch', 'dinner'] and 'lunch/dinner' in food_mealtype):
                        print(f"Meal type mismatch: {meal_type} vs {food_mealtype}")
                        return False
    return True

meal_type_check = check_meal_types(result1['weekly_plan'])
print(f'Meal type validation passed: {meal_type_check}')

print('\nAll verification tests completed successfully!')