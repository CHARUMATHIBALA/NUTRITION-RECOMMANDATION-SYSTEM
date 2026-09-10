"""
Test app imports and weekly planner functionality
"""

import sys
sys.path.insert(0, '.')

print("Testing imports...")

try:
    from backend.services.weekly_meal_planner import generate_weekly_recommendations
    print('Weekly planner import: SUCCESS')
except Exception as e:
    print(f'Weekly planner import FAILED: {e}')

try:
    from backend.services.recommendation_evaluator import evaluate_recommendations
    print('Evaluator import: SUCCESS')
except Exception as e:
    print(f'Evaluator import FAILED: {e}')

try:
    from backend.services.enhanced_recommender import EnhancedNutritionRecommender
    print('Enhanced recommender import: SUCCESS')
except Exception as e:
    print(f'Enhanced recommender import FAILED: {e}')

# Test the function
test_profile = {
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

print("\nTesting weekly planner function...")
try:
    result = generate_weekly_recommendations(test_profile)
    print('Function call: SUCCESS')
    unique_foods = result['diversity_metrics']['unique_foods']
    print(f'Unique foods: {unique_foods}')
except Exception as e:
    print(f'Function call FAILED: {e}')
    import traceback
    traceback.print_exc()

print("\nAll tests completed!")