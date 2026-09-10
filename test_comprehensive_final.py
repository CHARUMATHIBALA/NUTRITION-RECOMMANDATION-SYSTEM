"""
Final comprehensive test for the 7-day meal planner integration
"""

import sys
sys.path.insert(0, '.')

print('=== FINAL COMPREHENSIVE TEST ===\n')

# Test 1: Check all imports
print('Test 1: Import checks')
try:
    from backend.services.weekly_meal_planner import generate_weekly_recommendations
    print('  [OK] Weekly planner import')
except Exception as e:
    print(f'  [FAIL] Weekly planner: {e}')

try:
    from backend.services.recommendation_evaluator import evaluate_recommendations
    print('  [OK] Evaluator import')
except Exception as e:
    print(f'  [FAIL] Evaluator: {e}')

try:
    from backend.services.enhanced_recommender import EnhancedNutritionRecommender
    print('  [OK] Enhanced recommender import')
except Exception as e:
    print(f'  [FAIL] Enhanced recommender: {e}')

# Test 2: Weekly planner functionality
print('\nTest 2: Weekly planner functionality')
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

try:
    result = generate_weekly_recommendations(test_profile)
    unique_foods = result['diversity_metrics']['unique_foods']
    print(f'  [OK] Generated plan with {unique_foods} unique foods')
except Exception as e:
    print(f'  [FAIL] {e}')
    import traceback
    traceback.print_exc()

# Test 3: Evaluator functionality
print('\nTest 3: Evaluator functionality')
try:
    result = generate_weekly_recommendations(test_profile)
    evaluation = evaluate_recommendations(
        result['weekly_plan'],
        result['safety_report'],
        result['diversity_metrics'],
        test_profile
    )
    print('  [OK] Evaluation completed')
    safety_rate = evaluation['safety_metrics']['safety_filter_rate']
    print(f'  [OK] Safety filter rate: {safety_rate:.2%}')
except Exception as e:
    print(f'  [FAIL] {e}')
    import traceback
    traceback.print_exc()

# Test 4: Check app.py syntax
print('\nTest 4: App.py syntax check')
import ast
try:
    with open('app.py', 'r', encoding='utf-8') as f:
        code = f.read()
    ast.parse(code)
    print('  [OK] app.py syntax is valid')
except SyntaxError as e:
    print(f'  [FAIL] app.py syntax error at line {e.lineno}: {e.msg}')

# Test 5: Verify weekly planner integration in app.py
print('\nTest 5: App.py integration check')
with open('app.py', 'r', encoding='utf-8') as f:
    app_content = f.read()

checks = [
    ('WEEKLY_PLANNER_AVAILABLE defined', 'WEEKLY_PLANNER_AVAILABLE' in app_content),
    ('generate_weekly_recommendations import', 'generate_weekly_recommendations' in app_content),
    ('use_weekly_planner variable', 'use_weekly_planner' in app_content),
    ('7-Day Diverse Meal Plan checkbox', '7-Day Diverse Meal Plan' in app_content),
    ('g5 column defined', 'g1, g2, g3, g4, g5 = st.columns' in app_content),
]

for check_name, check_result in checks:
    status = '[OK]' if check_result else '[FAIL]'
    print(f'  {status} {check_name}')

print('\n=== ALL TESTS COMPLETED ===')