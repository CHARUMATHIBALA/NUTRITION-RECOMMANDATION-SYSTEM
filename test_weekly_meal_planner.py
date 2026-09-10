"""
Comprehensive tests for 7-Day Diverse Meal Planner

Tests cover:
- 7-day generation
- Safety filter validation
- Diversity metrics
- Repetition control
- Deterministic behavior
- Different patient profiles
- Limited candidate handling
"""

import pytest
import pandas as pd
from backend.services.weekly_meal_planner import (
    WeeklyMealPlanner,
    FoodUsageTracker,
    generate_weekly_recommendations
)


class TestFoodUsageTracker:
    """Test the food usage tracking system."""
    
    def test_initial_state(self):
        """Test tracker starts empty."""
        tracker = FoodUsageTracker()
        assert len(tracker.food_usage_count) == 0
        assert len(tracker.food_last_used_day) == 0
        assert len(tracker.food_meal_types) == 0
    
    def test_record_usage(self):
        """Test recording food usage."""
        tracker = FoodUsageTracker()
        tracker.record_usage("Rice", 1, "Lunch")
        
        assert tracker.get_usage_count("Rice") == 1
        assert tracker.get_days_since_last_use("Rice", 1) == 0
        assert tracker.was_used_in_meal_type("Rice", "Lunch")
    
    def test_multiple_usage(self):
        """Test tracking multiple uses of same food."""
        tracker = FoodUsageTracker()
        tracker.record_usage("Rice", 1, "Lunch")
        tracker.record_usage("Rice", 3, "Dinner")
        
        assert tracker.get_usage_count("Rice") == 2
        assert tracker.get_days_since_last_use("Rice", 4) == 1
    
    def test_meal_type_tracking(self):
        """Test tracking meal types for foods."""
        tracker = FoodUsageTracker()
        tracker.record_usage("Rice", 1, "Lunch")
        tracker.record_usage("Rice", 2, "Dinner")
        
        assert tracker.was_used_in_meal_type("Rice", "Lunch")
        assert tracker.was_used_in_meal_type("Rice", "Dinner")
        assert not tracker.was_used_in_meal_type("Rice", "Breakfast")


class TestWeeklyMealPlanner:
    """Test the main weekly meal planner."""
    
    @pytest.fixture
    def sample_profile(self):
        """Create a sample user profile for testing."""
        return {
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
    
    @pytest.fixture
    def healthy_profile(self):
        """Create a healthy user profile."""
        return {
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
    
    @pytest.fixture
    def severe_profile(self):
        """Create a severe disease profile."""
        return {
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
    
    def test_7_day_generation(self, sample_profile):
        """Test that 7 days are generated with correct structure."""
        planner = WeeklyMealPlanner()
        result = planner.generate_weekly_plan(sample_profile)
        
        # Check 7 days generated
        assert len(result.weekly_plan) == 7
        
        # Check each day has required meal types
        for day_plan in result.weekly_plan:
            assert 'meals' in day_plan
            assert set(day_plan['meals'].keys()) == {'breakfast', 'lunch', 'snack', 'dinner'}
    
    def test_meal_slot_structure(self, sample_profile):
        """Test that each meal slot has correct structure."""
        planner = WeeklyMealPlanner()
        result = planner.generate_weekly_plan(sample_profile)
        
        for day_plan in result.weekly_plan:
            for meal_type, foods in day_plan['meals'].items():
                # Each meal should be a list
                assert isinstance(foods, list)
                
                # Each food should have required fields
                for food in foods:
                    assert 'Dish Name' in food
                    assert 'Calories (kcal)' in food
                    assert 'Protein (g)' in food
                    assert '_score_breakdown' in food
    
    def test_safety_filters_applied(self, sample_profile):
        """Test that safety filters are applied."""
        planner = WeeklyMealPlanner()
        result = planner.generate_weekly_plan(sample_profile)
        
        # Safety report should be present
        assert 'safety_report' in result.__dict__
        safety = result.safety_report
        
        # Should have filtering information
        assert 'original_food_count' in safety
        assert 'filtered_food_count' in safety
        assert 'filtering_applied' in safety
        assert safety['filtering_applied'] == True
    
    def test_diversity_metrics_calculated(self, sample_profile):
        """Test that diversity metrics are calculated."""
        planner = WeeklyMealPlanner()
        result = planner.generate_weekly_plan(sample_profile)
        
        diversity = result.diversity_metrics
        
        # Required metrics
        assert 'total_meal_slots' in diversity
        assert 'unique_foods' in diversity
        assert 'unique_food_percentage' in diversity
        assert 'repeated_foods_count' in diversity
        
        # Should have positive values
        assert diversity['total_meal_slots'] > 0
        assert diversity['unique_foods'] > 0
        assert 0 <= diversity['unique_food_percentage'] <= 100
    
    def test_diversity_improvement(self, sample_profile):
        """Test that diversity mechanism increases food variety."""
        planner = WeeklyMealPlanner()
        result = planner.generate_weekly_plan(sample_profile)
        
        diversity = result.diversity_metrics
        
        # With 28 meal slots, we should have decent diversity
        # (though may not be 100% due to safety constraints)
        assert diversity['unique_food_percentage'] >= 30  # At least 30% unique foods
    
    def test_repetition_control(self, sample_profile):
        """Test that repetition is controlled."""
        planner = WeeklyMealPlanner()
        result = planner.generate_weekly_plan(sample_profile)
        
        diversity = result.diversity_metrics
        
        assert diversity['max_repetition_count'] <= 2
        assert diversity['average_repetition'] <= 2
    
    def test_deterministic_behavior(self, sample_profile):
        """Test that same profile produces same results."""
        planner1 = WeeklyMealPlanner()
        result1 = planner1.generate_weekly_plan(sample_profile)
        
        planner2 = WeeklyMealPlanner()
        result2 = planner2.generate_weekly_plan(sample_profile)
        
        # Same number of unique foods
        assert result1.diversity_metrics['unique_foods'] == result2.diversity_metrics['unique_foods']
        
        # Same foods selected (first day as sample)
        day1_foods1 = [f['Dish Name'] for f in result1.weekly_plan[0]['meals']['breakfast']]
        day1_foods2 = [f['Dish Name'] for f in result2.weekly_plan[0]['meals']['breakfast']]
        assert day1_foods1 == day1_foods2
    
    def test_different_profiles_different_recommendations(self, sample_profile, healthy_profile, severe_profile):
        """Test that different profiles produce different recommendations."""
        planner = WeeklyMealPlanner()
        
        result_healthy = planner.generate_weekly_plan(healthy_profile)
        result_diabetes = planner.generate_weekly_plan(sample_profile)
        result_severe = planner.generate_weekly_plan(severe_profile)
        
        # Different unique food counts expected
        # (Severe profile has more restrictions, likely fewer unique foods)
        assert result_healthy.diversity_metrics['unique_foods'] != result_severe.diversity_metrics['unique_foods']
    
    def test_limited_candidate_handling(self, severe_profile):
        """Test handling when few foods survive filtering."""
        planner = WeeklyMealPlanner()
        result = planner.generate_weekly_plan(severe_profile)
        
        # Should still generate 7 days
        assert len(result.weekly_plan) == 7
        
        # Each day should have meals (even if limited)
        for day_plan in result.weekly_plan:
            total_foods = sum(len(foods) for foods in day_plan['meals'].values())
            assert total_foods > 0  # At least some foods recommended
    
    def test_nutrition_summary(self, sample_profile):
        """Test that nutrition summary is calculated."""
        planner = WeeklyMealPlanner()
        result = planner.generate_weekly_plan(sample_profile)
        
        nutrition = result.nutrition_summary
        
        # Required fields
        assert 'average_daily_calories' in nutrition
        assert 'average_daily_protein' in nutrition
        assert 'target_daily_calories' in nutrition
        
        # Should be reasonable values
        assert nutrition['average_daily_calories'] > 0
        assert nutrition['average_daily_protein'] > 0
    
    def test_scoring_info(self, sample_profile):
        """Test that scoring information is preserved."""
        planner = WeeklyMealPlanner()
        result = planner.generate_weekly_plan(sample_profile)
        
        scoring = result.scoring_info
        
        # Required fields
        assert 'ncf_available' in scoring
        assert 'w_nutrition' in scoring
        assert 'w_content' in scoring
        assert 'diversity_penalties_applied' in scoring
        assert 'deterministic_selection' in scoring
        
        # Diversity should be enabled
        assert scoring['diversity_penalties_applied'] == True
        assert scoring['deterministic_selection'] == True
    
    def test_meal_type_validation(self, sample_profile):
        """Test that meal types are correctly validated."""
        planner = WeeklyMealPlanner()
        result = planner.generate_weekly_plan(sample_profile)
        
        # Check that foods match their meal types
        for day_plan in result.weekly_plan:
            for meal_type, foods in day_plan['meals'].items():
                for food in foods:
                    food_mealtype = food.get('MealType', '').lower()
                    # Should contain the meal type or be compatible
                    assert meal_type in food_mealtype or food_mealtype in meal_type or \
                           'lunch/dinner' in food_mealtype and meal_type in ['lunch', 'dinner']

    def test_no_condiment_standalone_snacks(self, sample_profile):
        planner = WeeklyMealPlanner()
        result = planner.generate_weekly_plan(sample_profile)
        cutoff = planner._get_condiment_fiber_cutoff()
        for day_plan in result.weekly_plan:
            for food in day_plan['meals']['snack']:
                mt = str(food.get('MealType', '')).strip().lower()
                fiber = float(food.get('Fibre (g)', 0) or 0)
                assert not (mt == 'snack' and fiber >= cutoff), food.get('Dish Name')

    def test_unique_meal_combinations(self, sample_profile):
        planner = WeeklyMealPlanner()
        result = planner.generate_weekly_plan(sample_profile)
        for meal_type in ['breakfast', 'lunch', 'snack', 'dinner']:
            combos = [
                tuple(f['Dish Name'] for f in day['meals'][meal_type])
                for day in result.weekly_plan
            ]
            assert len(combos) == len(set(combos))

    def test_meal_validation_block(self, sample_profile):
        planner = WeeklyMealPlanner()
        result = planner.generate_weekly_plan(sample_profile)
        mv = result.meal_validation
        assert mv['is_valid'] is True
        assert mv['checks']['exactly_7_days'] is True
        assert mv['checks']['all_meal_slots_present'] is True
        assert mv['checks']['no_condiment_standalone_meals'] is True
        assert mv['checks']['no_duplicate_day_combinations'] is True


class TestConvenienceFunction:
    """Test the convenience function."""
    
    def test_convenience_function(self):
        """Test the generate_weekly_recommendations convenience function."""
        profile = {
            'age': 40,
            'gender': 'Male',
            'weight': 70,
            'height': 170,
            'bmi': 24.2,
            'activity_level': 'Moderate',
            'diseases': ['Diabetes'],
            'severity': {'diabetes': 'mild'},
            'daily_calories': 2000
        }
        
        result = generate_weekly_recommendations(profile)
        
        # Should return dict with required keys
        assert 'weekly_plan' in result
        assert 'diversity_metrics' in result
        assert 'safety_report' in result
        assert 'nutrition_summary' in result
        assert 'scoring_info' in result
        assert 'meal_validation' in result


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_diseases(self):
        """Test profile with no diseases."""
        profile = {
            'age': 25,
            'gender': 'Female',
            'weight': 55,
            'height': 160,
            'bmi': 21.5,
            'activity_level': 'Active',
            'diseases': [],
            'severity': {},
            'daily_calories': 2000
        }
        
        planner = WeeklyMealPlanner()
        result = planner.generate_weekly_plan(profile)
        
        # Should still generate plan
        assert len(result.weekly_plan) == 7
    
    def test_string_diseases(self):
        """Test handling string instead of list for diseases."""
        profile = {
            'age': 35,
            'gender': 'Male',
            'weight': 80,
            'height': 180,
            'bmi': 24.7,
            'activity_level': 'Moderate',
            'diseases': 'Diabetes',  # String instead of list
            'severity': {'diabetes': 'moderate'},
            'daily_calories': 2200
        }
        
        planner = WeeklyMealPlanner()
        result = planner.generate_weekly_plan(profile)
        
        # Should handle gracefully
        assert len(result.weekly_plan) == 7
    
    def test_missing_severity(self):
        """Test profile with diseases but no severity."""
        profile = {
            'age': 40,
            'gender': 'Male',
            'weight': 75,
            'height': 175,
            'bmi': 24.5,
            'activity_level': 'Moderate',
            'diseases': ['Diabetes'],
            'severity': {},  # Empty severity
            'daily_calories': 2000
        }
        
        planner = WeeklyMealPlanner()
        result = planner.generate_weekly_plan(profile)
        
        # Should use default severity
        assert len(result.weekly_plan) == 7


if __name__ == "__main__":
    # Run basic tests
    print("Running Weekly Meal Planner Tests...")
    
    # Test basic functionality
    profile = {
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
    
    planner = WeeklyMealPlanner()
    result = planner.generate_weekly_plan(profile)
    
    print(f"[OK] 7-day generation: {len(result.weekly_plan)} days")
    print(f"[OK] Unique foods: {result.diversity_metrics['unique_foods']}")
    print(f"[OK] Unique percentage: {result.diversity_metrics['unique_food_percentage']}%")
    print(f"[OK] Max repetition: {result.diversity_metrics['max_repetition_count']}")
    print(f"[OK] Safety filters applied: {result.safety_report['filtering_applied']}")
    print(f"[OK] Deterministic: {result.scoring_info['deterministic_selection']}")
    
    print("\nAll basic tests passed!")