"""
Recommendation Evaluation Metrics

Calculates evaluation metrics for the 7-day meal planner.
Focuses on metrics that can be calculated without ground truth.
"""

from typing import Dict, List, Optional
from collections import defaultdict
import numpy as np


class RecommendationEvaluator:
    """
    Evaluates recommendation quality using intrinsic metrics.
    
    Note: Metrics requiring ground truth (Precision, Recall, NDCG) are
    marked as unavailable since we don't have user preference data.
    """
    
    def evaluate_weekly_plan(
        self, 
        weekly_plan: List[Dict], 
        safety_report: Dict,
        diversity_metrics: Dict
    ) -> Dict:
        """
        Generate comprehensive evaluation report for a weekly plan.
        
        Args:
            weekly_plan: The 7-day meal plan
            safety_report: Safety filtering report
            diversity_metrics: Diversity metrics from planner
            
        Returns:
            Dictionary with all available evaluation metrics
        """
        evaluation = {
            'intrinsic_metrics': self._calculate_intrinsic_metrics(weekly_plan),
            'diversity_metrics': diversity_metrics,
            'safety_metrics': self._calculate_safety_metrics(safety_report),
            'coverage_metrics': self._calculate_coverage_metrics(weekly_plan),
            'ground_truth_metrics': self._get_ground_truth_status()
        }
        
        return evaluation
    
    def _calculate_intrinsic_metrics(self, weekly_plan: List[Dict]) -> Dict:
        """
        Calculate metrics intrinsic to the recommendations (no ground truth needed).
        """
        total_meal_slots = 0
        total_calories = 0
        total_protein = 0
        total_fiber = 0
        total_sodium = 0
        
        score_components = defaultdict(list)
        
        for day_plan in weekly_plan:
            # Handle both dict formats
            if 'meals' in day_plan:
                meals_dict = day_plan['meals']
            else:
                meals_dict = day_plan
                
            for meal_type, foods in meals_dict.items():
                for food in foods:
                    if isinstance(food, dict):
                        total_meal_slots += 1
                        total_calories += food.get('Calories (kcal)', 0)
                        total_protein += food.get('Protein (g)', 0)
                        total_fiber += food.get('Fibre (g)', 0)
                        total_sodium += food.get('Sodium (mg)', 0)
                        
                        # Collect score components if available
                        breakdown = food.get('_score_breakdown', {})
                        if breakdown:
                            score_components['hybrid_score'].append(
                                breakdown.get('hybrid_score', 0)
                            )
                            score_components['nutrition_score'].append(
                                breakdown.get('nutrition_norm', 0)
                            )
                            score_components['content_score'].append(
                                breakdown.get('content_score', 0)
                            )
        
        # Calculate averages
        avg_calories_per_meal = total_calories / total_meal_slots if total_meal_slots > 0 else 0
        avg_protein_per_meal = total_protein / total_meal_slots if total_meal_slots > 0 else 0
        avg_fiber_per_meal = total_fiber / total_meal_slots if total_meal_slots > 0 else 0
        avg_sodium_per_meal = total_sodium / total_meal_slots if total_meal_slots > 0 else 0
        
        # Calculate score statistics
        avg_hybrid_score = np.mean(score_components['hybrid_score']) if score_components['hybrid_score'] else 0
        avg_nutrition_score = np.mean(score_components['nutrition_score']) if score_components['nutrition_score'] else 0
        avg_content_score = np.mean(score_components['content_score']) if score_components['content_score'] else 0
        
        return {
            'total_meal_slots': total_meal_slots,
            'total_weekly_calories': round(total_calories, 2),
            'total_weekly_protein': round(total_protein, 2),
            'total_weekly_fiber': round(total_fiber, 2),
            'total_weekly_sodium': round(total_sodium, 2),
            'avg_calories_per_meal': round(avg_calories_per_meal, 2),
            'avg_protein_per_meal': round(avg_protein_per_meal, 2),
            'avg_fiber_per_meal': round(avg_fiber_per_meal, 2),
            'avg_sodium_per_meal': round(avg_sodium_per_meal, 2),
            'avg_hybrid_score': round(avg_hybrid_score, 4),
            'avg_nutrition_score': round(avg_nutrition_score, 4),
            'avg_content_score': round(avg_content_score, 4),
            'score_std_dev': round(np.std(score_components['hybrid_score']), 4) if score_components['hybrid_score'] else 0
        }
    
    def _calculate_safety_metrics(self, safety_report: Dict) -> Dict:
        """
        Calculate safety-related metrics.
        """
        original_count = safety_report.get('original_food_count', 0)
        filtered_count = safety_report.get('filtered_food_count', 0)
        
        if original_count > 0:
            filter_rate = (original_count - filtered_count) / original_count
        else:
            filter_rate = 0
        
        return {
            'original_food_count': original_count,
            'filtered_food_count': filtered_count,
            'foods_removed_by_safety': original_count - filtered_count,
            'safety_filter_rate': round(filter_rate, 4),
            'safety_filters_applied': safety_report.get('filtering_applied', False),
            'diseases_filtered': safety_report.get('diseases_filtered', []),
            'severity_levels': safety_report.get('severity_levels', {})
        }
    
    def _calculate_coverage_metrics(self, weekly_plan: List[Dict]) -> Dict:
        """
        Calculate coverage metrics (how well meal types are covered).
        """
        meal_type_coverage = defaultdict(int)
        meal_type_food_counts = defaultdict(int)
        
        expected_meal_types = {'breakfast', 'lunch', 'snack', 'dinner'}
        
        for day_plan in weekly_plan:
            if 'meals' in day_plan:
                meals_dict = day_plan['meals']
            else:
                meals_dict = day_plan
                
            for meal_type, foods in meals_dict.items():
                if foods:  # If meal has foods
                    meal_type_coverage[meal_type] += 1
                    meal_type_food_counts[meal_type] += len(foods)
        
        # Calculate coverage percentage
        total_days = len(weekly_plan)
        coverage = {}
        for meal_type in expected_meal_types:
            days_covered = meal_type_coverage.get(meal_type, 0)
            coverage[meal_type] = {
                'days_covered': days_covered,
                'coverage_percentage': round(days_covered / total_days * 100, 2) if total_days > 0 else 0,
                'total_foods_recommended': meal_type_food_counts.get(meal_type, 0)
            }
        
        return {
            'meal_type_coverage': coverage,
            'total_days_planned': total_days,
            'expected_meal_types': list(expected_meal_types)
        }
    
    def _get_ground_truth_status(self) -> Dict:
        """
        Report status of ground-truth dependent metrics.
        
        Since we don't have user preference data or expert ratings,
        these metrics cannot be calculated.
        """
        return {
            'precision_at_k': {
                'available': False,
                'reason': 'Ground truth user preference data unavailable',
                'value': None
            },
            'recall_at_k': {
                'available': False,
                'reason': 'Ground truth user preference data unavailable',
                'value': None
            },
            'ndcg_at_k': {
                'available': False,
                'reason': 'Ground truth relevance judgments unavailable',
                'value': None
            },
            'accuracy': {
                'available': False,
                'reason': 'Ground truth dietary outcome data unavailable',
                'value': None
            },
            'note': 'These metrics require user preference data, expert ratings, or dietary outcome data which is not available in the current system.'
        }
    
    def generate_evaluation_report(
        self, 
        evaluation: Dict, 
        user_profile: Dict
    ) -> str:
        """
        Generate a human-readable evaluation report.
        
        Args:
            evaluation: The evaluation dictionary
            user_profile: User profile for context
            
        Returns:
            Formatted string report
        """
        report = []
        report.append("=" * 60)
        report.append("7-DAY MEAL PLAN EVALUATION REPORT")
        report.append("=" * 60)
        
        # User context
        diseases = user_profile.get('diseases', [])
        report.append(f"\nUser Profile:")
        report.append(f"  Diseases: {diseases if diseases else 'None'}")
        report.append(f"  Daily Calories: {user_profile.get('daily_calories', 'N/A')}")
        
        # Intrinsic metrics
        intrinsic = evaluation['intrinsic_metrics']
        report.append(f"\nIntrinsic Metrics:")
        report.append(f"  Total Meal Slots: {intrinsic['total_meal_slots']}")
        report.append(f"  Total Weekly Calories: {intrinsic['total_weekly_calories']}")
        report.append(f"  Avg Calories/Meal: {intrinsic['avg_calories_per_meal']}")
        report.append(f"  Avg Protein/Meal: {intrinsic['avg_protein_per_meal']}g")
        report.append(f"  Avg Fiber/Meal: {intrinsic['avg_fiber_per_meal']}g")
        report.append(f"  Avg Sodium/Meal: {intrinsic['avg_sodium_per_meal']}mg")
        report.append(f"  Avg Hybrid Score: {intrinsic['avg_hybrid_score']}")
        
        # Diversity metrics
        diversity = evaluation['diversity_metrics']
        report.append(f"\nDiversity Metrics:")
        report.append(f"  Total Meal Slots: {diversity['total_meal_slots']}")
        report.append(f"  Unique Foods: {diversity['unique_foods']}")
        report.append(f"  Unique Food %: {diversity['unique_food_percentage']}%")
        report.append(f"  Repeated Foods: {diversity['repeated_foods_count']}")
        report.append(f"  Max Repetition: {diversity['max_repetition_count']}")
        report.append(f"  Avg Repetition: {diversity['average_repetition']}")
        
        # Safety metrics
        safety = evaluation['safety_metrics']
        report.append(f"\nSafety Metrics:")
        report.append(f"  Original Foods: {safety['original_food_count']}")
        report.append(f"  Filtered Foods: {safety['filtered_food_count']}")
        report.append(f"  Foods Removed: {safety['foods_removed_by_safety']}")
        report.append(f"  Safety Filter Rate: {safety['safety_filter_rate']:.2%}")
        report.append(f"  Safety Filters Applied: {safety['safety_filters_applied']}")
        
        # Coverage metrics
        coverage = evaluation['coverage_metrics']
        report.append(f"\nCoverage Metrics:")
        report.append(f"  Total Days Planned: {coverage['total_days_planned']}")
        for meal_type, cov in coverage['meal_type_coverage'].items():
            report.append(f"  {meal_type.title()}: {cov['coverage_percentage']}% coverage ({cov['total_foods_recommended']} foods)")
        
        # Ground truth status
        gt = evaluation['ground_truth_metrics']
        report.append(f"\nGround Truth Metrics:")
        report.append(f"  Precision@K: UNAVAILABLE - {gt['precision_at_k']['reason']}")
        report.append(f"  Recall@K: UNAVAILABLE - {gt['recall_at_k']['reason']}")
        report.append(f"  NDCG@K: UNAVAILABLE - {gt['ndcg_at_k']['reason']}")
        report.append(f"  Accuracy: UNAVAILABLE - {gt['accuracy']['reason']}")
        
        report.append(f"\n{gt['note']}")
        report.append("=" * 60)
        
        return "\n".join(report)


def evaluate_recommendations(
    weekly_plan: List[Dict],
    safety_report: Dict,
    diversity_metrics: Dict,
    user_profile: Dict
) -> Dict:
    """
    Convenience function to evaluate a weekly meal plan.
    
    Args:
        weekly_plan: The 7-day meal plan
        safety_report: Safety filtering report
        diversity_metrics: Diversity metrics
        user_profile: User profile for context
        
    Returns:
        Complete evaluation dictionary
    """
    evaluator = RecommendationEvaluator()
    evaluation = evaluator.evaluate_weekly_plan(
        weekly_plan, safety_report, diversity_metrics
    )
    
    # Add human-readable report
    evaluation['text_report'] = evaluator.generate_evaluation_report(
        evaluation, user_profile
    )
    
    return evaluation


if __name__ == "__main__":
    # Test the evaluator
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from backend.services.weekly_meal_planner import generate_weekly_recommendations
    
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
    
    # Generate plan
    plan_result = generate_weekly_recommendations(test_profile)
    
    # Evaluate
    evaluation = evaluate_recommendations(
        plan_result['weekly_plan'],
        plan_result['safety_report'],
        plan_result['diversity_metrics'],
        test_profile
    )
    
    print(evaluation['text_report'])