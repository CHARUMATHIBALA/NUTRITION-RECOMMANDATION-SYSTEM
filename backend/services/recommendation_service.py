"""
Unified Recommendation Service
Primary recommendation pipeline for FRRANC nutrition recommendations.

Pipeline:
User Profile → Disease Filtering → Nutrition Scoring → Meal Planning → Results
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from models import food_df
import config


class FoodFilter:
    """Filters food items based on disease restrictions and meal type."""
    
    def __init__(self):
        self.df = food_df.copy()
    
    def get_disease_conditions(self, diseases: List[str] | str) -> Dict[str, bool]:
        """Convert disease list to condition flags."""
        conditions = {
            'has_diabetes': False,
            'has_kidney_disease': False,
            'has_obesity': False
        }
        
        if isinstance(diseases, str):
            diseases = [diseases]
        
        for disease in diseases:
            disease_lower = str(disease).lower()
            if 'diabetes' in disease_lower:
                conditions['has_diabetes'] = True
            elif 'kidney' in disease_lower:
                conditions['has_kidney_disease'] = True
            elif 'obesity' in disease_lower or 'overweight' in disease_lower:
                conditions['has_obesity'] = True
        
        return conditions
    
    def apply_disease_filters(self, conditions: Dict[str, bool]) -> pd.DataFrame:
        """Apply disease-specific nutritional filters."""
        filtered_df = self.df.copy()
        
        if conditions['has_diabetes']:
            filtered_df = filtered_df[
                (filtered_df["Free Sugar (g)"] <= config.DIABETES_MAX_SUGAR) &
                (filtered_df["Calories (kcal)"] <= config.DIABETES_MAX_CALORIES)
            ]
        
        if conditions['has_kidney_disease']:
            filtered_df = filtered_df[
                (filtered_df["Sodium (mg)"] <= config.KIDNEY_MAX_SODIUM) &
                (filtered_df["Protein (g)"] <= config.KIDNEY_MAX_PROTEIN)
            ]
        
        if conditions['has_obesity']:
            filtered_df = filtered_df[
                (filtered_df["Calories (kcal)"] <= config.OBESITY_MAX_CALORIES) &
                (filtered_df["Fats (g)"] <= config.OBESITY_MAX_FAT)
            ]
        
        if not any(conditions.values()):
            filtered_df = filtered_df[
                (filtered_df["Calories (kcal)"] <= config.CALORIES_MAX_NORMAL) &
                (filtered_df["Free Sugar (g)"] <= config.FREE_SUGAR_MAX_NORMAL)
            ]
        
        return filtered_df
    
    def filter_by_meal_type(self, df: pd.DataFrame, meal_type: str) -> pd.DataFrame:
        """Filter foods by meal type."""
        meal_lower = meal_type.lower()
        return df[df["MealType"].str.lower().str.contains(meal_lower, na=False)]


class NutritionScorer:
    """Scores food items based on nutritional suitability."""
    
    def __init__(self):
        pass
    
    def calculate_nutrition_score(self, row: pd.Series, conditions: Dict[str, bool], 
                                   user_profile: Optional[Dict] = None) -> float:
        """Calculate nutrition score based on disease conditions and user profile."""
        score = 0
        
        # Base nutrition factors
        fiber = float(row.get('Fibre (g)', 0))
        protein = float(row.get('Protein (g)', 0))
        sugar = float(row.get('Free Sugar (g)', 0))
        calories = float(row.get('Calories (kcal)', 0))
        sodium = float(row.get('Sodium (mg)', 0))
        
        # Disease-specific scoring
        if conditions['has_diabetes']:
            score -= sugar * 3
            score -= float(row.get('Carbohydrates (g)', 0)) * 0.5
            score += fiber * 3
        
        if conditions['has_kidney_disease']:
            score -= sodium * 0.1
            score -= protein * 0.5
        
        if conditions['has_obesity']:
            score -= calories * 0.05
            score -= float(row.get('Fats (g)', 0)) * 0.5
            score += protein * 1.5
            score += fiber * 2
        
        # User profile adjustments (if provided)
        if user_profile:
            age = int(user_profile.get('age', 30))
            gender = str(user_profile.get('gender', 'Male'))
            bmi = float(user_profile.get('bmi', 22))
            
            # Age-based
            if age > 65:
                score += protein * 1.2
            elif age < 30:
                score += min(calories / 100, 5) * 0.5
            
            # Gender-based
            if gender.lower() == 'male':
                score += protein * 1.1
            else:
                score += float(row.get('Iron (mg)', 0)) * 2
            
            # BMI-based
            if bmi > 25:
                score -= calories * 0.03
                score += fiber * 2
            elif bmi < 18.5:
                score += calories * 0.02
                score += protein * 1.5
        
        # General nutrition benefits
        if not any(conditions.values()):
            score += fiber * 1.5
            score += protein * 1
            score -= sugar * 1
        
        return score


class MealPlanner:
    """Constructs meal plans from filtered and scored foods."""
    
    def __init__(self):
        self.meal_types = ["Breakfast", "Lunch", "Snack", "Dinner"]
        self.calorie_distribution = {
            'Breakfast': 0.25,
            'Lunch': 0.35,
            'Snack': 0.15,
            'Dinner': 0.25
        }
    
    def generate_meal_plan(self, filtered_df: pd.DataFrame, scorer: NutritionScorer,
                          conditions: Dict[str, bool], user_profile: Optional[Dict] = None,
                          daily_calories: Optional[float] = None) -> Dict[str, List[Dict]]:
        """Generate meal plan with food recommendations for each meal type."""
        meal_plan = {}
        selected_foods = set()
        
        for meal_type in self.meal_types:
            # Filter by meal type
            meal_foods = filtered_df[
                filtered_df["MealType"].str.lower().str.contains(meal_type.lower(), na=False)
            ].copy()
            
            if meal_foods.empty:
                meal_plan[meal_type] = []
                continue
            
            # Score foods
            meal_foods['score'] = meal_foods.apply(
                lambda row: scorer.calculate_nutrition_score(row, conditions, user_profile),
                axis=1
            )
            
            # Sort by score
            meal_foods = meal_foods.sort_values('score', ascending=False)
            
            # Filter out already selected foods
            meal_foods = meal_foods[~meal_foods['Dish Name'].isin(selected_foods)]
            
            # Select top foods
            top_foods = meal_foods.head(5)
            
            # Convert to list of dicts
            meal_plan[meal_type] = top_foods.to_dict('records')
            
            # Track selected foods
            for food in meal_plan[meal_type]:
                selected_foods.add(food['Dish Name'])
        
        return meal_plan


class RecommendationService:
    """Primary recommendation service orchestrating the complete pipeline."""
    
    def __init__(self):
        self.filter = FoodFilter()
        self.scorer = NutritionScorer()
        self.planner = MealPlanner()
    
    def generate_recommendations(
        self,
        diseases: List[str] | str,
        age: int,
        gender: str,
        height: float,
        weight: float,
        bmi: float,
        activity_level: str,
        daily_calories: float,
        hba1c: float,
        glucose: float,
        bp: float,
        sodium: float,
        potassium: float,
        creatinine: float,
        goal: Optional[str] = None,
        region: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive nutrition recommendations.
        
        Args:
            diseases: List of detected diseases
            age: User age
            gender: User gender
            height: User height in cm
            weight: User weight in kg
            bmi: User BMI
            activity_level: Activity level
            daily_calories: Daily calorie target (TDEE)
            hba1c: HbA1c level
            glucose: Glucose level
            bp: Blood pressure
            sodium: Sodium level
            potassium: Potassium level
            creatinine: Creatinine level
            goal: Health goal (optional)
            region: Region preference (optional)
        
        Returns:
            Dictionary containing meal plan, foods to avoid, water intake, etc.
        """
        # Build user profile
        user_profile = {
            'diseases': diseases if isinstance(diseases, list) else [diseases],
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
            'goal': goal,
            'region': region
        }
        
        # Get disease conditions
        conditions = self.filter.get_disease_conditions(diseases)
        
        # Apply disease filters
        filtered_df = self.filter.apply_disease_filters(conditions)
        
        # Generate meal plan
        meal_plan = self.planner.generate_meal_plan(
            filtered_df, self.scorer, conditions, user_profile, daily_calories
        )
        
        # Calculate additional recommendations
        foods_to_avoid = self._get_foods_to_avoid(conditions)
        water_intake = self._calculate_water_intake(weight, activity_level)
        protein_requirement = self._calculate_protein_requirement(weight, gender, age, diseases)
        nutrition_tips = self._get_nutrition_tips(diseases, bmi, activity_level)
        
        return {
            'meal_plan': meal_plan,
            'foods_to_avoid': foods_to_avoid,
            'water_intake': water_intake,
            'protein_requirement': protein_requirement,
            'nutrition_tips': nutrition_tips,
            'meal_validation': {'is_valid': True, 'warnings': [], 'errors': []},
            'use_unified_service': True
        }
    
    def _get_foods_to_avoid(self, conditions: Dict[str, bool]) -> List[str]:
        """Get list of foods to avoid based on conditions."""
        avoid_foods = []
        
        if conditions['has_diabetes']:
            avoid_foods.extend(['Sweets', 'Desserts', 'Sugary drinks', 'High sugar foods'])
        if conditions['has_kidney_disease']:
            avoid_foods.extend(['High sodium foods', 'Processed foods', 'Canned foods'])
        if conditions['has_obesity']:
            avoid_foods.extend(['Fried foods', 'High calorie foods', 'High fat foods'])
        
        return avoid_foods
    
    def _calculate_water_intake(self, weight: float, activity_level: str) -> float:
        """Calculate daily water intake requirement."""
        base_water = weight * 0.033  # 33ml per kg
        multiplier = config.WATER_ACTIVITY_MULTIPLIERS.get(activity_level, 1.0)
        return round(base_water * multiplier, 2)
    
    def _calculate_protein_requirement(self, weight: float, gender: str, 
                                       age: int, diseases: List[str]) -> float:
        """Calculate daily protein requirement."""
        base_protein = weight * 0.8  # 0.8g per kg
        
        # Adjust for age
        if age > 65:
            base_protein *= 1.2
        
        # Adjust for gender
        if gender.lower() == 'male':
            base_protein *= 1.1
        
        # Adjust for diseases
        if any('kidney' in d.lower() for d in diseases):
            base_protein *= 0.8  # Reduce for kidney disease
        elif any('obesity' in d.lower() or 'overweight' in d.lower() for d in diseases):
            base_protein *= 1.2  # Increase for weight management
        
        return round(base_protein, 2)
    
    def _get_nutrition_tips(self, diseases: List[str], bmi: float, 
                           activity_level: str) -> List[str]:
        """Get nutrition tips based on health profile."""
        tips = []
        
        if any('diabetes' in d.lower() for d in diseases):
            tips.extend([
                "Choose foods with low glycemic index",
                "Monitor carbohydrate intake",
                "Increase fiber consumption"
            ])
        
        if any('kidney' in d.lower() for d in diseases):
            tips.extend([
                "Limit sodium intake",
                "Monitor protein consumption",
                "Choose fresh over processed foods"
            ])
        
        if any('obesity' in d.lower() or 'overweight' in d.lower() for d in diseases):
            tips.extend([
                "Focus on portion control",
                "Choose nutrient-dense foods",
                "Increase physical activity"
            ])
        
        if bmi < 18.5:
            tips.append("Increase calorie intake with nutrient-rich foods")
        elif bmi > 25:
            tips.append("Create a moderate calorie deficit for weight management")
        
        if activity_level == 'Sedentary':
            tips.append("Consider increasing daily physical activity")
        
        return tips


# Singleton instance
_service_instance = None

def get_recommendation_service() -> RecommendationService:
    """Get singleton instance of RecommendationService."""
    global _service_instance
    if _service_instance is None:
        _service_instance = RecommendationService()
    return _service_instance
