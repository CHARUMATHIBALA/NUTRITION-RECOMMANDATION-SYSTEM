import pandas as pd
from models import food_df
import config
from typing import List, Dict, Any, Optional

class IntelligentNutritionRecommender:
    """
    Intelligent nutrition recommendation engine that considers
    multiple health parameters and disease conditions.
    """
    
    def __init__(self):
        self.df = food_df.copy()
        
    def _get_disease_conditions(self, diseases: List[str] | str) -> Dict[str, bool]:
        """
        Get filtering conditions based on predicted diseases.
        Handles multiple diseases by combining all constraints.
        """
        conditions = {
            'has_diabetes': False,
            'has_kidney_disease': False,
            'has_obesity': False
        }
        
        for disease in diseases:
            disease_lower = str(disease).lower()
            if 'diabetes' in disease_lower:
                conditions['has_diabetes'] = True
            elif 'kidney' in disease_lower:
                conditions['has_kidney_disease'] = True
            elif 'obesity' in disease_lower or 'overweight' in disease_lower:
                conditions['has_obesity'] = True
                
        return conditions
    
    def _apply_disease_filters(self, conditions: Dict[str, bool]) -> pd.DataFrame:
        """
        Apply food filters based on disease conditions.
        Multiple diseases result in stricter combined filters.
        """
        filtered_df = self.df.copy()
        
        # Diabetes filters: Low GI,高 fiber, low sugar
        if conditions['has_diabetes']:
            filtered_df = filtered_df[
                (filtered_df["Free Sugar (g)"] <= config.DIABETES_MAX_SUGAR) &
            (filtered_df["Calories (kcal)"] <= config.DIABETES_MAX_CALORIES)
            ]
        
        # Kidney Disease filters: Low sodium, low potassium, controlled protein
        if conditions['has_kidney_disease']:
            filtered_df = filtered_df[
                (filtered_df["Sodium (mg)"] <= config.KIDNEY_MAX_SODIUM) &
            (filtered_df["Protein (g)"] <= config.KIDNEY_MAX_PROTEIN)
            ]
        
        # Obesity filters: Low calorie, high protein, high fiber
        if conditions['has_obesity']:
            filtered_df = filtered_df[
                (filtered_df["Calories (kcal)"] <= config.OBESITY_MAX_CALORIES) &
            (filtered_df["Fats (g)"] <= config.OBESITY_MAX_FAT)
            ]
        
        # If no disease, apply balanced diet filters
        if not any(conditions.values()):
            filtered_df = filtered_df[
                (filtered_df["Calories (kcal)"] <= config.CALORIES_MAX_NORMAL) &
            (filtered_df["Free Sugar (g)"] <= config.FREE_SUGAR_MAX_NORMAL)
            ]
        
        return filtered_df
    
    def _calculate_nutritional_score(self, row: pd.Series, conditions: Dict[str, bool]) -> float:
        """
        Calculate a nutritional score based on disease conditions.
        Higher score = better match for the condition.
        """
        score = 0
        
        # Fiber is always beneficial

        
        if conditions['has_diabetes']:
            # Penalize sugar and carbs
            score -= row["Free Sugar (g)"] * 3
            score -= row["Carbohydrates (g)"] * 0.5

        if conditions['has_kidney_disease']:
            # Penalize sodium and protein
            score -= row["Sodium (mg)"] * 0.1
            score -= row["Protein (g)"] * 0.5

        if conditions['has_obesity']:
            # Penalize calories and fats, reward protein
            score -= row["Calories (kcal)"] * 0.05
            score -= row["Fats (g)"] * 0.5
            score += row["Protein (g)"] * 1.5

        # Determine fiber contribution (apply highest applicable weight)
        fiber_weight = 0
        if conditions['has_diabetes']:
            fiber_weight = max(fiber_weight, 3)
        if conditions['has_obesity']:
            fiber_weight = max(fiber_weight, 2)
        if not any(conditions.values()):
            fiber_weight = max(fiber_weight, 1.5)
        if fiber_weight:
            score += row["Fibre (g)"] * fiber_weight

        # Normal diet adjustments (protein and sugar) when no disease
        if not any(conditions.values()):
            score += row["Protein (g)"] * 1
            score -= row["Free Sugar (g)"] * 1
            
        return score
    
    def _get_nutritional_benefits(self, row: pd.Series) -> str:
        """
        Generate nutritional benefits description for a food item.
        """
        benefits = []
        
        if row["Fibre (g)"] >= 3:
            benefits.append(f"High fiber ({row['Fibre (g)']:.1f}g)")
        if row["Protein (g)"] >= 5:
            benefits.append(f"Good protein ({row['Protein (g)']:.1f}g)")
        if row["Free Sugar (g)"] <= 5:
            benefits.append("Low sugar")
        if row["Sodium (mg)"] <= 100:
            benefits.append("Low sodium")
        if row["Fats (g)"] <= 10:
            benefits.append("Low fat")
        if row["Calories (kcal)"] <= 200:
            benefits.append("Low calorie")
        if row["Vitamin C (mg)"] >= 10:
            benefits.append(f"Rich in Vitamin C ({row['Vitamin C (mg)']:.1f}mg)")
        if row["Iron (mg)"] >= 1:
            benefits.append(f"Good iron source ({row['Iron (mg)']:.2f}mg)")
        if row["Calcium (mg)"] >= 50:
            benefits.append(f"Calcium rich ({row['Calcium (mg)']:.1f}mg)")
            
        return ", ".join(benefits) if benefits else "Balanced nutrition"
    
    def _get_recommendation_reason(self, row: pd.Series, conditions: Dict[str, bool]) -> str:
        """
        Generate reason for recommendation based on disease conditions.
        """
        reasons = []
        
        if conditions['has_diabetes']:
            if row["Free Sugar (g)"] <= 5:
                reasons.append("Low sugar content helps manage blood glucose")
            if row["Fibre (g)"] >= 3:
                reasons.append("High fiber slows sugar absorption")
                
        if conditions['has_kidney_disease']:
            if row["Sodium (mg)"] <= 100:
                reasons.append("Low sodium reduces kidney workload")
            if row["Protein (g)"] <= 10:
                reasons.append("Controlled protein for kidney health")
                
        if conditions['has_obesity']:
            if row["Calories (kcal)"] <= 250:
                reasons.append("Low calorie supports weight management")
            if row["Protein (g)"] >= 5:
                reasons.append("High protein promotes satiety")
            if row["Fibre (g)"] >= 3:
                reasons.append("High fiber aids digestion and fullness")
                
        if not any(conditions.values()):
            reasons.append("Balanced nutrition for overall health")
            if row["Protein (g)"] >= 5:
                reasons.append("Good protein content")
            if row["Fibre (g)"] >= 2:
                reasons.append("Adequate fiber for digestive health")
                
        return ". ".join(reasons) if reasons else "Nutritious choice"
    
    def recommend_food(self, diseases: List[str] | str, meal_type: Optional[str] = None, top_n: int = 5) -> pd.DataFrame:
        """Recommend foods based on disease conditions and meal type.

        Args:
            diseases: List of predicted diseases or a single disease string.
            meal_type: Optional meal type filter (e.g., 'Breakfast', 'Lunch').
            top_n: Number of top recommendations to return.

        Returns:
            pandas.DataFrame with recommended foods and additional info.
        """
        # Handle single disease string
        if isinstance(diseases, str):
            diseases = [diseases]
        
        # Get disease conditions
        conditions = self._get_disease_conditions(diseases)
        
        # Apply disease filters
        filtered_df = self._apply_disease_filters(conditions)
        
        # Filter by meal type
        if meal_type:
            # Handle both exact match and partial match (e.g., "Lunch/Dinner")
            meal_lower = meal_type.lower()
            filtered_df = filtered_df[
                filtered_df["MealType"].str.lower().str.contains(meal_lower, na=False)
            ]
        
        # Calculate nutritional scores

        filtered_df['nutritional_score'] = filtered_df.apply(
            lambda row: self._calculate_nutritional_score(row, conditions), axis=1
        )
        
        # Sort by nutritional score (descending)
        filtered_df = filtered_df.sort_values('nutritional_score', ascending=False)
        
        # Get top recommendations
        recommendations = filtered_df.head(top_n).copy()
        
        # Add nutritional benefits and reasons
        recommendations['Nutritional Benefits'] = recommendations.apply(
            lambda row: self._get_nutritional_benefits(row), axis=1
        )
        recommendations['Reason for Recommendation'] = recommendations.apply(
            lambda row: self._get_recommendation_reason(row, conditions), axis=1
        )
        
        # Select and order columns
        result_columns = [
            "Dish Name",
            "Calories (kcal)",
            "Nutritional Benefits",
            "Reason for Recommendation"
        ]
        
        return recommendations[result_columns]
    
    def get_foods_to_avoid(self, diseases: List[str] | str) -> List[Dict[str, str]]:
        """Get list of foods to avoid based on disease conditions.

        Args:
            diseases: List of predicted diseases or a single disease string.

        Returns:
            List of dictionaries with 'food' and 'reason' keys.
        """
        if isinstance(diseases, str):
            diseases = [diseases]
            
        conditions = self._get_disease_conditions(diseases)
        avoid_df = self.df.copy()
        if avoid_df.empty:
            return []
        
        avoid_reasons = []
        
        if conditions['has_diabetes']:
            high_sugar = avoid_df[avoid_df["Free Sugar (g)"] > 15]
            for _, row in high_sugar.iterrows():
                avoid_reasons.append({
                    'food': row['Dish Name'],
                    'reason': f"High sugar ({row['Free Sugar (g)']:.1f}g) - spikes blood glucose"
                })
                
        if conditions['has_kidney_disease']:
            high_sodium = avoid_df[avoid_df["Sodium (mg)"] > 400]
            for _, row in high_sodium.iterrows():
                avoid_reasons.append({
                    'food': row['Dish Name'],
                    'reason': f"High sodium ({row['Sodium (mg)']:.0f}mg) - strains kidneys"
                })
            high_protein = avoid_df[avoid_df["Protein (g)"] > 20]
            for _, row in high_protein.iterrows():
                avoid_reasons.append({
                    'food': row['Dish Name'],
                    'reason': f"High protein ({row['Protein (g)']:.1f}g) - increases kidney workload"
                })
                
        if conditions['has_obesity']:
            high_calorie = avoid_df[avoid_df["Calories (kcal)"] > 500]
            for _, row in high_calorie.iterrows():
                avoid_reasons.append({
                    'food': row['Dish Name'],
                    'reason': f"High calorie ({row['Calories (kcal)']:.0f}kcal) - hinders weight loss"
                })
                
        if not any(conditions.values()):
            # For normal users, avoid extremely high sugar/sodium items
            extreme_items = avoid_df[
                (avoid_df["Free Sugar (g)"] > 20) |
                (avoid_df["Sodium (mg)"] > 600)
            ]
            for _, row in extreme_items.iterrows():
                avoid_reasons.append({
                    'food': row['Dish Name'],
                    'reason': "Excessive sugar or sodium - consume in moderation"
                })
        
        # Remove duplicates and return top 10
        seen = set()
        unique_avoids = []
        for item in avoid_reasons:
            if item['food'] not in seen:
                seen.add(item['food'])
                unique_avoids.append(item)
                if len(unique_avoids) >= 10:
                    break
                    
        return unique_avoids[:10]
    
    def calculate_water_intake(self, weight: float, activity_level: str) -> float:
        """Calculate daily water intake recommendation.

        Args:
            weight: Weight in kilograms.
            activity_level: Activity level string (e.g., 'Sedentary', 'Active').

        Returns:
            Daily water intake in liters, rounded to one decimal place.
        """
        # Base: 35ml per kg (using config factor)
        base_water = weight * config.WATER_INTAKE_FACTOR

        # Activity multiplier
        multiplier = config.WATER_ACTIVITY_MULTIPLIERS.get(activity_level, 1.0)
        total_water = base_water * multiplier

        total_water = min(total_water, config.MAX_WATER_INTAKE_L)
        return round(total_water, 1)
    
    def calculate_protein_requirement(self, weight: float, gender: str, age: int, diseases) -> float:
        """
        Calculate daily protein requirement based on health conditions.
        
        Args:
            weight: Weight in kg
            gender: 'Male' or 'Female'
            age: Age in years
            diseases: List of predicted diseases
            
        Returns:
            Daily protein requirement in grams
        """
        if isinstance(diseases, str):
            diseases = [diseases]
            
        conditions = self._get_disease_conditions(diseases)
        
        # Base protein requirement (g per kg)
        if conditions['has_kidney_disease']:
            # Kidney disease: lower protein
            protein_per_kg = 0.6
        elif conditions['has_obesity']:
            # Obesity: higher protein for satiety
            protein_per_kg = 1.2
        elif conditions['has_diabetes']:
            # Diabetes: moderate protein
            protein_per_kg = 1.0
        else:
            # Normal: standard requirement
            protein_per_kg = 0.8
            
        # Age adjustment
        if age > 65:
            protein_per_kg += 0.1  # Older adults need slightly more
            
        # Gender adjustment
        if gender.lower() == 'male':
            protein_per_kg += 0.1
            
        total_protein = weight * protein_per_kg
        
        if conditions['has_kidney_disease']:
            total_protein = min(total_protein, config.KIDNEY_MAX_PROTEIN)
        return round(total_protein, 1)
    
    def get_nutrition_tips(self, diseases, bmi: float, activity_level: str) -> List[str]:
        """
        Generate personalized nutrition tips.
        """
        if isinstance(diseases, str):
            diseases = [diseases]
            
        conditions = self._get_disease_conditions(diseases)
        tips = []
        
        # General tips
        tips.append("Eat meals at regular intervals to maintain stable energy levels")
        tips.append("Include a variety of colorful vegetables in your diet")
        tips.append("Stay hydrated throughout the day")
        
        # Disease-specific tips
        if conditions['has_diabetes']:
            tips.append("Choose whole grains over refined carbohydrates")
            tips.append("Monitor carbohydrate intake and pair with protein/fiber")
            tips.append("Avoid sugary beverages and processed foods")
            tips.append("Eat smaller, frequent meals to manage blood sugar")
            
        if conditions['has_kidney_disease']:
            tips.append("Limit sodium intake by avoiding processed and canned foods")
            tips.append("Control portion sizes of high-potassium foods")
            tips.append("Choose lean protein sources in moderate amounts")
            tips.append("Avoid adding salt during cooking; use herbs instead")
            
        if conditions['has_obesity']:
            tips.append("Focus on portion control and mindful eating")
            tips.append("Include protein-rich foods to increase satiety")
            tips.append("Choose fiber-rich foods to feel full longer")
            tips.append("Limit empty calories from sugary drinks and snacks")
            tips.append("Consider eating slowly to recognize hunger cues")
            
        if not any(conditions.values()):
            tips.append("Maintain a balanced diet with all food groups")
            tips.append("Practice portion control for weight maintenance")
            tips.append("Include healthy fats from nuts, seeds, and olive oil")
            
        # Activity-specific tips
        if activity_level in ["Active", "Very Active"]:
            tips.append("Ensure adequate carbohydrate intake for energy")
            tips.append("Time meals around workouts for optimal performance")
            
        # BMI-specific tips
        if bmi < 18.5:
            tips.append("Include nutrient-dense foods to support healthy weight gain")
        elif bmi > 25:
            tips.append("Focus on nutrient density over calorie density")
            
        return tips[:config.MAX_TIPS]  # Return top configured tips