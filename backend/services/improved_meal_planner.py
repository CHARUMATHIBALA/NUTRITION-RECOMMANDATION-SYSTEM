"""
Improved Meal Planner - Generates balanced, nutritionally appropriate meals.

This module implements a sophisticated meal planning system that:
- Classifies foods into categories (main_meal, side_dish, protein, vegetable, fruit, beverage, dessert, snack)
- Uses weighted scoring for food selection
- Constructs balanced meals (main + protein + vegetable)
- Respects disease-specific restrictions
- Prevents food duplication
- Generates meal explanations
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from models import food_df


# ════════════════════════════════════════════════════════════════════
#  FOOD CLASSIFICATION SYSTEM
# ════════════════════════════════════════════════════════════════════

class FoodClassifier:
    """Classifies foods into categories based on name and nutritional characteristics."""
    
    # Keywords for each category
    BEVERAGE_KEYWORDS = [
        'tea', 'coffee', 'chai', 'milkshake', 'juice', 'drink', 'sharbat', 
        'lassi', 'cooler', 'lemonade', 'water', 'stock', 'soup', 'squashes'
    ]
    
    DESSERT_KEYWORDS = [
        'burfi', 'fudge', 'halwa', 'jam', 'sweet', 'kheer', 'payasam', 
        'pudding', 'cake', 'pastry', 'cookie', 'ice cream', 'gulab jamun'
    ]
    
    PROTEIN_KEYWORDS = [
        'egg', 'chicken', 'fish', 'meat', 'paneer', 'dal', 'lentil', 
        'channa', 'chickpea', 'rajma', 'soya', 'tofu', 'keema'
    ]
    
    VEGETABLE_KEYWORDS = [
        'vegetable', 'palak', 'spinach', 'gobi', 'cauliflower', 'aloo', 
        'potato', 'gajar', 'carrot', 'beans', 'peas', 'matar', 'bhindi',
        'okra', 'baingan', 'brinjal', 'tomato', 'onion', 'mushroom', 'cabbage'
    ]
    
    FRUIT_KEYWORDS = [
        'apple', 'banana', 'mango', 'orange', 'guava', 'papaya', 'grapes',
        'pineapple', 'watermelon', 'melon', 'pear', 'peach', 'plum'
    ]
    
    MAIN_MEAL_KEYWORDS = [
        'rice', 'chawal', 'roti', 'chapati', 'paratha', 'dosa', 'idli', 
        'pongal', 'upma', 'khichdi', 'pulao', 'biryani', 'naan', 'bread',
        'pasta', 'noodle', 'macroni', 'oats', 'daliya', 'porridge'
    ]
    
    SNACK_KEYWORDS = [
        'sundal', 'sprouts', 'nuts', 'seeds', 'chivda', 'murmura', 'puffed',
        'sandwich', 'roll', 'wrap', 'samosa', 'pakora', 'bhaji'
    ]
    
    @staticmethod
    def classify_food(food_name: str, calories: float, protein: float, 
                     carbs: float, fat: float, meal_type: str) -> str:
        """
        Classify a food item into a category.
        
        Returns: 'main_meal', 'side_dish', 'protein', 'vegetable', 'fruit', 
                 'beverage', 'dessert', 'snack'
        """
        food_lower = food_name.lower()
        
        # Check for beverages first
        for keyword in FoodClassifier.BEVERAGE_KEYWORDS:
            if keyword in food_lower:
                return 'beverage'
        
        # Check for desserts
        for keyword in FoodClassifier.DESSERT_KEYWORDS:
            if keyword in food_lower:
                return 'dessert'
        
        # Check for fruits
        for keyword in FoodClassifier.FRUIT_KEYWORDS:
            if keyword in food_lower:
                return 'fruit'
        
        # Check for protein sources
        for keyword in FoodClassifier.PROTEIN_KEYWORDS:
            if keyword in food_lower:
                return 'protein'
        
        # Check for vegetables
        for keyword in FoodClassifier.VEGETABLE_KEYWORDS:
            if keyword in food_lower:
                return 'vegetable'
        
        # Check for main meals
        for keyword in FoodClassifier.MAIN_MEAL_KEYWORDS:
            if keyword in food_lower:
                return 'main_meal'
        
        # Check for snacks
        for keyword in FoodClassifier.SNACK_KEYWORDS:
            if keyword in food_lower:
                return 'snack'
        
        # Fallback classification based on nutritional profile
        if protein > 10:
            return 'protein'
        elif calories < 100 and carbs < 15:
            return 'snack'
        elif carbs > 20 and calories > 200:
            return 'main_meal'
        else:
            return 'side_dish'
    
    @staticmethod
    def is_suitable_as_main_meal(category: str) -> bool:
        """Check if a food category is suitable as a main meal item."""
        return category in ['main_meal', 'protein']
    
    @staticmethod
    def is_suitable_as_side_dish(category: str) -> bool:
        """Check if a food category is suitable as a side dish."""
        return category in ['vegetable', 'side_dish', 'protein']


# ════════════════════════════════════════════════════════════════════
#  WEIGHTED SCORING SYSTEM
# ════════════════════════════════════════════════════════════════════

class FoodScorer:
    """Calculates scores for foods based on multiple factors."""
    
    # Configurable weights
    NUTRITION_WEIGHT = 0.30
    DISEASE_WEIGHT = 0.30
    CALORIE_WEIGHT = 0.20
    PROTEIN_FIBRE_WEIGHT = 0.10
    DIVERSITY_WEIGHT = 0.10
    
    @staticmethod
    def calculate_nutrition_score(row: pd.Series) -> float:
        """Calculate nutrition score based on macronutrients."""
        score = 0
        
        # Fiber is always beneficial
        if pd.notna(row.get('Fibre (g)', 0)):
            score += row['Fibre (g)'] * 2
        
        # Protein is beneficial
        if pd.notna(row.get('Protein (g)', 0)):
            score += row['Protein (g)'] * 1.5
        
        # Penalize free sugar
        if pd.notna(row.get('Free Sugar (g)', 0)):
            score -= row['Free Sugar (g)'] * 3
        
        # Penalize excessive fat
        if pd.notna(row.get('Fats (g)', 0)):
            if row['Fats (g)'] > 15:
                score -= (row['Fats (g)'] - 15) * 0.5
        
        return max(0, score)  # Ensure non-negative
    
    @staticmethod
    def calculate_disease_score(row: pd.Series, diseases: List[str], 
                               severity: str = 'moderate') -> float:
        """Calculate disease-specific score."""
        score = 0
        
        has_diabetes = any('diabetes' in d.lower() for d in diseases)
        has_obesity = any('obesity' in d.lower() or 'overweight' in d.lower() for d in diseases)
        has_kidney = any('kidney' in d.lower() for d in diseases)
        
        # Diabetes-specific scoring
        if has_diabetes:
            sugar = row.get('Free Sugar (g)', 0)
            fiber = row.get('Fibre (g)', 0)
            carbs = row.get('Carbohydrates (g)', 0)
            
            if sugar <= 5:
                score += 10
            elif sugar <= 10:
                score += 5
            else:
                score -= 10
            
            if fiber >= 3:
                score += 5
            
            if carbs <= 30:
                score += 3
        
        # Obesity-specific scoring
        if has_obesity:
            calories = row.get('Calories (kcal)', 0)
            fiber = row.get('Fibre (g)', 0)
            protein = row.get('Protein (g)', 0)
            
            if calories <= 250:
                score += 8
            elif calories <= 350:
                score += 4
            else:
                score -= 5
            
            if fiber >= 3:
                score += 5
            
            if protein >= 5:
                score += 4
        
        # Kidney-specific scoring
        if has_kidney:
            sodium = row.get('Sodium (mg)', 0)
            protein = row.get('Protein (g)', 0)
            potassium = row.get('Potassium (mg)', 0) if 'Potassium (mg)' in row else 0
            
            if sodium <= 100:
                score += 10
            elif sodium <= 200:
                score += 5
            else:
                score -= 10
            
            if protein <= 10:
                score += 5
            elif protein <= 15:
                score += 2
            else:
                score -= 5
        
        return max(0, score)
    
    @staticmethod
    def calculate_calorie_score(row: pd.Series, target_calories: float) -> float:
        """Calculate score based on calorie suitability."""
        calories = row.get('Calories (kcal)', 0)
        
        # Calculate how close the food is to target (for single item)
        # This is a simplified version - actual implementation would consider meal context
        if target_calories > 0:
            ratio = calories / target_calories
            if 0.8 <= ratio <= 1.2:
                return 10
            elif 0.5 <= ratio <= 1.5:
                return 5
            else:
                return 0
        return 5  # Neutral if no target
    
    @staticmethod
    def calculate_total_score(row: pd.Series, diseases: List[str], 
                             target_calories: float, used_foods: Set[str],
                             category: str) -> float:
        """Calculate total weighted score for a food item."""
        nutrition_score = FoodScorer.calculate_nutrition_score(row)
        disease_score = FoodScorer.calculate_disease_score(row, diseases)
        calorie_score = FoodScorer.calculate_calorie_score(row, target_calories)
        
        # Protein/fibre score
        protein_fibre_score = 0
        if pd.notna(row.get('Protein (g)', 0)):
            protein_fibre_score += row['Protein (g)'] * 0.5
        if pd.notna(row.get('Fibre (g)', 0)):
            protein_fibre_score += row['Fibre (g)'] * 0.5
        
        # Diversity score (penalize if already used)
        diversity_score = 10 if row['Dish Name'] not in used_foods else 0
        
        # Weighted total
        total = (
            nutrition_score * FoodScorer.NUTRITION_WEIGHT +
            disease_score * FoodScorer.DISEASE_WEIGHT +
            calorie_score * FoodScorer.CALORIE_WEIGHT +
            protein_fibre_score * FoodScorer.PROTEIN_FIBRE_WEIGHT +
            diversity_score * FoodScorer.DIVERSITY_WEIGHT
        )
        
        # Bonus for appropriate category
        if category in ['main_meal', 'protein', 'vegetable']:
            total += 5
        
        return total


# ════════════════════════════════════════════════════════════════════
#  BALANCED MEAL CONSTRUCTION
# ════════════════════════════════════════════════════════════════════

class MealBuilder:
    """Constructs balanced meals from food items."""
    
    # Calorie distribution for meals
    MEAL_CALORIE_DISTRIBUTION = {
        'Breakfast': 0.25,  # 25% of daily calories
        'Lunch': 0.35,      # 35% of daily calories
        'Snack': 0.15,      # 15% of daily calories
        'Dinner': 0.25      # 25% of daily calories
    }
    
    @staticmethod
    def filter_by_meal_type(df: pd.DataFrame, meal_type: str) -> pd.DataFrame:
        """Filter foods by meal type."""
        meal_lower = meal_type.lower()
        
        # Handle Lunch/Dinner combined type
        if meal_type in ['Lunch', 'Dinner']:
            filtered = df[
                (df['MealType'].str.lower().str.contains(meal_lower, na=False)) |
                (df['MealType'].str.lower().str.contains('lunch/dinner', na=False))
            ]
        else:
            filtered = df[
                df['MealType'].str.lower().str.contains(meal_lower, na=False)
            ]
        
        return filtered.copy()
    
    @staticmethod
    def filter_by_category(df: pd.DataFrame, categories: List[str]) -> pd.DataFrame:
        """Filter foods by category."""
        classified_df = df.copy()
        classified_df['category'] = classified_df.apply(
            lambda row: FoodClassifier.classify_food(
                row['Dish Name'],
                row.get('Calories (kcal)', 0),
                row.get('Protein (g)', 0),
                row.get('Carbohydrates (g)', 0),
                row.get('Fats (g)', 0),
                row.get('MealType', '')
            ),
            axis=1
        )
        
        return classified_df[classified_df['category'].isin(categories)]
    
    @staticmethod
    def build_balanced_meal(df: pd.DataFrame, meal_type: str, 
                           target_calories: float, diseases: List[str],
                           used_foods: Set[str]) -> Dict:
        """
        Build a balanced meal with main item + protein + vegetable.
        
        Returns: {
            'main': food_item,
            'protein': food_item or None,
            'vegetable': food_item or None,
            'total_calories': float,
            'explanation': str
        }
        """
        meal_df = MealBuilder.filter_by_meal_type(df, meal_type)
        
        if meal_df.empty:
            return MealBuilder._create_fallback_meal(meal_type, target_calories)
        
        # Get main meal items
        main_items = MealBuilder.filter_by_category(meal_df, ['main_meal'])
        
        # If no main items, use protein items as main
        if main_items.empty:
            main_items = MealBuilder.filter_by_category(meal_df, ['protein'])
        
        # Score and select main item
        main_item = MealBuilder._select_best_food(
            main_items, diseases, target_calories * 0.6, used_foods
        )
        
        if main_item is None:
            return MealBuilder._create_fallback_meal(meal_type, target_calories)
        
        # Add to used foods
        used_foods.add(main_item['Dish Name'])
        
        # Get protein side (if main is not protein-rich)
        protein_item = None
        if main_item.get('Protein (g)', 0) < 10:
            protein_items = MealBuilder.filter_by_category(meal_df, ['protein'])
            protein_items = protein_items[~protein_items['Dish Name'].isin(used_foods)]
            protein_item = MealBuilder._select_best_food(
                protein_items, diseases, target_calories * 0.2, used_foods
            )
            if protein_item:
                used_foods.add(protein_item['Dish Name'])
        
        # Get vegetable side
        vegetable_item = None
        vegetable_items = MealBuilder.filter_by_category(meal_df, ['vegetable'])
        vegetable_items = vegetable_items[~vegetable_items['Dish Name'].isin(used_foods)]
        vegetable_item = MealBuilder._select_best_food(
            vegetable_items, diseases, target_calories * 0.2, used_foods
        )
        if vegetable_item:
            used_foods.add(vegetable_item['Dish Name'])
        
        # Calculate total calories
        total_calories = main_item.get('Calories (kcal)', 0)
        if protein_item:
            total_calories += protein_item.get('Calories (kcal)', 0)
        if vegetable_item:
            total_calories += vegetable_item.get('Calories (kcal)', 0)
        
        # Generate explanation
        explanation = MealBuilder._generate_explanation(
            main_item, protein_item, vegetable_item, diseases
        )
        
        return {
            'main': main_item,
            'protein': protein_item,
            'vegetable': vegetable_item,
            'total_calories': total_calories,
            'explanation': explanation
        }
    
    @staticmethod
    def _select_best_food(df: pd.DataFrame, diseases: List[str], 
                         target_calories: float, used_foods: Set[str]) -> Optional[pd.Series]:
        """Select the best food from a filtered dataframe."""
        if df.empty:
            return None
        
        # Calculate scores
        df = df.copy()
        df['category'] = df.apply(
            lambda row: FoodClassifier.classify_food(
                row['Dish Name'],
                row.get('Calories (kcal)', 0),
                row.get('Protein (g)', 0),
                row.get('Carbohydrates (g)', 0),
                row.get('Fats (g)', 0),
                row.get('MealType', '')
            ),
            axis=1
        )
        
        df['score'] = df.apply(
            lambda row: FoodScorer.calculate_total_score(
                row, diseases, target_calories, used_foods, row['category']
            ),
            axis=1
        )
        
        # Sort by score and return top
        df = df.sort_values('score', ascending=False)
        
        # Filter out beverages and desserts from main selection
        df = df[~df['category'].isin(['beverage', 'dessert'])]
        
        if df.empty:
            return None
        
        return df.iloc[0]
    
    @staticmethod
    def _generate_explanation(main: pd.Series, protein: Optional[pd.Series],
                            vegetable: Optional[pd.Series], diseases: List[str]) -> str:
        """Generate explanation for the meal."""
        parts = []
        
        # Main item description
        main_fiber = main.get('Fibre (g)', 0)
        main_protein = main.get('Protein (g)', 0)
        
        if main_fiber >= 3:
            parts.append("High-fibre")
        if main_protein >= 5:
            parts.append("protein-rich")
        
        has_diabetes = any('diabetes' in d.lower() for d in diseases)
        has_obesity = any('obesity' in d.lower() or 'overweight' in d.lower() for d in diseases)
        
        if has_diabetes:
            parts.append("controlled-carbohydrate")
        if has_obesity:
            parts.append("calorie-conscious")
        
        if not parts:
            parts.append("balanced")
        
        explanation = f"{' '.join(parts)} meal"
        
        if protein or vegetable:
            explanation += " with "
            components = []
            if protein:
                components.append("protein support")
            if vegetable:
                components.append("vegetables")
            explanation += " and ".join(components)
        
        explanation += " suitable for your health profile."
        
        return explanation
    
    @staticmethod
    def _create_fallback_meal(meal_type: str, target_calories: float) -> Dict:
        """Create a fallback meal when suitable foods are not available."""
        return {
            'main': None,
            'protein': None,
            'vegetable': None,
            'total_calories': 0,
            'explanation': f"No suitable {meal_type.lower()} options available in the dataset."
        }


# ════════════════════════════════════════════════════════════════════
#  MAIN MEAL PLANNER
# ════════════════════════════════════════════════════════════════════

class ImprovedMealPlanner:
    """Main class for generating improved meal plans."""
    
    def __init__(self):
        self.df = food_df.copy()
    
    def generate_daily_meal_plan(self, diseases: List[str], 
                                  daily_calories: float) -> Dict:
        """
        Generate a complete daily meal plan.
        
        Returns: {
            'Breakfast': meal_dict,
            'Lunch': meal_dict,
            'Snack': meal_dict,
            'Dinner': meal_dict,
            'total_calories': float,
            'validation': dict
        }
        """
        used_foods = set()
        meal_plan = {}
        total_calories = 0
        
        meal_types = ['Breakfast', 'Lunch', 'Snack', 'Dinner']
        
        for meal_type in meal_types:
            target_calories = daily_calories * MealBuilder.MEAL_CALORIE_DISTRIBUTION[meal_type]
            
            meal = MealBuilder.build_balanced_meal(
                self.df, meal_type, target_calories, diseases, used_foods
            )
            
            meal_plan[meal_type] = meal
            total_calories += meal['total_calories']
        
        # Validate meal plan
        validation = self._validate_meal_plan(meal_plan, diseases, daily_calories)
        
        return {
            'meal_plan': meal_plan,
            'total_calories': total_calories,
            'validation': validation
        }
    
    def _validate_meal_plan(self, meal_plan: Dict, diseases: List[str],
                           target_calories: float) -> Dict:
        """Validate the generated meal plan."""
        validation = {
            'is_valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Check if any meal has no main item
        for meal_type, meal in meal_plan.items():
            if meal['main'] is None:
                validation['errors'].append(f"No main item for {meal_type}")
                validation['is_valid'] = False
        
        # Check calorie range
        total_calories = meal_plan['total_calories'] if 'total_calories' in meal_plan else 0
        calorie_ratio = total_calories / target_calories if target_calories > 0 else 0
        
        if calorie_ratio < 0.7:
            validation['warnings'].append(f"Total calories ({total_calories:.0f}) are significantly below target ({target_calories:.0f})")
        elif calorie_ratio > 1.3:
            validation['warnings'].append(f"Total calories ({total_calories:.0f}) exceed target ({target_calories:.0f})")
        
        # Check for duplicates
        all_foods = []
        for meal in meal_plan.values():
            if meal['main'] is not None:
                all_foods.append(meal['main']['Dish Name'])
            if meal['protein'] is not None:
                all_foods.append(meal['protein']['Dish Name'])
            if meal['vegetable'] is not None:
                all_foods.append(meal['vegetable']['Dish Name'])
        
        if len(all_foods) != len(set(all_foods)):
            validation['warnings'].append("Duplicate foods detected in meal plan")
        
        return validation
    
    def get_swap_alternatives(self, meal_type: str, current_food_name: str,
                            diseases: List[str], target_calories: float,
                            used_foods: Set[str], limit: int = 3) -> List[pd.Series]:
        """
        Get suitable swap alternatives for a food item.
        
        Returns list of food items (as Series) suitable for swapping.
        """
        meal_df = MealBuilder.filter_by_meal_type(self.df, meal_type)
        
        if meal_df.empty:
            return []
        
        # Get category of current food
        current_row = self.df[self.df['Dish Name'] == current_food_name]
        if current_row.empty:
            return []
        
        current_row = current_row.iloc[0]
        current_category = FoodClassifier.classify_food(
            current_food_name,
            current_row.get('Calories (kcal)', 0),
            current_row.get('Protein (g)', 0),
            current_row.get('Carbohydrates (g)', 0),
            current_row.get('Fats (g)', 0),
            current_row.get('MealType', '')
        )
        
        # Filter by same category
        category_df = MealBuilder.filter_by_category(meal_df, [current_category])
        
        # Exclude current food and already used foods
        category_df = category_df[
            (category_df['Dish Name'] != current_food_name) &
            (~category_df['Dish Name'].isin(used_foods))
        ]
        
        if category_df.empty:
            return []
        
        # Score and select top alternatives
        category_df = category_df.copy()
        category_df['score'] = category_df.apply(
            lambda row: FoodScorer.calculate_total_score(
                row, diseases, target_calories, used_foods, current_category
            ),
            axis=1
        )
        
        category_df = category_df.sort_values('score', ascending=False)
        
        # Return top N alternatives
        alternatives = []
        for _, row in category_df.head(limit).iterrows():
            alternatives.append(row)
        
        return alternatives


def generate_improved_meal_plan(diseases: List[str], daily_calories: float) -> Dict:
    """
    Convenience function to generate an improved meal plan.
    
    Parameters
    ----------
    diseases : List[str]
        List of predicted diseases.
    daily_calories : float
        Daily calorie target.
    
    Returns
    -------
    Dict
        Complete meal plan with validation.
    """
    planner = ImprovedMealPlanner()
    return planner.generate_daily_meal_plan(diseases, daily_calories)
