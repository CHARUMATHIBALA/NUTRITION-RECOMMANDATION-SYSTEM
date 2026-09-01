"""
Nutrition Filtering Module
Filters food items based on disease-specific dietary restrictions
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple


class NutritionFilter:
    """
    Filters food items based on health conditions and dietary restrictions.
    Ensures recommended foods are safe for the user's health profile.
    """
    
    def __init__(self, food_dataset_path='../food_dataset.csv'):
        """
        Initialize nutrition filter with food dataset.
        
        Args:
            food_dataset_path: Path to the food dataset CSV file
        """
        self.food_df = None
        self.load_food_dataset(food_dataset_path)
        
        # Define disease-specific food restrictions
        self.restrictions = {
            'diabetes': {
                'avoid_high_sugar': True,
                'avoid_high_glycemic': True,
                'max_sugar_per_serving': 10,  # grams
                'preferred_carbs_range': (20, 45),  # grams
                'avoid_processed': True
            },
            'kidney_disease': {
                'avoid_high_sodium': True,
                'avoid_high_potassium': True,
                'avoid_high_phosphorus': True,
                'max_sodium_per_serving': 140,  # mg
                'max_potassium_per_serving': 200,  # mg
                'avoid_high_protein': True,
                'max_protein_per_serving': 20  # grams
            },
            'obesity': {
                'avoid_high_calorie': True,
                'avoid_high_fat': True,
                'max_calories_per_serving': 300,  # kcal
                'max_fat_per_serving': 15,  # grams
                'prefer_high_fiber': True,
                'min_fiber_per_serving': 3  # grams
            },
            'hypertension': {
                'avoid_high_sodium': True,
                'max_sodium_per_serving': 140,  # mg
                'prefer_high_potassium': True
            },
            'heart_disease': {
                'avoid_high_saturated_fat': True,
                'avoid_high_cholesterol': True,
                'max_saturated_fat_per_serving': 5,  # grams
                'max_cholesterol_per_serving': 20,  # mg
                'prefer_omega3': True
            }
        }
    
    def load_food_dataset(self, food_dataset_path):
        """
        Load food dataset for filtering.
        
        Args:
            food_dataset_path: Path to the food dataset
        """
        try:
            import os
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            full_path = os.path.join(project_root, food_dataset_path)
            
            if os.path.exists(full_path):
                self.food_df = pd.read_csv(full_path)
                print(f"Loaded {len(self.food_df)} food items for nutrition filtering")
            else:
                print(f"Food dataset not found at {full_path}")
                self._create_synthetic_food_data()
        except Exception as e:
            print(f"Error loading food dataset: {e}")
            self._create_synthetic_food_data()
    
    def _create_synthetic_food_data(self):
        """
        Create synthetic food data if dataset is not available.
        """
        print("Creating synthetic food data for nutrition filtering...")
        
        foods = []
        food_names = [
            'Roti', 'Dal', 'Rice', 'Vegetable Curry', 'Chicken Curry',
            'Fish Curry', 'Sambar', 'Idli', 'Dosa', 'Upma',
            'Poha', 'Paratha', 'Paneer Tikka', 'Mixed Veg', 'Dal Makhani',
            'Biryani', 'Pulao', 'Khichdi', 'Salad', 'Soup',
            'Oats', 'Yogurt', 'Fruits', 'Nuts', 'Sprouts'
        ]
        
        meal_types = ['Breakfast', 'Lunch', 'Dinner', 'Snacks']
        
        for i, name in enumerate(food_names):
            foods.append({
                'food_id': i,
                'food_name': name,
                'calories': np.random.randint(50, 500),
                'protein': np.random.uniform(1, 30),
                'carbs': np.random.uniform(5, 80),
                'fats': np.random.uniform(1, 25),
                'fiber': np.random.uniform(0, 15),
                'sugar': np.random.uniform(0, 20),
                'sodium': np.random.uniform(10, 500),
                'potassium': np.random.uniform(50, 400),
                'MealType': np.random.choice(meal_types),
                'is_vegetarian': np.random.choice([0, 1], p=[0.3, 0.7]),
                'diabetes_friendly': np.random.choice([0, 1], p=[0.3, 0.7]),
                'kidney_friendly': np.random.choice([0, 1], p=[0.4, 0.6]),
                'obesity_friendly': np.random.choice([0, 1], p=[0.3, 0.7])
            })
        
        self.food_df = pd.DataFrame(foods)
        print(f"Created {len(self.food_df)} synthetic food items")
    
    def filter_by_disease(self, disease: str, food_ids: List[int] = None) -> List[int]:
        """
        Filter food items based on specific disease restrictions.
        
        Args:
            disease: Disease name ('diabetes', 'kidney_disease', 'obesity', etc.)
            food_ids: List of food IDs to filter (if None, filter all foods)
            
        Returns:
            List of allowed food IDs
        """
        if self.food_df is None:
            raise ValueError("Food dataset not loaded")
        
        # Get restriction rules for the disease
        if disease.lower() not in self.restrictions:
            print(f"No restrictions defined for {disease}. Returning all foods.")
            return food_ids if food_ids else self.food_df['food_id'].tolist()
        
        rules = self.restrictions[disease.lower()]
        filtered_df = self.food_df.copy()
        
        # Apply disease-specific filters
        if disease.lower() == 'diabetes':
            if rules['avoid_high_sugar']:
                filtered_df = filtered_df[filtered_df['sugar'] <= rules['max_sugar_per_serving']]
            if rules['avoid_high_glycemic']:
                filtered_df = filtered_df[filtered_df['carbs'] <= rules['preferred_carbs_range'][1]]
        
        elif disease.lower() == 'kidney_disease':
            if rules['avoid_high_sodium']:
                filtered_df = filtered_df[filtered_df['sodium'] <= rules['max_sodium_per_serving']]
            if rules['avoid_high_potassium']:
                filtered_df = filtered_df[filtered_df['potassium'] <= rules['max_potassium_per_serving']]
            if rules['avoid_high_protein']:
                filtered_df = filtered_df[filtered_df['protein'] <= rules['max_protein_per_serving']]
        
        elif disease.lower() == 'obesity':
            if rules['avoid_high_calorie']:
                filtered_df = filtered_df[filtered_df['calories'] <= rules['max_calories_per_serving']]
            if rules['avoid_high_fat']:
                filtered_df = filtered_df[filtered_df['fats'] <= rules['max_fat_per_serving']]
            if rules['prefer_high_fiber']:
                filtered_df = filtered_df[filtered_df['fiber'] >= rules['min_fiber_per_serving']]
        
        # Use pre-defined friendly flags if available
        friendly_column = f"{disease.lower()}_friendly"
        if friendly_column in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[friendly_column] == 1]
        
        # Filter by provided food IDs if specified
        if food_ids:
            filtered_df = filtered_df[filtered_df['food_id'].isin(food_ids)]
        
        allowed_food_ids = filtered_df['food_id'].tolist()
        
        print(f"Filtered {len(food_ids) if food_ids else len(self.food_df)} foods to {len(allowed_food_ids)} for {disease}")
        
        return allowed_food_ids
    
    def filter_by_multiple_diseases(self, diseases: List[str], 
                                    food_ids: List[int] = None) -> List[int]:
        """
        Filter food items based on multiple diseases (intersection of allowed foods).
        
        Args:
            diseases: List of disease names
            food_ids: List of food IDs to filter (if None, filter all foods)
            
        Returns:
            List of allowed food IDs (intersection of all disease filters)
        """
        if not diseases:
            return food_ids if food_ids else self.food_df['food_id'].tolist()
        
        # Get allowed foods for each disease
        allowed_sets = []
        for disease in diseases:
            allowed_foods = self.filter_by_disease(disease, food_ids)
            allowed_sets.append(set(allowed_foods))
        
        # Get intersection (foods allowed for all diseases)
        if allowed_sets:
            allowed_foods = list(set.intersection(*allowed_sets))
        else:
            allowed_foods = food_ids if food_ids else self.food_df['food_id'].tolist()
        
        print(f"Filtered for multiple diseases {diseases}: {len(allowed_foods)} foods allowed")
        
        return allowed_foods
    
    def filter_by_dietary_preferences(self, preferences: Dict[str, bool],
                                     food_ids: List[int] = None) -> List[int]:
        """
        Filter food items based on dietary preferences.
        
        Args:
            preferences: Dictionary of dietary preferences
                e.g., {'vegetarian': True, 'low_sodium': True, 'low_sugar': True}
            food_ids: List of food IDs to filter
            
        Returns:
            List of allowed food IDs
        """
        if self.food_df is None:
            raise ValueError("Food dataset not loaded")
        
        filtered_df = self.food_df.copy()
        
        if food_ids:
            filtered_df = filtered_df[filtered_df['food_id'].isin(food_ids)]
        
        # Apply dietary preference filters
        if preferences.get('vegetarian', False):
            if 'is_vegetarian' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['is_vegetarian'] == 1]
        
        if preferences.get('low_sodium', False):
            if 'sodium' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['sodium'] <= 140]
        
        if preferences.get('low_sugar', False):
            if 'sugar' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['sugar'] <= 10]
        
        if preferences.get('low_calorie', False):
            if 'calories' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['calories'] <= 300]
        
        if preferences.get('high_protein', False):
            if 'protein' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['protein'] >= 15]
        
        allowed_food_ids = filtered_df['food_id'].tolist()
        
        print(f"Filtered by dietary preferences: {len(allowed_food_ids)} foods allowed")
        
        return allowed_food_ids
    
    def get_food_nutrition_info(self, food_id: int) -> Dict:
        """
        Get nutritional information for a specific food item.
        
        Args:
            food_id: ID of the food item
            
        Returns:
            Dictionary with nutritional information
        """
        if self.food_df is None:
            raise ValueError("Food dataset not loaded")
        
        food_info = self.food_df[self.food_df['food_id'] == food_id]
        
        if food_info.empty:
            return {}
        
        return food_info.iloc[0].to_dict()
    
    def get_allowed_foods_summary(self, diseases: List[str]) -> pd.DataFrame:
        """
        Get summary of allowed foods for given diseases.
        
        Args:
            diseases: List of disease names
            
        Returns:
            DataFrame with allowed foods and their nutritional info
        """
        allowed_food_ids = self.filter_by_multiple_diseases(diseases)
        
        if self.food_df is None:
            return pd.DataFrame()
        
        allowed_foods_df = self.food_df[self.food_df['food_id'].isin(allowed_food_ids)]
        
        return allowed_foods_df
    
    def score_food_suitability(self, food_id: int, diseases: List[str]) -> float:
        """
        Score how suitable a food item is for given diseases.
        
        Args:
            food_id: ID of the food item
            diseases: List of disease names
            
        Returns:
            Suitability score (0 to 1, higher is better)
        """
        if self.food_df is None:
            return 0.0
        
        food_info = self.get_food_nutrition_info(food_id)
        
        if not food_info:
            return 0.0
        
        score = 1.0
        
        for disease in diseases:
            disease = disease.lower()
            
            # Check if food is marked as friendly for this disease
            friendly_column = f"{disease}_friendly"
            if friendly_column in food_info:
                if food_info[friendly_column] == 0:
                    score *= 0.5  # Penalize unfriendly foods
            
            # Apply disease-specific scoring
            if disease == 'diabetes':
                if food_info.get('sugar', 0) > 10:
                    score *= 0.7
                if food_info.get('carbs', 0) > 50:
                    score *= 0.8
            
            elif disease == 'kidney_disease':
                if food_info.get('sodium', 0) > 140:
                    score *= 0.6
                if food_info.get('potassium', 0) > 200:
                    score *= 0.6
                if food_info.get('protein', 0) > 20:
                    score *= 0.7
            
            elif disease == 'obesity':
                if food_info.get('calories', 0) > 300:
                    score *= 0.7
                if food_info.get('fats', 0) > 15:
                    score *= 0.7
                if food_info.get('fiber', 0) < 3:
                    score *= 0.8
        
        return max(0.0, min(1.0, score))


def main():
    """
    Test the nutrition filtering module.
    """
    print("=== Testing Nutrition Filter ===\n")
    
    # Initialize filter
    filter = NutritionFilter()
    
    # Test single disease filtering
    print("Testing diabetes filtering...")
    diabetes_foods = filter.filter_by_disease('diabetes')
    print(f"Allowed foods for diabetes: {len(diabetes_foods)}")
    
    print("\nTesting kidney disease filtering...")
    kidney_foods = filter.filter_by_disease('kidney_disease')
    print(f"Allowed foods for kidney disease: {len(kidney_foods)}")
    
    print("\nTesting obesity filtering...")
    obesity_foods = filter.filter_by_disease('obesity')
    print(f"Allowed foods for obesity: {len(obesity_foods)}")
    
    # Test multiple disease filtering
    print("\nTesting multiple disease filtering...")
    multi_disease_foods = filter.filter_by_multiple_diseases(['diabetes', 'obesity'])
    print(f"Allowed foods for diabetes + obesity: {len(multi_disease_foods)}")
    
    # Test dietary preferences
    print("\nTesting dietary preferences...")
    pref_foods = filter.filter_by_dietary_preferences({
        'vegetarian': True,
        'low_sodium': True
    })
    print(f"Allowed foods for vegetarian + low sodium: {len(pref_foods)}")
    
    # Test food suitability scoring
    print("\nTesting food suitability scoring...")
    sample_food_id = 0
    score = filter.score_food_suitability(sample_food_id, ['diabetes', 'obesity'])
    print(f"Suitability score for food {sample_food_id}: {score:.4f}")
    
    # Get food nutrition info
    print("\nFood nutrition info:")
    food_info = filter.get_food_nutrition_info(sample_food_id)
    for key, value in food_info.items():
        print(f"  {key}: {value}")
    
    print("\nNutrition filter test complete!")


if __name__ == "__main__":
    main()
