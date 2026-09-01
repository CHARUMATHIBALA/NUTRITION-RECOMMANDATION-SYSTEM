"""
Synthetic User-Food Interaction Dataset Generator
Generates realistic user-food rating data for Neural Collaborative Filtering
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

class UserFoodDatasetGenerator:
    """
    Generates synthetic user-food interaction dataset for NCF training.
    Creates realistic ratings based on food properties and user preferences.
    """
    
    def __init__(self, num_users=1000, num_foods=500, num_interactions=10000):
        """
        Initialize dataset generator.
        
        Args:
            num_users: Number of unique users
            num_foods: Number of unique food items
            num_interactions: Total number of user-food interactions
        """
        self.num_users = num_users
        self.num_foods = num_foods
        self.num_interactions = num_interactions
        self.food_df = None
        self.interactions_df = None
        
    def load_food_dataset(self, food_csv_path='../food_dataset.csv'):
        """
        Load the existing Indian food dataset.
        
        Args:
            food_csv_path: Path to the food dataset CSV file
        """
        try:
            # Try to load from project root
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            full_path = os.path.join(project_root, food_csv_path)
            
            if os.path.exists(full_path):
                self.food_df = pd.read_csv(full_path)
                print(f"Loaded {len(self.food_df)} food items from dataset")
            else:
                print(f"Food dataset not found at {full_path}")
                self._generate_synthetic_foods()
        except Exception as e:
            print(f"Error loading food dataset: {e}")
            self._generate_synthetic_foods()
    
    def _generate_synthetic_foods(self):
        """
        Generate synthetic food items if dataset is not available.
        """
        print("Generating synthetic food items...")
        
        meal_types = ['Breakfast', 'Lunch', 'Dinner', 'Snacks']
        food_names = [
            'Roti', 'Dal', 'Rice', 'Vegetable Curry', 'Chicken Curry',
            'Fish Curry', 'Sambar', 'Idli', 'Dosa', 'Upma',
            'Poha', 'Paratha', 'Paneer Tikka', 'Mixed Veg', 'Dal Makhani',
            'Biryani', 'Pulao', 'Khichdi', 'Salad', 'Soup'
        ]
        
        foods = []
        for i in range(self.num_foods):
            food_name = f"{food_names[i % len(food_names)]} {i//len(food_names)+1}"
            foods.append({
                'food_id': i,
                'food_name': food_name,
                'calories': np.random.randint(50, 500),
                'protein': np.random.uniform(1, 30),
                'carbs': np.random.uniform(5, 80),
                'fats': np.random.uniform(1, 25),
                'MealType': np.random.choice(meal_types),
                'diabetes_friendly': np.random.choice([0, 1], p=[0.3, 0.7]),
                'kidney_friendly': np.random.choice([0, 1], p=[0.4, 0.6]),
                'obesity_friendly': np.random.choice([0, 1], p=[0.3, 0.7])
            })
        
        self.food_df = pd.DataFrame(foods)
        print(f"Generated {len(self.food_df)} synthetic food items")
    
    def generate_user_profiles(self):
        """
        Generate user profiles with dietary preferences and health conditions.
        """
        users = []
        for user_id in range(self.num_users):
            # Random user attributes
            age = np.random.randint(18, 80)
            gender = np.random.choice(['Male', 'Female'])
            bmi = np.random.uniform(15, 40)
            
            # Determine health conditions based on attributes
            has_diabetes = 1 if (bmi > 25 or np.random.random() < 0.2) else 0
            has_kidney_disease = 1 if (age > 50 or np.random.random() < 0.15) else 0
            has_obesity = 1 if bmi > 30 else 0
            
            # Dietary preferences
            prefers_vegetarian = np.random.choice([0, 1], p=[0.3, 0.7])
            prefers_low_sodium = 1 if has_kidney_disease else np.random.choice([0, 1], p=[0.7, 0.3])
            prefers_low_sugar = 1 if has_diabetes else np.random.choice([0, 1], p=[0.6, 0.4])
            
            users.append({
                'user_id': user_id,
                'age': age,
                'gender': gender,
                'bmi': bmi,
                'has_diabetes': has_diabetes,
                'has_kidney_disease': has_kidney_disease,
                'has_obesity': has_obesity,
                'prefers_vegetarian': prefers_vegetarian,
                'prefers_low_sodium': prefers_low_sodium,
                'prefers_low_sugar': prefers_low_sugar
            })
        
        return pd.DataFrame(users)
    
    def generate_interactions(self, users_df):
        """
        Generate user-food interaction ratings.
        Ratings are based on food properties and user preferences.
        """
        interactions = []
        
        for _ in range(self.num_interactions):
            # Random user and food
            user_id = np.random.randint(0, self.num_users)
            food_id = np.random.randint(0, self.num_foods)
            
            user = users_df[users_df['user_id'] == user_id].iloc[0]
            food = self.food_df[self.food_df['food_id'] == food_id].iloc[0]
            
            # Base rating
            rating = np.random.uniform(1, 5)
            
            # Adjust rating based on health conditions and food properties
            if user['has_diabetes'] and not food.get('diabetes_friendly', 1):
                rating -= np.random.uniform(0.5, 1.5)
            
            if user['has_kidney_disease'] and not food.get('kidney_friendly', 1):
                rating -= np.random.uniform(0.5, 1.5)
            
            if user['has_obesity'] and not food.get('obesity_friendly', 1):
                rating -= np.random.uniform(0.5, 1.5)
            
            # Adjust based on calorie preferences
            if food['calories'] > 400 and user['bmi'] > 30:
                rating -= np.random.uniform(0.3, 1.0)
            
            # Ensure rating is within valid range
            rating = max(1.0, min(5.0, rating))
            
            interactions.append({
                'user_id': user_id,
                'food_id': food_id,
                'rating': round(rating, 1),
                'timestamp': datetime.now().timestamp()
            })
        
        self.interactions_df = pd.DataFrame(interactions)
        
        # Remove duplicate user-food pairs (keep highest rating)
        self.interactions_df = self.interactions_df.sort_values('rating').drop_duplicates(
            ['user_id', 'food_id'], keep='last'
        )
        
        print(f"Generated {len(self.interactions_df)} unique interactions")
        return self.interactions_df
    
    def save_datasets(self, output_dir='../data'):
        """
        Save generated datasets to CSV files.
        
        Args:
            output_dir: Directory to save the datasets
        """
        # Create output directory if it doesn't exist
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        full_output_dir = os.path.join(project_root, 'ncf_integration', 'data')
        os.makedirs(full_output_dir, exist_ok=True)
        
        # Save food dataset
        if self.food_df is not None:
            food_path = os.path.join(full_output_dir, 'food_items.csv')
            self.food_df.to_csv(food_path, index=False)
            print(f"Saved food dataset to {food_path}")
        
        # Save interactions dataset
        if self.interactions_df is not None:
            interactions_path = os.path.join(full_output_dir, 'user_food_interactions.csv')
            self.interactions_df.to_csv(interactions_path, index=False)
            print(f"Saved interactions dataset to {interactions_path}")
        
        # Generate and save user profiles
        users_df = self.generate_user_profiles()
        users_path = os.path.join(full_output_dir, 'user_profiles.csv')
        users_df.to_csv(users_path, index=False)
        print(f"Saved user profiles to {users_path}")
        
        return full_output_dir
    
    def get_statistics(self):
        """
        Print statistics about the generated datasets.
        """
        if self.interactions_df is not None:
            print("\n=== Dataset Statistics ===")
            print(f"Number of users: {self.interactions_df['user_id'].nunique()}")
            print(f"Number of foods: {self.interactions_df['food_id'].nunique()}")
            print(f"Number of interactions: {len(self.interactions_df)}")
            print(f"Average rating: {self.interactions_df['rating'].mean():.2f}")
            print(f"Rating distribution:")
            print(self.interactions_df['rating'].value_counts().sort_index())
            print(f"Sparcity: {(1 - len(self.interactions_df) / (self.num_users * self.num_foods)) * 100:.2f}%")


def main():
    """
    Main function to generate the synthetic dataset.
    """
    print("=== Neural Collaborative Filtering Dataset Generator ===\n")
    
    # Initialize generator
    generator = UserFoodDatasetGenerator(
        num_users=1000,
        num_foods=500,
        num_interactions=50000
    )
    
    # Load or generate food dataset
    generator.load_food_dataset()
    
    # Generate user profiles
    users_df = generator.generate_user_profiles()
    print(f"Generated {len(users_df)} user profiles")
    
    # Generate interactions
    generator.generate_interactions(users_df)
    
    # Save datasets
    output_dir = generator.save_datasets()
    
    # Print statistics
    generator.get_statistics()
    
    print(f"\nDatasets saved to: {output_dir}")
    print("Dataset generation complete!")


if __name__ == "__main__":
    main()
