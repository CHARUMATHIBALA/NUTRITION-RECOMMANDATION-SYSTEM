"""
Hybrid Recommendation System Pipeline
Combines Disease Prediction, Nutrition Filtering, and Neural Collaborative Filtering
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ncf_integration.models.ncf_model import NCFModel
from ncf_integration.utils.nutrition_filter import NutritionFilter
from predict import predict_diabetes, predict_kidney, predict_obesity


class HybridRecommender:
    """
    Hybrid recommendation system that combines:
    1. Disease Prediction (Random Forest)
    2. Nutrition Filtering (Disease-specific restrictions)
    3. Neural Collaborative Filtering (Personalized recommendations)
    
    Pipeline:
    User Health Data → Disease Prediction → Nutrition Filtering → NCF → Top-N Recommendations
    """
    
    def __init__(self, ncf_model_path: str = None, food_dataset_path: str = '../food_dataset.csv'):
        """
        Initialize hybrid recommender.
        
        Args:
            ncf_model_path: Path to trained NCF model
            food_dataset_path: Path to food dataset
        """
        self.ncf_model = None
        self.nutrition_filter = None
        self.ncf_model_path = ncf_model_path
        self.food_dataset_path = food_dataset_path
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all components of the hybrid system."""
        # Initialize nutrition filter
        self.nutrition_filter = NutritionFilter(self.food_dataset_path)
        
        # Load NCF model if path provided
        if self.ncf_model_path and os.path.exists(self.ncf_model_path):
            self.load_ncf_model(self.ncf_model_path)
    
    def load_ncf_model(self, model_path: str):
        """
        Load trained NCF model.
        
        Args:
            model_path: Path to the saved NCF model
        """
        try:
            self.ncf_model = NCFModel(num_users=1000, num_items=500)  # Placeholder dimensions
            self.ncf_model.load_model(model_path)
            print(f"NCF model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading NCF model: {e}")
            self.ncf_model = None
    
    def predict_diseases(self, age: int, gender: str, bmi: float, 
                         hba1c: float, glucose: float, 
                         sodium: float, potassium: float, 
                         bp: float, creatinine: float) -> Dict[str, str]:
        """
        Predict diseases using Random Forest models.
        
        Args:
            age: Patient age
            gender: Patient gender
            bmi: Body Mass Index
            hba1c: HbA1c level
            glucose: Blood glucose level
            sodium: Sodium level
            potassium: Potassium level
            bp: Blood pressure
            creatinine: Serum creatinine level
            
        Returns:
            Dictionary with disease predictions
        """
        predictions = {}
        
        try:
            # Predict diabetes
            diabetes = predict_diabetes(age, gender, bmi, hba1c, glucose)
            predictions['diabetes'] = diabetes
            
            # Predict kidney disease
            kidney = predict_kidney(age, gender, bmi, sodium, potassium, bp, creatinine)
            predictions['kidney_disease'] = kidney
            
            # Predict obesity
            obesity = predict_obesity(age, gender, bmi)
            predictions['obesity'] = obesity
            
        except Exception as e:
            print(f"Error predicting diseases: {e}")
            predictions = {
                'diabetes': 'Normal',
                'kidney_disease': 'Normal',
                'obesity': 'Normal'
            }
        
        return predictions
    
    def extract_diseases_from_predictions(self, predictions: Dict[str, str]) -> List[str]:
        """
        Extract disease names from prediction results.

        Each value in *predictions* may be either a plain string (legacy) or a
        dict with a 'label' key (current format from predict.py, e.g.
        {"label": "Diabetes", "confidence": 95.0}).  We normalise both shapes
        to a lowercase string before any comparison.

        Args:
            predictions: Dictionary with disease predictions

        Returns:
            List of detected diseases
        """
        diseases = []

        def _label(val) -> str:
            """Return the lowercase label string regardless of whether val is
            a plain string or a {'label': ..., 'confidence': ...} dict."""
            if isinstance(val, dict):
                return str(val.get('label', '')).lower().strip()
            return str(val).lower().strip()

        # Check diabetes
        if _label(predictions.get('diabetes', '')) == 'diabetes':
            diseases.append('diabetes')

        # Check kidney disease
        if _label(predictions.get('kidney_disease', '')) == 'kidney disease':
            diseases.append('kidney_disease')

        # Check obesity
        obesity_label = _label(predictions.get('obesity', ''))
        if obesity_label in ['obese class i', 'obese class ii', 'obese class iii', 'overweight']:
            diseases.append('obesity')

        return diseases
    
    def filter_foods_by_diseases(self, diseases: List[str], 
                                 food_ids: List[int] = None) -> List[int]:
        """
        Filter foods based on detected diseases.
        
        Args:
            diseases: List of detected diseases
            food_ids: Optional list of food IDs to filter
            
        Returns:
            List of allowed food IDs
        """
        if not diseases:
            # If no diseases, return all foods
            if food_ids:
                return food_ids
            elif self.nutrition_filter.food_df is not None:
                return self.nutrition_filter.food_df['food_id'].tolist()
            else:
                return []
        
        # Filter by multiple diseases
        allowed_foods = self.nutrition_filter.filter_by_multiple_diseases(diseases, food_ids)
        
        return allowed_foods
    
    def get_personalized_recommendations(self, user_id: int, diseases: List[str], 
                                         top_n: int = 10) -> List[Dict]:
        """
        Get personalized food recommendations using NCF.
        
        Args:
            user_id: User ID for personalization
            diseases: List of detected diseases
            top_n: Number of recommendations to return
            
        Returns:
            List of recommended foods with details
        """
        if self.ncf_model is None:
            print("NCF model not loaded. Using fallback recommendation.")
            return self._get_fallback_recommendations(diseases, top_n)
        
        # Get all food IDs
        if self.nutrition_filter.food_df is not None:
            all_food_ids = self.nutrition_filter.food_df['food_id'].tolist()
        else:
            return []
        
        # Filter foods by diseases
        allowed_food_ids = self.filter_foods_by_diseases(diseases, all_food_ids)
        
        if not allowed_food_ids:
            print(f"No foods allowed for diseases: {diseases}")
            return []
        
        # Get NCF predictions for allowed foods
        try:
            predictions = self.ncf_model.predict(user_id, allowed_food_ids)
            
            # Sort by predicted rating
            food_predictions = list(zip(allowed_food_ids, predictions))
            food_predictions.sort(key=lambda x: x[1], reverse=True)
            
            # Get top-N recommendations
            top_recommendations = food_predictions[:top_n]
            
            # Add food details
            recommendations = []
            for food_id, predicted_rating in top_recommendations:
                food_info = self.nutrition_filter.get_food_nutrition_info(food_id)
                if food_info:
                    food_info['predicted_rating'] = predicted_rating
                    food_info['suitability_score'] = self.nutrition_filter.score_food_suitability(
                        food_id, diseases
                    )
                    recommendations.append(food_info)
            
            return recommendations
            
        except Exception as e:
            print(f"Error getting NCF recommendations: {e}")
            return self._get_fallback_recommendations(diseases, top_n)
    
    def _get_fallback_recommendations(self, diseases: List[str], top_n: int = 10) -> List[Dict]:
        """
        Fallback recommendation when NCF is not available.
        Uses nutrition filtering and random selection.
        
        Args:
            diseases: List of detected diseases
            top_n: Number of recommendations to return
            
        Returns:
            List of recommended foods
        """
        # Get allowed foods
        allowed_food_ids = self.filter_foods_by_diseases(diseases)
        
        if not allowed_food_ids:
            return []
        
        # Randomly select top-N foods
        selected_ids = np.random.choice(allowed_food_ids, min(top_n, len(allowed_food_ids)), replace=False)
        
        recommendations = []
        for food_id in selected_ids:
            food_info = self.nutrition_filter.get_food_nutrition_info(food_id)
            if food_info:
                food_info['predicted_rating'] = np.random.uniform(3.0, 5.0)
                food_info['suitability_score'] = self.nutrition_filter.score_food_suitability(
                    food_id, diseases
                )
                recommendations.append(food_info)
        
        # Sort by suitability score
        recommendations.sort(key=lambda x: x['suitability_score'], reverse=True)
        
        return recommendations
    
    def recommend(self, user_id: int, age: int, gender: str, bmi: float,
                 hba1c: float, glucose: float, sodium: float, potassium: float,
                 bp: float, creatinine: float, top_n: int = 10,
                 dietary_preferences: Dict[str, bool] = None) -> Dict:
        """
        Complete recommendation pipeline.
        
        Pipeline Steps:
        1. Disease Prediction (Random Forest)
        2. Disease Extraction
        3. Nutrition Filtering (Disease-specific)
        4. Dietary Preference Filtering
        5. Neural Collaborative Filtering (Personalization)
        6. Top-N Selection
        
        Args:
            user_id: User ID for personalization
            age: Patient age
            gender: Patient gender
            bmi: Body Mass Index
            hba1c: HbA1c level
            glucose: Blood glucose level
            sodium: Sodium level
            potassium: Potassium level
            bp: Blood pressure
            creatinine: Serum creatinine level
            top_n: Number of recommendations to return
            dietary_preferences: Optional dietary preferences
            
        Returns:
            Dictionary with complete recommendation results
        """
        result = {
            'user_id': user_id,
            'health_profile': {
                'age': age,
                'gender': gender,
                'bmi': bmi,
                'hba1c': hba1c,
                'glucose': glucose,
                'sodium': sodium,
                'potassium': potassium,
                'bp': bp,
                'creatinine': creatinine
            },
            'disease_predictions': None,
            'detected_diseases': None,
            'allowed_foods_count': 0,
            'recommendations': [],
            'recommendation_method': None
        }
        
        # Step 1: Disease Prediction
        print("Step 1: Predicting diseases...")
        disease_predictions = self.predict_diseases(
            age, gender, bmi, hba1c, glucose, sodium, potassium, bp, creatinine
        )
        result['disease_predictions'] = disease_predictions
        
        # Step 2: Extract diseases
        print("Step 2: Extracting detected diseases...")
        detected_diseases = self.extract_diseases_from_predictions(disease_predictions)
        result['detected_diseases'] = detected_diseases
        
        if not detected_diseases:
            print("No diseases detected. Using general recommendations.")
            detected_diseases = ['normal']
        
        # Step 3: Get personalized recommendations
        print("Step 3: Getting personalized recommendations...")
        recommendations = self.get_personalized_recommendations(user_id, detected_diseases, top_n * 2)
        
        # Step 4: Apply dietary preferences if provided
        if dietary_preferences and recommendations:
            print("Step 4: Applying dietary preferences...")
            pref_food_ids = self.nutrition_filter.filter_by_dietary_preferences(dietary_preferences)
            recommendations = [r for r in recommendations if r['food_id'] in pref_food_ids]
        
        # Step 5: Select top-N
        print(f"Step 5: Selecting top-{top_n} recommendations...")
        recommendations = recommendations[:top_n]
        
        result['recommendations'] = recommendations
        result['allowed_foods_count'] = len(self.filter_foods_by_diseases(detected_diseases))
        result['recommendation_method'] = 'NCF' if self.ncf_model else 'Fallback'
        
        print(f"Recommendation complete. Generated {len(recommendations)} recommendations.")
        
        return result
    
    def format_recommendations_for_display(self, recommendations: List[Dict]) -> pd.DataFrame:
        """
        Format recommendations for display in Streamlit.
        
        Args:
            recommendations: List of recommendation dictionaries
            
        Returns:
            DataFrame with formatted recommendations
        """
        if not recommendations:
            return pd.DataFrame()
        
        formatted_data = []
        for rec in recommendations:
            formatted_data.append({
                'Food': rec.get('food_name', 'Unknown'),
                'Calories': rec.get('calories', 0),
                'Protein (g)': round(rec.get('protein', 0), 1),
                'Carbs (g)': round(rec.get('carbs', 0), 1),
                'Fats (g)': round(rec.get('fats', 0), 1),
                'Fiber (g)': round(rec.get('fiber', 0), 1),
                'Predicted Rating': round(rec.get('predicted_rating', 0), 2),
                'Suitability Score': round(rec.get('suitability_score', 0), 2),
                'Meal Type': rec.get('MealType', 'Unknown')
            })
        
        return pd.DataFrame(formatted_data)
    
    def get_recommendation_explanation(self, diseases: List[str]) -> str:
        """
        Generate explanation for recommendations based on diseases.
        
        Args:
            diseases: List of detected diseases
            
        Returns:
            Explanation string
        """
        if not diseases or 'normal' in diseases:
            return "Based on your health profile, here are generally healthy food recommendations."
        
        explanations = {
            'diabetes': "For diabetes management, we recommend foods low in sugar and with controlled carbohydrate content to help maintain stable blood glucose levels.",
            'kidney_disease': "For kidney health, we recommend foods low in sodium, potassium, and protein to reduce kidney workload and prevent complications.",
            'obesity': "For weight management, we recommend low-calorie, low-fat foods high in fiber to promote satiety and support healthy weight loss."
        }
        
        explanation_parts = []
        for disease in diseases:
            if disease in explanations:
                explanation_parts.append(explanations[disease])
        
        if explanation_parts:
            return " ".join(explanation_parts)
        else:
            return "Based on your health profile, we've selected foods that align with your nutritional needs."


def main():
    """
    Test the hybrid recommendation system.
    """
    print("=== Testing Hybrid Recommendation System ===\n")
    
    # Initialize hybrid recommender
    recommender = HybridRecommender()
    
    # Test with sample user data
    user_id = 0
    age = 45
    gender = 'Male'
    bmi = 28.5
    hba1c = 6.8
    glucose = 140
    sodium = 145
    potassium = 4.2
    bp = 130
    creatinine = 1.1
    
    print("Testing recommendation pipeline...")
    print(f"User ID: {user_id}")
    print(f"Age: {age}, Gender: {gender}, BMI: {bmi}")
    
    # Get recommendations
    result = recommender.recommend(
        user_id=user_id,
        age=age,
        gender=gender,
        bmi=bmi,
        hba1c=hba1c,
        glucose=glucose,
        sodium=sodium,
        potassium=potassium,
        bp=bp,
        creatinine=creatinine,
        top_n=10
    )
    
    # Print results
    print("\n" + "="*60)
    print("RECOMMENDATION RESULTS")
    print("="*60)
    
    print(f"\nDisease Predictions:")
    for disease, prediction in result['disease_predictions'].items():
        print(f"  {disease}: {prediction}")
    
    print(f"\nDetected Diseases: {result['detected_diseases']}")
    print(f"Allowed Foods Count: {result['allowed_foods_count']}")
    print(f"Recommendation Method: {result['recommendation_method']}")
    
    print(f"\nTop Recommendations:")
    df = recommender.format_recommendations_for_display(result['recommendations'])
    print(df.to_string(index=False))
    
    print(f"\nExplanation:")
    print(recommender.get_recommendation_explanation(result['detected_diseases']))
    
    print("\nHybrid recommender test complete!")


if __name__ == "__main__":
    main()
