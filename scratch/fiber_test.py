import pandas as pd
from recommendation import IntelligentNutritionRecommender
import config

# Create dummy row as pandas Series
row = pd.Series({
    "Free Sugar (g)": 0,
    "Carbohydrates (g)": 0,
    "Fibre (g)": 5,
    "Sodium (mg)": 0,
    "Protein (g)": 0,
    "Calories (kcal)": 0,
    "Fats (g)": 0,
})

rec = IntelligentNutritionRecommender()
# Set conditions for both diabetes and obesity
conditions = {'has_diabetes': True, 'has_kidney_disease': False, 'has_obesity': True}
score = rec._calculate_nutritional_score(row, conditions)
print('Score with both diabetes and obesity:', score)
