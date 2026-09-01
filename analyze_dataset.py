import pandas as pd

df = pd.read_csv('food_dataset.csv')

print(f'Total records: {len(df)}')
print(f'Columns: {list(df.columns)}')
print(f'\nDataset shape: {df.shape}')
print(f'\nMeal types: {df["MealType"].unique()}')
print(f'\nMissing values:\n{df.isnull().sum()}')
print(f'\nBasic statistics:\n{df.describe()}')
