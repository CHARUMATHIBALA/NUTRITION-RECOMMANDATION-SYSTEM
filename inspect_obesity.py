import joblib, os, json, sys
BASE_DIR = r'd:\Healthcare_Project'
model_path = os.path.join(BASE_DIR, 'obesity_model.pkl')
model = joblib.load(model_path)
print('Model type:', type(model))
print('Model classes:', getattr(model, 'classes_', None))
print('Feature names in:', getattr(model, 'feature_names_in_', None))
print('Number of features in:', getattr(model, 'n_features_in_', None))
print('Estimator params:', model.get_params())
# try predicting on synthetic data
import numpy as np, pandas as pd
age = np.random.uniform(10,80,10)
gender = np.random.randint(0,2,10)
bmi = np.random.uniform(12,55,10)
X_demo = pd.DataFrame({'age': age, 'gender': gender, 'bmi': bmi})
print('Predict shape', model.predict(X_demo).shape)
