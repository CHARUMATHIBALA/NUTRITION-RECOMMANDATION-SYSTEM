import joblib
import sklearn

# Load models
obesity_model = joblib.load('obesity_model.pkl')
kidney_model = joblib.load('kidney_model.pkl')
disease_model = joblib.load('disease_model.pkl')

print("Obesity Model:")
print(f"Type: {type(obesity_model)}")
print(f"Algorithm: {obesity_model.__class__.__name__}")
if hasattr(obesity_model, 'n_features_in_'):
    print(f"Features: {obesity_model.n_features_in_}")
if hasattr(obesity_model, 'classes_'):
    print(f"Classes: {obesity_model.classes_}")
print()

print("Kidney Disease Model:")
print(f"Type: {type(kidney_model)}")
print(f"Algorithm: {kidney_model.__class__.__name__}")
if hasattr(kidney_model, 'n_features_in_'):
    print(f"Features: {kidney_model.n_features_in_}")
if hasattr(kidney_model, 'classes_'):
    print(f"Classes: {kidney_model.classes_}")
print()

print("Diabetes Disease Model:")
print(f"Type: {type(disease_model)}")
print(f"Algorithm: {disease_model.__class__.__name__}")
if hasattr(disease_model, 'n_features_in_'):
    print(f"Features: {disease_model.n_features_in_}")
if hasattr(disease_model, 'classes_'):
    print(f"Classes: {disease_model.classes_}")
