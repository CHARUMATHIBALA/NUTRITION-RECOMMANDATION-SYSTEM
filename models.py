import joblib
import pandas as pd

# -------------------------
# Load ML Models
# -------------------------
obesity_model = joblib.load("obesity_model.pkl")

kidney_model = joblib.load("kidney_model.pkl")

disease_model = joblib.load("disease_model.pkl")

obesity_encoder = joblib.load("obesity_encoder.pkl")
# -------------------------
# Load Encoders
# -------------------------
gender_encoder = joblib.load("gender_encoder.pkl")

disease_encoder = joblib.load("disease_encoder.pkl")

kidney_encoder = joblib.load("kidney_encoder.pkl")


# -------------------------
# Load Food Dataset
# -------------------------
food_df = pd.read_csv("food_dataset.csv")

# Encode Gender
def encode_gender(gender):
    return gender_encoder.transform([gender])[0]

def decode_obesity(prediction):
    return obesity_encoder.inverse_transform(prediction)[0]
# Decode Diabetes Prediction
def decode_diabetes(prediction):
    return disease_encoder.inverse_transform(prediction)[0]


# Decode Kidney Prediction
def decode_kidney(prediction):
    label = kidney_encoder.inverse_transform(prediction)[0]
    return label.strip()  # Remove any extra whitespace/tabs