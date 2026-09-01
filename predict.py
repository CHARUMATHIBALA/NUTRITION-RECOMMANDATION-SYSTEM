import pandas as pd
import models


# Custom mapping for obesity model predictions (based on test cases)
obesity_label_map = {
    0: "Underweight",
    1: "Normal Weight",
    2: "Obese Class I",
    3: "Obese Class II",
    4: "Obese Class III",
    5: "Overweight",
    6: "Overweight"
}

# ----------------------------
# Obesity Prediction
# ----------------------------
def predict_obesity(age, gender, bmi):

    gender = models.encode_gender(gender)

    X = pd.DataFrame(
        [[age, gender, bmi]],
        columns=[
            "age",
            "gender",
            "bmi"
        ]
    )

    prediction = models.obesity_model.predict(X)
    pred_value = prediction[0]

    # Get confidence score if available
    confidence = None
    if hasattr(models.obesity_model, 'predict_proba'):
        proba = models.obesity_model.predict_proba(X)[0]
        confidence = round(max(proba) * 100, 1)

    # Return the mapped label, or "Unknown" if not in the map
    label = obesity_label_map.get(pred_value, str(pred_value))
    return {"label": label, "confidence": confidence}


# ----------------------------
# Diabetes Prediction
# ----------------------------
def predict_diabetes(age, gender, bmi, hba1c, glucose):

    gender = models.encode_gender(gender)

    X = pd.DataFrame(
        [[age, gender, bmi, hba1c, glucose]],
        columns=[
            "age",
            "gender",
            "bmi",
            "HbA1c",
            "blood glucose"
        ]
    )

    prediction = models.disease_model.predict(X)

    # Get confidence score if available
    confidence = None
    if hasattr(models.disease_model, 'predict_proba'):
        proba = models.disease_model.predict_proba(X)[0]
        confidence = round(max(proba) * 100, 1)

    label = models.decode_diabetes(prediction)
    return {"label": label, "confidence": confidence}


# ----------------------------
# Kidney Disease Prediction
# ----------------------------
def predict_kidney(age, gender, bmi, sodium, potassium, bp, creatinine):

    gender = models.encode_gender(gender)

    X = pd.DataFrame(
        [[
            age,
            gender,
            bmi,
            sodium,
            potassium,
            bp,
            creatinine
        ]],
        columns=[
            "age",
            "gender",
            "bmi",
            "sodium",
            "potassium",
            "BloodPressure",
            "SerumCreatinine"
        ]
    )

    prediction = models.kidney_model.predict(X)

    # Get confidence score if available
    confidence = None
    if hasattr(models.kidney_model, 'predict_proba'):
        proba = models.kidney_model.predict_proba(X)[0]
        confidence = round(max(proba) * 100, 1)

    label = models.decode_kidney(prediction)
    return {"label": label, "confidence": confidence}