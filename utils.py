import math

# -----------------------------
# BMI Calculation
# -----------------------------
def calculate_bmi(weight, height):
    """
    weight -> kg
    height -> cm
    """
    height = height / 100
    bmi = weight / (height ** 2)
    return round(bmi, 2)


# -----------------------------
# BMI Category
# -----------------------------
def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"


# -----------------------------
# BMR Calculation
# -----------------------------
def calculate_bmr(age, gender, weight, height):
    if gender.lower() == "male":
        return 10 * weight + 6.25 * height - 5 * age + 5
    else:
        return 10 * weight + 6.25 * height - 5 * age - 161


# -----------------------------
# TDEE Calculation
# -----------------------------
def calculate_tdee(bmr, activity_factor):
    return round(bmr * activity_factor, 2)