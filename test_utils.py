from utils import *

bmi = calculate_bmi(80, 160)

print("BMI :", bmi)

print("Category :", bmi_category(bmi))

bmr = calculate_bmr(22, "Female", 80, 160)

print("BMR :", bmr)

print("TDEE :", calculate_tdee(bmr, 1.55))