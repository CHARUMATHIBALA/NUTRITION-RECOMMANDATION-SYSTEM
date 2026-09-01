from predict import *

bmi = 22.66

print(
    predict_obesity(
        22,
        "Female",
        bmi
    )
)

print(
    predict_diabetes(
        22,
        "Female",
        bmi,
        6.8,
        145
    )
)

print(
    predict_kidney(
        18,
        "Female",
        18.0,
        120,
        3.2,
        80,
        0.4
    )
)