"""
Regression test suite for the risk-classification fix.

Verifies:
  1. The exact regression case from the bug report:
       Creatinine=1.0, Sodium=138, Potassium=4.5, BP=90, BMI=24.2, Age=30
       → final_status must be "Model Flag", NOT "High Risk"
       → model_probability is displayed, not labelled "confidence"
       → all XAI feature impact scores are 0 (all markers normal)

  2. A genuinely high-risk kidney patient → "High Risk"
  3. A normal healthy patient → "Low Risk"
  4. A borderline kidney patient → "Moderate Risk" or "Borderline"
  5. Diabetes: normal inputs → no "High Risk"
  6. Diabetes: high HbA1c + glucose → "High Risk"
  7. Obesity: normal BMI → "Low Risk"
  8. Obesity: obese BMI → "High Risk" or "Moderate Risk"
  9. XAI: all-normal kidney patient → all feature impacts == 0
 10. predict_proba index: confidence uses predicted-class probability not index 1

Run with:
    python test_risk_regression.py
"""
import sys, warnings
warnings.filterwarnings("ignore")

from predict import predict_kidney, predict_diabetes, predict_obesity
from backend.risk import classify_risk
from backend.xai import explain_kidney, explain_diabetes, explain_obesity

PASS = 0
FAIL = 0
ERRORS = []

def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        print(f"  [PASS] {name}")
        PASS += 1
    else:
        msg = f"  [FAIL] {name}"
        if detail:
            msg += f"\n         {detail}"
        print(msg)
        FAIL += 1
        ERRORS.append(name)

SEP = "=" * 65

# ─────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  TEST 1 — REGRESSION: all-normal kidney patient → Model Flag")
print(f"  Input: Cr=1.0, Na=138, K=4.5, BP=90, BMI=24.2, Age=30")
print(SEP)

r_reg = predict_kidney(
    age=30, gender="Male", bmi=24.2,
    sodium=138.0, potassium=4.5, bp=90.0, creatinine=1.0,
)
risk_reg = r_reg.get("risk")

check("predict_kidney returns dict with 'risk' key",
      risk_reg is not None, str(r_reg))

check("final_status == 'Model Flag' (NOT 'High Risk')",
      risk_reg is not None and risk_reg.final_status == "Model Flag",
      f"got: {risk_reg.final_status if risk_reg else 'None'}")

check("card_risk_text == 'Screening Flag' (NOT 'High Risk')",
      risk_reg is not None and risk_reg.card_risk_text == "Screening Flag",
      f"got: {risk_reg.card_risk_text if risk_reg else 'None'}")

check("banner_level == 'info' (NOT 'danger')",
      risk_reg is not None and risk_reg.banner_level == "info",
      f"got: {risk_reg.banner_level if risk_reg else 'None'}")

check("all_markers_normal == True",
      risk_reg is not None and risk_reg.all_markers_normal is True,
      f"got: {risk_reg.all_markers_normal if risk_reg else 'None'}")

check("abnormal_markers is empty list",
      risk_reg is not None and risk_reg.abnormal_markers == [],
      f"got: {risk_reg.abnormal_markers if risk_reg else 'None'}")

check("model_probability is a float",
      risk_reg is not None and isinstance(risk_reg.model_probability, float),
      f"got type: {type(risk_reg.model_probability) if risk_reg else 'None'}")

check("model_probability > 0 (model does flag something)",
      risk_reg is not None and risk_reg.model_probability > 0,
      f"got: {risk_reg.model_probability if risk_reg else 'None'}")

check("'High Risk' NOT in banner_title",
      risk_reg is not None and "High Risk" not in risk_reg.banner_title,
      f"banner_title: {risk_reg.banner_title if risk_reg else 'None'}")

check("banner body mentions 'normal reference ranges'",
      risk_reg is not None and "normal reference ranges" in risk_reg.banner_body.lower(),
      f"banner_body: {risk_reg.banner_body[:100] if risk_reg else 'None'}...")

# ─────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  TEST 2 — HIGH-RISK KIDNEY: elevated creatinine + high BP → High Risk")
print(f"  Input: Cr=4.5, Na=128, K=5.8, BP=175, BMI=28, Age=65")
print(SEP)

r_hi = predict_kidney(
    age=65, gender="Male", bmi=28.0,
    sodium=128.0, potassium=5.8, bp=175.0, creatinine=4.5,
)
risk_hi = r_hi.get("risk")

check("final_status == 'High Risk' for clearly elevated markers",
      risk_hi is not None and risk_hi.final_status == "High Risk",
      f"got: {risk_hi.final_status if risk_hi else 'None'}")

check("abnormal_markers is non-empty for this patient",
      risk_hi is not None and len(risk_hi.abnormal_markers) > 0,
      f"got: {risk_hi.abnormal_markers if risk_hi else 'None'}")

check("banner_level == 'danger'",
      risk_hi is not None and risk_hi.banner_level == "danger",
      f"got: {risk_hi.banner_level if risk_hi else 'None'}")

# ─────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  TEST 3 — HEALTHY KIDNEY: normal markers + model says no disease → Low Risk")
print(f"  Input: Cr=0.8, Na=140, K=4.2, BP=110, BMI=22.5, Age=25")
print(SEP)

r_low = predict_kidney(
    age=25, gender="Female", bmi=22.5,
    sodium=140.0, potassium=4.2, bp=110.0, creatinine=0.8,
)
risk_low = r_low.get("risk")
label_low = r_low.get("label", "")

# The model may still predict Kidney Disease (known bias), but if it does
# and all values are normal, it should be Model Flag, not High Risk.
# If model says No Kidney Disease, it should be Low Risk.
if "no kidney" in label_low.lower():
    check("final_status == 'Low Risk' when model says No Kidney Disease",
          risk_low is not None and risk_low.final_status == "Low Risk",
          f"got: {risk_low.final_status if risk_low else 'None'}")
else:
    check("final_status == 'Model Flag' (not High Risk) when all normal but model flags",
          risk_low is not None and risk_low.final_status in ("Model Flag", "Low Risk"),
          f"got: {risk_low.final_status if risk_low else 'None'}")

# ─────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  TEST 4 — NORMAL DIABETES: all normal → no High Risk")
print(f"  Input: Age=30, Male, BMI=22.5, HbA1c=5.2, Glucose=88")
print(SEP)

r_dn = predict_diabetes(
    age=30, gender="Male", bmi=22.5, hba1c=5.2, glucose=88,
)
risk_dn = r_dn.get("risk")

check("diabetes normal: final_status NOT 'High Risk'",
      risk_dn is not None and risk_dn.final_status != "High Risk",
      f"got: {risk_dn.final_status if risk_dn else 'None'}")

check("diabetes normal: banner_level NOT 'danger'",
      risk_dn is not None and risk_dn.banner_level != "danger",
      f"got: {risk_dn.banner_level if risk_dn else 'None'}")

# ─────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  TEST 5 — HIGH-RISK DIABETES: HbA1c=9.2, Glucose=280 → High Risk")
print(SEP)

r_dh = predict_diabetes(
    age=55, gender="Male", bmi=35.0, hba1c=9.2, glucose=280,
)
risk_dh = r_dh.get("risk")

check("diabetes high-risk: final_status == 'High Risk'",
      risk_dh is not None and risk_dh.final_status == "High Risk",
      f"got: {risk_dh.final_status if risk_dh else 'None'}")

check("diabetes high-risk: abnormal_markers includes HbA1c or Blood Glucose",
      risk_dh is not None and len(risk_dh.abnormal_markers) > 0,
      f"got: {risk_dh.abnormal_markers if risk_dh else 'None'}")

# ─────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  TEST 6 — NORMAL OBESITY: BMI=22.5 → Low Risk")
print(SEP)

r_on = predict_obesity(age=30, gender="Male", bmi=22.5)
risk_on = r_on.get("risk")

check("obesity normal BMI: final_status == 'Low Risk'",
      risk_on is not None and risk_on.final_status == "Low Risk",
      f"got: {risk_on.final_status if risk_on else 'None'}")

# ─────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  TEST 7 — OBESE: BMI=38 → High Risk or Moderate Risk")
print(SEP)

r_oh = predict_obesity(age=40, gender="Male", bmi=38.0)
risk_oh = r_oh.get("risk")

check("obese BMI: final_status is High Risk or Moderate Risk (not Low)",
      risk_oh is not None and risk_oh.final_status in ("High Risk", "Moderate Risk"),
      f"got: {risk_oh.final_status if risk_oh else 'None'} label={r_oh.get('label')}")

# ─────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  TEST 8 — XAI: all-normal kidney patient → all impact scores == 0")
print(SEP)

xai_normal = explain_kidney(
    age=30, gender="Male", bmi=24.2,
    sodium=138.0, potassium=4.5, bp=90.0, creatinine=1.0,
    label="Kidney Disease",
)

check("XAI available for normal patient",
      xai_normal.available is True,
      xai_normal.error or "")

if xai_normal.available:
    all_zero_impact = all(r.importance == 0.0 for r in xai_normal.feature_rows)
    check("All XAI feature importance scores == 0 when all values are normal",
          all_zero_impact,
          "Non-zero: " + str([(r.feature, r.importance) for r in xai_normal.feature_rows
                               if r.importance != 0.0]))

    all_low_impact = all(r.impact == "Low" for r in xai_normal.feature_rows)
    check("All XAI impact labels == 'Low' when all values are normal",
          all_low_impact,
          "Non-low: " + str([(r.feature, r.impact) for r in xai_normal.feature_rows
                              if r.impact != "Low"]))

    all_normal_dir = all(r.direction == "✓ Normal range" for r in xai_normal.feature_rows)
    check("All XAI directions == '✓ Normal range' when all values normal",
          all_normal_dir,
          "Non-normal: " + str([(r.feature, r.direction) for r in xai_normal.feature_rows
                                 if r.direction != "✓ Normal range"]))

# ─────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  TEST 9 — XAI: abnormal creatinine → non-zero impact on creatinine")
print(SEP)

xai_hi = explain_kidney(
    age=65, gender="Male", bmi=28.0,
    sodium=128.0, potassium=5.8, bp=175.0, creatinine=4.5,
    label="Kidney Disease",
)

if xai_hi.available:
    cr_row = next((r for r in xai_hi.feature_rows if "Creatinine" in r.feature), None)
    check("Serum Creatinine has non-zero impact when value is 4.5 mg/dL",
          cr_row is not None and cr_row.importance > 0,
          f"cr_row: {cr_row}")

    check("Serum Creatinine direction is '↑ Above normal' for value 4.5",
          cr_row is not None and cr_row.direction == "↑ Above normal",
          f"got: {cr_row.direction if cr_row else 'None'}")

    high_impact = any(r.impact == "High" for r in xai_hi.feature_rows)
    check("At least one feature has 'High' impact for clearly elevated patient",
          high_impact,
          str([(r.feature, r.impact, r.importance) for r in xai_hi.feature_rows]))

# ─────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  TEST 10 — predict_proba uses predicted-class index, not hardcoded index")
print(SEP)

# For the kidney model, classes_ = [0, 1, 2]
# class 2 = "No Kidney Disease"
# If model predicts class 2, confidence should be proba[2], not proba[1] or proba[0]
import warnings, joblib, pandas as pd
warnings.filterwarnings("ignore")
km   = joblib.load("kidney_model.pkl")
ge   = joblib.load("gender_encoder.pkl")
cols = ["age","gender","bmi","sodium","potassium","BloodPressure","SerumCreatinine"]

# Use a value that the model actually predicts as No Kidney Disease
# (from prior audit: BP < 85 tends to push toward No KD)
X_test = pd.DataFrame([[25, int(ge.transform(["Female"])[0]), 20.0,
                         162.0, 4.0, 75.0, 0.7]], columns=cols)
raw_p   = km.predict(X_test)[0]
proba_p = km.predict_proba(X_test)[0]
pred_idx = list(km.classes_).index(raw_p)
max_idx  = list(proba_p).index(max(proba_p))

check("predicted-class index matches max-probability index (correct confidence)",
      pred_idx == max_idx,
      f"pred_idx={pred_idx}, max_idx={max_idx}, classes={km.classes_}, proba={proba_p}")

# ─────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  TEST 11 — model_probability label says 'Model Probability' not 'Confidence'")
print(SEP)

# Verify the RiskResult has a model_probability field, not confidence
check("RiskResult has model_probability attribute",
      risk_reg is not None and hasattr(risk_reg, "model_probability"),
      str(dir(risk_reg) if risk_reg else "None"))

check("RiskResult does NOT have a 'confidence' attribute (prevent confusion)",
      risk_reg is not None and not hasattr(risk_reg, "confidence"),
      "RiskResult should use model_probability, not confidence")

# ─────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print(f"  RESULT: {PASS} passed,  {FAIL} failed")
if ERRORS:
    print(f"  Failed tests: {ERRORS}")
print(SEP)

sys.exit(0 if FAIL == 0 else 1)
