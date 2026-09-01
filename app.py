"""app.py — Smart Health Dashboard main entry point.

Flow
----
1. Page config + CSS injection
2. Theme initialisation (light / dark)
3. Authentication gate  — stops here if not logged in
4. Sidebar  — all patient inputs, medical params, goals, settings
5. Analysis — triggered by "Analyse Health" button
6. Main area — 4 tabs: Patient Profile · Health Summary · Predictions · Recommendations
"""

import streamlit as st
import pandas as pd

# ── Local utilities ──────────────────────────────────────────────────
from utils import calculate_bmi, bmi_category, calculate_bmr, calculate_tdee
from predict import predict_diabetes, predict_kidney, predict_obesity
from meal_planner import generate_comprehensive_recommendations
import components

# ── NCF (optional — fails gracefully if not installed) ───────────────
try:
    from ncf_integration.utils.hybrid_recommender import HybridRecommender
    NCF_AVAILABLE = True
except Exception as e:
    NCF_AVAILABLE = False
    st.info("NCF integration not available. Recommendations will be based on rule‑based system.")

# ── Backend helpers ──────────────────────────────────────────────────
from backend.auth import authenticate, logout
from backend.database import save_analysis, get_history  # wired below

# ── XAI (Explainable AI) ─────────────────────────────────────────────
try:
    from backend.xai import explain_diabetes, explain_obesity, explain_kidney
    XAI_AVAILABLE = True
except Exception:
    XAI_AVAILABLE = False

# ════════════════════════════════════════════════════════════════════
#  PAGE CONFIG  (must be the very first Streamlit call)
# ════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Smart Health Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject stylesheet immediately (covers the login page too)
components.load_css()

# ════════════════════════════════════════════════════════════════════
#  THEME
# ════════════════════════════════════════════════════════════════════
if "theme" not in st.session_state:
    st.session_state.theme = "light"

# Push theme attribute to the DOM so [data-theme] CSS selectors work
st.markdown(
    f"<script>document.documentElement.setAttribute('data-theme',"
    f"'{st.session_state.theme}');</script>",
    unsafe_allow_html=True,
)

# ════════════════════════════════════════════════════════════════════
#  AUTHENTICATION
# ════════════════════════════════════════════════════════════════════
name, auth_status, username = authenticate()

if not auth_status:
    # Styled login landing — the login widget itself is rendered by
    # streamlit-authenticator inside authenticate() above
    st.markdown(
        """
        <div class='login-wrapper'>
            <span class='login-logo'>🩺</span>
            <div class='login-title'>Smart Health Dashboard</div>
            <div class='login-subtitle'>
                AI-Powered Nutrition &amp; Disease Risk Analysis
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if auth_status is False:
        st.error("❌ Incorrect username or password. Please try again.")
    st.stop()

# ════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════
with st.sidebar:

    # Brand block
    st.markdown(
        """
        <div class='sidebar-brand-block'>
            <div class='sidebar-logo'>🩺</div>
            <div class='sidebar-brand'>Smart Health</div>
            <div class='sidebar-tagline'>Nutrition &amp; Risk Dashboard</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Logout
    if st.button("🚪 Logout", use_container_width=True):
        logout()
        st.rerun()

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ── Patient Information ──────────────────────────────────────────
    st.subheader("Patient Information")
    pat_name = st.text_input(
        "Patient Name",
        value=st.session_state.get("pat_name", ""),
        max_chars=100,
        help="Enter the full name of the patient (required).",
    )
    age = st.number_input(
        "Age",
        min_value=1, max_value=120,
        value=int(st.session_state.get("age", 30)),
        step=1,
        help="Patient age in years (1 – 120).",
    )
    gender = st.selectbox(
        "Gender", ["Male", "Female"],
        index=0 if st.session_state.get("gender", "Male") == "Male" else 1,
        help="Biological sex used for BMR and ML model calculations.",
    )
    height = st.number_input(
        "Height (cm)",
        min_value=100, max_value=250,
        value=int(st.session_state.get("height", 170)),
        step=1,
        help="Patient height in centimetres (100 – 250 cm).",
    )
    weight = st.number_input(
        "Weight (kg)",
        min_value=30, max_value=300,
        value=int(st.session_state.get("weight", 70)),
        step=1,
        help="Patient weight in kilograms (30 – 300 kg).",
    )
    activity = st.selectbox(
        "Activity Level",
        ["Sedentary", "Light", "Moderate", "Active", "Very Active"],
        index=["Sedentary", "Light", "Moderate", "Active", "Very Active"].index(
            st.session_state.get("activity", "Sedentary")
        ),
        help="Daily physical activity level used to calculate caloric needs.",
    )

    # ── Medical Parameters ───────────────────────────────────────────
    st.subheader("Medical Parameters")
    hba1c = st.number_input(
        "HbA1c (%)",
        min_value=3.0, max_value=15.0,
        value=float(st.session_state.get("hba1c", 5.5)),
        step=0.1,
        format="%.1f",
        help="Glycated haemoglobin — normal < 5.7 %, pre-diabetes 5.7–6.4 %, diabetes ≥ 6.5 %.",
    )
    glucose = st.number_input(
        "Blood Glucose (mg/dL)",
        min_value=50, max_value=500,
        value=int(st.session_state.get("glucose", 90)),
        step=1,
        help="Fasting blood glucose — normal 70–99 mg/dL, pre-diabetes 100–125 mg/dL, diabetes ≥ 126 mg/dL.",
    )
    bp = st.number_input(
        "Systolic BP (mmHg)",
        min_value=80, max_value=200,
        value=int(st.session_state.get("bp", 120)),
        step=1,
        help="Systolic (upper) blood pressure in mmHg — normal < 120 mmHg.",
    )
    bp_diastolic = st.number_input(
        "Diastolic BP (mmHg)",
        min_value=40, max_value=140,
        value=int(st.session_state.get("bp_diastolic", 80)),
        step=1,
        help="Diastolic (lower) blood pressure in mmHg — normal < 80 mmHg. "
             "Recorded for your profile; the disease models use systolic BP.",
    )
    creatinine = st.number_input(
        "Serum Creatinine (mg/dL)",
        min_value=0.1, max_value=15.0,
        value=float(st.session_state.get("creatinine", 1.0)),
        step=0.1,
        format="%.1f",
        help="Kidney function marker — normal 0.7–1.2 mg/dL (male), 0.5–1.0 mg/dL (female). Source: Mayo Clinic.",
    )
    sodium = st.number_input(
        "Sodium (mEq/L)",
        min_value=115.0, max_value=170.0,
        value=float(st.session_state.get("sodium", 138.0)),
        step=0.5,
        format="%.1f",
        help="Serum sodium — normal 135–145 mEq/L. Values below 115 mEq/L indicate severe hyponatremia.",
    )
    potassium = st.number_input(
        "Potassium (mEq/L)",
        min_value=2.0, max_value=7.0,
        value=float(st.session_state.get("potassium", 4.5)),
        step=0.1,
        format="%.1f",
        help="Serum potassium — normal 3.5–5.0 mEq/L.",
    )

    # ── Goals ────────────────────────────────────────────────────────
    st.subheader("Goals")
    goal = st.radio(
        "Weight Goal",
        ["Weight Loss", "Weight Gain", "Weight Maintenance"],
        index=0,
    )

    # ── Appearance ───────────────────────────────────────────────────
    st.subheader("Appearance")
    dark_mode = st.checkbox(
        "🌙 Dark Mode",
        value=st.session_state.get("theme", "light") == "dark",
    )
    st.session_state.theme = "dark" if dark_mode else "light"

    # ── Disease Selection ────────────────────────────────────────────
    st.subheader("Disease Selection")
    auto_predict = st.checkbox("Auto Prediction", value=True,
                               help="Use AI models to predict diseases automatically")
    manual_override = st.checkbox("Manual Override", value=False,
                                  help="Manually select diseases instead of AI prediction")
    manual_diseases: list = []
    if manual_override:
        manual_diseases = st.multiselect(
            "Select Diseases",
            ["Diabetes", "Obesity", "Kidney Disease"],
            help="These will replace the AI predictions",
        )

    # ── AI Recommendation (NCF) ──────────────────────────────────────
    if NCF_AVAILABLE:
        st.subheader("AI Recommendation")
        use_ncf = st.checkbox(
            "Neural Collaborative Filtering",
            value=False,
            help="Enable AI-powered personalised food recommendations",
        )
        st.session_state.use_ncf = use_ncf

    # ── Region ───────────────────────────────────────────────────────
    st.subheader("Region")
    indian_states = [
        "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar",
        "Chhattisgarh", "Goa", "Gujarat", "Haryana", "Himachal Pradesh",
        "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra",
        "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
        "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
        "Uttar Pradesh", "Uttarakhand", "West Bengal",
        "Andaman and Nicobar Islands", "Chandigarh",
        "Dadra and Nagar Haveli and Daman and Diu", "Delhi",
        "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry",
    ]
    region = st.selectbox("State / Union Territory", indian_states, index=0)

    # ── Action Buttons ───────────────────────────────────────────────
    st.subheader("Actions")
    analyze = st.button("🔍 Analyse Health", use_container_width=True, type="primary")
    reset   = st.button("♻️ Reset Results",  use_container_width=True)

# ════════════════════════════════════════════════════════════════════
#  Re-apply theme after sidebar renders
# ════════════════════════════════════════════════════════════════════
st.markdown(
    f"<script>document.documentElement.setAttribute('data-theme',"
    f"'{st.session_state.theme}');</script>",
    unsafe_allow_html=True,
)

# ════════════════════════════════════════════════════════════════════
#  RESET
# ════════════════════════════════════════════════════════════════════
if reset:
    keep = {"theme", "authenticator", "authentication_status", "name", "username"}
    for key in list(st.session_state.keys()):
        if key not in keep:
            del st.session_state[key]
    st.rerun()

# ════════════════════════════════════════════════════════════════════
#  HERO BANNER
# ════════════════════════════════════════════════════════════════════
components.hero_banner(
    title="Smart Health Dashboard",
    subtitle="AI-Powered Personalised Nutrition & Disease Risk Analysis",
    username=name or username or "",
)

# ════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ════════════════════════════════════════════════════════════════════
ACTIVITY_FACTOR = {
    "Sedentary":   1.2,
    "Light":       1.375,
    "Moderate":    1.55,
    "Active":      1.725,
    "Very Active": 1.9,
}

# ════════════════════════════════════════════════════════════════════
#  INPUT VALIDATION
# ════════════════════════════════════════════════════════════════════

def _validate_inputs(
    pat_name, age, height, weight,
    hba1c, glucose, bp, bp_diastolic,
    creatinine, sodium, potassium,
) -> list:
    """Return a list of human-readable error strings.

    Returns an empty list when all inputs are valid.
    Only called when the user presses 'Analyse Health'.
    Does NOT modify any widget or session state.
    """
    errors = []

    # ── Patient name ─────────────────────────────────────────────────
    name_clean = str(pat_name).strip() if pat_name else ""
    if not name_clean:
        errors.append("❗ **Patient Name** is required. Please enter the patient's full name.")
    elif len(name_clean) < 2:
        errors.append("❗ **Patient Name** must be at least 2 characters.")
    elif not any(c.isalpha() for c in name_clean):
        errors.append("❗ **Patient Name** must contain at least one letter.")

    # ── Age ──────────────────────────────────────────────────────────
    try:
        age_v = int(age)
        if age_v < 1 or age_v > 120:
            errors.append("❗ **Age** must be between 1 and 120 years.")
    except (TypeError, ValueError):
        errors.append("❗ **Age** must be a whole number between 1 and 120.")

    # ── Height ───────────────────────────────────────────────────────
    try:
        h_v = float(height)
        if h_v < 100 or h_v > 250:
            errors.append("❗ **Height** must be between 100 and 250 cm.")
    except (TypeError, ValueError):
        errors.append("❗ **Height** must be a number between 100 and 250 cm.")

    # ── Weight ───────────────────────────────────────────────────────
    try:
        w_v = float(weight)
        if w_v < 30 or w_v > 300:
            errors.append("❗ **Weight** must be between 30 and 300 kg.")
    except (TypeError, ValueError):
        errors.append("❗ **Weight** must be a number between 30 and 300 kg.")

    # ── HbA1c ────────────────────────────────────────────────────────
    try:
        h1c = float(hba1c)
        if h1c < 3.0 or h1c > 15.0:
            errors.append("❗ **HbA1c** must be between 3.0 % and 15.0 %.")
    except (TypeError, ValueError):
        errors.append("❗ **HbA1c** must be a number between 3.0 and 15.0.")

    # ── Blood Glucose ────────────────────────────────────────────────
    try:
        glc = float(glucose)
        if glc < 50 or glc > 500:
            errors.append("❗ **Blood Glucose** must be between 50 and 500 mg/dL.")
    except (TypeError, ValueError):
        errors.append("❗ **Blood Glucose** must be a number between 50 and 500 mg/dL.")

    # ── Systolic BP ──────────────────────────────────────────────────
    try:
        sbp = int(bp)
        if sbp < 80 or sbp > 200:
            errors.append("❗ **Systolic BP** must be between 80 and 200 mmHg.")
    except (TypeError, ValueError):
        errors.append("❗ **Systolic BP** must be a whole number between 80 and 200 mmHg.")

    # ── Diastolic BP ─────────────────────────────────────────────────
    try:
        dbp = int(bp_diastolic)
        if dbp < 40 or dbp > 140:
            errors.append("❗ **Diastolic BP** must be between 40 and 140 mmHg.")
        else:
            # cross-field: diastolic must be lower than systolic
            try:
                if dbp >= int(bp):
                    errors.append(
                        f"❗ **Diastolic BP** ({bp_diastolic} mmHg) must be lower than "
                        f"**Systolic BP** ({bp} mmHg)."
                    )
            except (TypeError, ValueError):
                pass  # systolic already flagged separately
    except (TypeError, ValueError):
        errors.append("❗ **Diastolic BP** must be a whole number between 40 and 140 mmHg.")

    # ── Serum Creatinine ─────────────────────────────────────────────
    try:
        cr = float(creatinine)
        if cr < 0.1 or cr > 15.0:
            errors.append("❗ **Serum Creatinine** must be between 0.1 and 15.0 mg/dL.")
    except (TypeError, ValueError):
        errors.append("❗ **Serum Creatinine** must be a number between 0.1 and 15.0.")

    # ── Sodium ───────────────────────────────────────────────────────
    try:
        na = float(sodium)
        if na < 115.0 or na > 170.0:
            errors.append("❗ **Sodium** must be between 115.0 and 170.0 mEq/L.")
    except (TypeError, ValueError):
        errors.append("❗ **Sodium** must be a number between 115.0 and 170.0 mEq/L.")

    # ── Potassium ────────────────────────────────────────────────────
    try:
        k = float(potassium)
        if k < 2.0 or k > 7.0:
            errors.append("❗ **Potassium** must be between 2.0 and 7.0 mEq/L.")
    except (TypeError, ValueError):
        errors.append("❗ **Potassium** must be a number between 2.0 and 7.0 mEq/L.")

    return errors


# ════════════════════════════════════════════════════════════════════
#  ANALYSIS  — triggered by "Analyse Health"
# ════════════════════════════════════════════════════════════════════
if analyze:

    # ── Run validation before any calculation ────────────────────────
    _errors = _validate_inputs(
        pat_name=pat_name, age=age, height=height, weight=weight,
        hba1c=hba1c, glucose=glucose, bp=bp, bp_diastolic=bp_diastolic,
        creatinine=creatinine, sodium=sodium, potassium=potassium,
    )

    if _errors:
        # Show every error in the main area so the user can see them
        # without having to scroll the sidebar.
        st.error("### ⚠️ Please fix the following before running the analysis:")
        for _err in _errors:
            st.markdown(_err)
        # Also echo a short note inside the sidebar
        with st.sidebar:
            st.error(f"⚠️ {len(_errors)} validation error(s). See main panel.")

    else:
        # All inputs valid — persist current values to session state so
        # widgets restore correctly after a rerun, then run full analysis.
        st.session_state.update({
            "pat_name":     str(pat_name).strip(),
            "age":          int(age),
            "gender":       gender,
            "height":       int(height),
            "weight":       int(weight),
            "activity":     activity,
            "hba1c":        float(hba1c),
            "glucose":      int(glucose),
            "bp":           int(bp),
            "bp_diastolic": int(bp_diastolic),
            "creatinine":   float(creatinine),
            "sodium":       float(sodium),
            "potassium":    float(potassium),
        })

    if not _errors:
        try:
            # ── Core calculations ─────────────────────────────────────────
            bmi     = calculate_bmi(weight, height)
            bmi_cat = bmi_category(bmi)
            bmr     = calculate_bmr(age, gender, weight, height)
            tdee    = calculate_tdee(bmr, ACTIVITY_FACTOR[activity])

            # ── Disease predictions ───────────────────────────────────────
            # Run ML models always (needed for recommendations even with override)
            obesity_result  = predict_obesity(age, gender, bmi)
            diabetes_result = predict_diabetes(age, gender, bmi, hba1c, glucose)
            kidney_result   = predict_kidney(age, gender, bmi, sodium, potassium, bp, creatinine)

            # Extract labels and confidence scores
            obesity_label = obesity_result["label"] if isinstance(obesity_result, dict) else obesity_result
            obesity_confidence = obesity_result["confidence"] if isinstance(obesity_result, dict) else None

            diabetes_label = diabetes_result["label"] if isinstance(diabetes_result, dict) else diabetes_result
            diabetes_confidence = diabetes_result["confidence"] if isinstance(diabetes_result, dict) else None

            kidney_label = kidney_result["label"] if isinstance(kidney_result, dict) else kidney_result
            kidney_confidence = kidney_result["confidence"] if isinstance(kidney_result, dict) else None

            # Apply manual override if selected
            if manual_override and manual_diseases:
                # Display overridden labels in the UI
                obesity  = {"label": "Obesity (Manual)", "confidence": None} if "Obesity" in manual_diseases else obesity_result
                diabetes = {"label": "Diabetes (Manual)", "confidence": None} if "Diabetes" in manual_diseases else diabetes_result
                kidney   = {"label": "Kidney Disease (Manual)", "confidence": None} if "Kidney Disease" in manual_diseases else kidney_result
                # Build diseases list from manual selection
                diseases = list(manual_diseases)
            else:
                obesity  = obesity_result
                diabetes = diabetes_result
                kidney   = kidney_result
                # Build diseases list from model outputs
                diseases = []
                diabetes_str = str(diabetes_label).lower() if diabetes_label is not None else ""
                kidney_str   = str(kidney_label).lower() if kidney_label is not None else ""
                obesity_str  = str(obesity_label).lower() if obesity_label is not None else ""
                if diabetes_str == "diabetes":
                    diseases.append("Diabetes")
                if kidney_str == "kidney disease":
                    diseases.append("Kidney Disease")
                if obesity_str in [
                    "obese class i", "obese class ii", "obese class iii", "overweight"
                ]:
                    diseases.append("Obesity")
                if not diseases:
                    diseases = ["Normal"]

            # ── NCF recommendations (optional) ────────────────────────────
            use_ncf = st.session_state.get("use_ncf", False) and NCF_AVAILABLE
            if use_ncf:
                try:
                    if "hybrid_recommender" not in st.session_state:
                        st.session_state.hybrid_recommender = HybridRecommender()
                    hr = st.session_state.hybrid_recommender
                    try:
                        user_id = (
                            int(username) if str(username).isdigit()
                            else hash(str(username)) % 1000
                        )
                    except Exception:
                        user_id = 0
                    ncf_result = hr.recommend(
                        user_id=user_id, age=age, gender=gender, bmi=bmi,
                        hba1c=hba1c, glucose=glucose, sodium=sodium,
                        potassium=potassium, bp=bp, creatinine=creatinine, top_n=20,
                    )
                    st.session_state.ncf_recommendations = (
                        hr.format_recommendations_for_display(ncf_result["recommendations"])
                    )
                    st.session_state.ncf_explanation = hr.get_recommendation_explanation(
                        ncf_result["detected_diseases"]
                    )
                    st.session_state.ncf_diseases = ncf_result["detected_diseases"]
                except Exception as exc:
                    st.error(f"NCF error: {exc}. Using rule-based recommendations.")
                    use_ncf = False

            # ── Rule-based recommendations ────────────────────────────────
            recommendations = generate_comprehensive_recommendations(
                diseases=diseases, age=age, gender=gender,
                height=height, weight=weight, bmi=bmi,
                activity_level=activity, daily_calories=tdee,
                hba1c=hba1c, glucose=glucose, bp=bp,
                sodium=sodium, potassium=potassium, creatinine=creatinine,
            )

            # ── Build report_data (used for PDF/JSON export) ──────────────
            report_data = {
                "personal": {
                    "name":     pat_name or "—",
                    "age":      age,
                    "gender":   gender,
                    "height":   f"{height} cm",
                    "weight":   f"{weight} kg",
                    "activity": activity,
                    "region":   region,
                },
                "metrics": {
                    "BMI":                    round(bmi, 2),
                    "BMI Category":           bmi_cat,
                    "BMR (kcal)":             round(bmr, 2),
                    "TDEE (kcal)":            round(tdee, 2),
                    "Water Intake (L)":       recommendations.get("water_intake"),
                    "Protein Requirement (g)":recommendations.get("protein_requirement"),
                },
                "predictions": {
                    "Obesity": {
                        "label": obesity_label if isinstance(obesity, dict) else str(obesity),
                        "confidence": obesity_confidence if isinstance(obesity, dict) else None
                    },
                    "Diabetes": {
                        "label": diabetes_label if isinstance(diabetes, dict) else str(diabetes),
                        "confidence": diabetes_confidence if isinstance(diabetes, dict) else None
                    },
                    "Kidney Disease": {
                        "label": kidney_label if isinstance(kidney, dict) else str(kidney),
                        "confidence": kidney_confidence if isinstance(kidney, dict) else None
                    },
                },
                "nutrition_tips":   recommendations.get("nutrition_tips", []),
                "foods_to_avoid":   recommendations.get("foods_to_avoid", []),
            }

            # ── Persist to SQLite (wired!) ────────────────────────────────
            try:
                save_analysis(
                    user_id=None,         # anonymous until user table is populated
                    patient_name=pat_name or username or "unknown",
                    data=report_data,
                )
            except Exception:
                pass  # database errors must never crash the UI

            # ── Store everything in session state ─────────────────────────
            st.session_state.report_data = report_data
            st.session_state.analysis = {
                "bmi": bmi, "bmi_cat": bmi_cat, "bmr": bmr, "tdee": tdee,
                "obesity": obesity, "diabetes": diabetes, "kidney": kidney,
                "obesity_label": obesity_label, "obesity_confidence": obesity_confidence,
                "diabetes_label": diabetes_label, "diabetes_confidence": diabetes_confidence,
                "kidney_label": kidney_label, "kidney_confidence": kidney_confidence,
                # RiskResult objects — single source of truth for Tab 2
                "obesity_risk":  obesity_result.get("risk")  if isinstance(obesity_result,  dict) else None,
                "diabetes_risk": diabetes_result.get("risk") if isinstance(diabetes_result, dict) else None,
                "kidney_risk":   kidney_result.get("risk")   if isinstance(kidney_result,   dict) else None,
                "diseases": diseases, "recommendations": recommendations,
                "use_ncf": use_ncf,
                # Store all input variables for display in tabs
                "pat_name": pat_name, "age": age, "gender": gender,
                "height": height, "weight": weight, "activity": activity,
                "region": region, "goal": goal,
                "bp_diastolic": int(bp_diastolic),
                # Medical params stored here so XAI in Tab 2 uses the exact
                # values that were fed to the models — not widget state.
                "hba1c":      float(hba1c),
                "glucose":    float(glucose),
                "bp":         int(bp),
                "sodium":     float(sodium),
                "potassium":  float(potassium),
                "creatinine": float(creatinine),
            }
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

# ════════════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ════════════════════════════════════════════════════════════════════

if "analysis" not in st.session_state:
    # ── Welcome / landing screen ─────────────────────────────────────
    components.welcome_screen()

else:
    # ── Unpack analysis from session state ───────────────────────────
    a = st.session_state.analysis
    bmi     = a["bmi"];      bmi_cat  = a["bmi_cat"]
    bmr     = a["bmr"];      tdee     = a["tdee"]
    obesity  = a["obesity"]
    diabetes = a["diabetes"]
    kidney   = a["kidney"]
    diseases        = a["diseases"]
    recommendations = a["recommendations"]
    use_ncf         = a["use_ncf"]
    # Retrieve confidence scores
    obesity_label = a.get("obesity_label", str(obesity) if not isinstance(obesity, dict) else obesity.get("label", ""))
    obesity_confidence = a.get("obesity_confidence", None)
    diabetes_label = a.get("diabetes_label", str(diabetes) if not isinstance(diabetes, dict) else diabetes.get("label", ""))
    diabetes_confidence = a.get("diabetes_confidence", None)
    kidney_label = a.get("kidney_label", str(kidney) if not isinstance(kidney, dict) else kidney.get("label", ""))
    kidney_confidence = a.get("kidney_confidence", None)
    # RiskResult objects from backend.risk — single source of truth for Tab 2
    obesity_risk  = a.get("obesity_risk",  None)
    diabetes_risk = a.get("diabetes_risk", None)
    kidney_risk   = a.get("kidney_risk",   None)
    # Retrieve input variables
    pat_name     = a.get("pat_name", "")
    age          = a.get("age", 30)
    gender       = a.get("gender", "Male")
    height       = a.get("height", 170)
    weight       = a.get("weight", 70)
    activity     = a.get("activity", "Sedentary")
    region       = a.get("region", "Andhra Pradesh")
    goal         = a.get("goal", "Weight Loss")
    bp_diastolic = a.get("bp_diastolic", 80)
    # ── Tabs ─────────────────────────────────────────────────────────
    tabs = st.tabs([
        "👤 Patient Profile",
        "📊 Health Summary",
        "🧬 Predictions",
        "🍽️ Recommendations",
    ])

    # ══════════════════════════════════════════════════════════════════
    #  TAB 0 — Patient Profile
    # ══════════════════════════════════════════════════════════════════
    with tabs[0]:
        components.section_header("👤", "Patient Profile")

        col_a, col_b = st.columns(2, gap="large")
        left_items = [
            ("🧑", "Name",           pat_name or "—"),
            ("📅", "Age",            f"{age} years"),
            ("⚥",  "Gender",         gender),
        ]
        right_items = [
            ("📏", "Height",          f"{height} cm"),
            ("⚖️", "Weight",          f"{weight} kg"),
            ("🏃", "Activity Level",  activity),
        ]
        with col_a:
            for ico, lbl, val in left_items:
                components.profile_card(ico, lbl, val)
        with col_b:
            for ico, lbl, val in right_items:
                components.profile_card(ico, lbl, val)

        # Blood pressure summary row
        components.section_header("🩺", "Blood Pressure")
        bp_col1, bp_col2 = st.columns(2, gap="large")
        # Retrieve stored bp from analysis for display
        _stored_bp = a.get("bp", "—")
        with bp_col1:
            components.profile_card("💉", "Systolic BP",  f"{_stored_bp} mmHg")
        with bp_col2:
            components.profile_card("💉", "Diastolic BP", f"{bp_diastolic} mmHg")

        components.section_header("📍", "Region & Goal")
        col_r, col_g = st.columns(2, gap="large")
        with col_r:
            components.profile_card("🗺️", "State / UT", region)
        with col_g:
            components.profile_card("🏆", "Weight Goal", goal)

    # ══════════════════════════════════════════════════════════════════
    #  TAB 1 — Health Summary
    # ══════════════════════════════════════════════════════════════════
    with tabs[1]:
        components.section_header("📊", "Key Health Metrics")

        # Row 1 — 4 KPI cards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            components.metric_card("BMI", f"{bmi:.1f}", "⚖️", "#2563EB")
        with c2:
            cat_color = (
                "#10B981" if "normal"     in bmi_cat.lower() else
                "#F59E0B" if "overweight" in bmi_cat.lower() else
                "#EF4444"
            )
            components.metric_card("BMI Category", bmi_cat, "🏷️", cat_color)
        with c3:
            components.metric_card("Daily Calories", f"{tdee:.0f} kcal", "🔥", "#F59E0B")
        with c4:
            components.metric_card("BMR", f"{bmr:.0f} kcal", "⚡", "#6366F1")

        # Row 2 — Water & Protein
        c5, c6 = st.columns(2)
        with c5:
            components.metric_card(
                "Water Intake",
                f"{recommendations.get('water_intake', '—')} L",
                "💧", "#06B6D4",
            )
        with c6:
            components.metric_card(
                "Protein Requirement",
                f"{recommendations.get('protein_requirement', '—')} g",
                "🥩", "#10B981",
            )

        # BMI status alert
        if bmi < 18.5:
            components.status_banner("⚠️", "Underweight",
                f"Your BMI of <strong>{bmi:.1f}</strong> is below the healthy range "
                f"(18.5–24.9). Consider consulting a dietitian.", "warning")
        elif bmi < 25:
            components.status_banner("✅", "Healthy Weight",
                f"Your BMI of <strong>{bmi:.1f}</strong> is within the healthy range. "
                f"Keep it up!", "ok")
        elif bmi < 30:
            components.status_banner("⚠️", "Overweight",
                f"Your BMI of <strong>{bmi:.1f}</strong> is above the healthy range. "
                f"Lifestyle changes are recommended.", "warning")
        else:
            components.status_banner("🚨", "Obese",
                f"Your BMI of <strong>{bmi:.1f}</strong> indicates obesity. "
                f"Please consult a healthcare provider.", "danger")

        # Charts
        components.section_header("📈", "Visual Analytics")
        ch1, ch2 = st.columns(2, gap="large")

        with ch1:
            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            components.chart_bmi_gauge(bmi)
            st.markdown("</div>", unsafe_allow_html=True)

        # Build calorie breakdown from meal plan
        meal_cals = []
        for meal_name, meal_data in recommendations.get("meal_plan", {}).items():
            if isinstance(meal_data, pd.DataFrame) and "Calories (kcal)" in meal_data.columns:
                total = pd.to_numeric(meal_data["Calories (kcal)"], errors="coerce").sum()
                if total > 0:
                    meal_cals.append({"meal": meal_name.capitalize(), "calories": round(total)})
            elif isinstance(meal_data, list):
                total = sum(
                    float(item.get("Calories (kcal)", item.get("Calories", item.get("calories", 0))))
                    for item in meal_data if isinstance(item, dict)
                )
                if total > 0:
                    meal_cals.append({"meal": meal_name.capitalize(), "calories": round(total)})

        with ch2:
            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            if meal_cals:
                components.chart_calorie_breakdown(pd.DataFrame(meal_cals))
            else:
                st.info("Calorie breakdown chart will appear after meal plan is generated.")
            st.markdown("</div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════
    #  TAB 2 — Predictions
    # ══════════════════════════════════════════════════════════════════
    with tabs[2]:
        components.section_header("🧬", "Disease Risk Predictions")

        # Show if manual override is active
        if manual_override and manual_diseases:
            components.status_banner(
                "🔧", "Manual Override Active",
                f"Showing results for manually selected conditions: "
                f"<strong>{', '.join(manual_diseases)}</strong>. "
                f"AI predictions have been replaced.",
                "info",
            )

        pred_cols = st.columns(3, gap="large")

        # ── Diabetes ─────────────────────────────────────────────────
        with pred_cols[0]:
            # All UI values come from the RiskResult — no re-derivation
            _d_risk  = diabetes_risk
            _d_level = _d_risk.card_level        if _d_risk else "low"
            _d_badge = _d_risk.card_risk_text    if _d_risk else "Unknown"
            _d_prob  = _d_risk.model_probability if _d_risk else diabetes_confidence
            _d_fs    = _d_risk.final_status      if _d_risk else "Unknown"

            components.prediction_card(
                icon="🩸", name="Diabetes",
                pred_class=diabetes_label,
                risk=_d_badge,
                level=_d_level,
                final_status=_d_fs,
                model_probability=_d_prob,
            )
            if _d_risk:
                components.status_banner(
                    "🩸", _d_risk.banner_title,
                    _d_risk.banner_body,
                    _d_risk.banner_level,
                )
            else:
                components.status_banner("🩸", "Diabetes — No Data",
                    "Run the analysis to see the diabetes prediction.", "info")

        # ── Obesity ──────────────────────────────────────────────────
        with pred_cols[1]:
            _o_risk  = obesity_risk
            _o_level = _o_risk.card_level        if _o_risk else "low"
            _o_badge = _o_risk.card_risk_text    if _o_risk else "Unknown"
            _o_prob  = _o_risk.model_probability if _o_risk else obesity_confidence
            _o_fs    = _o_risk.final_status      if _o_risk else "Unknown"

            components.prediction_card(
                icon="⚖️", name="Obesity",
                pred_class=obesity_label,
                risk=_o_badge,
                level=_o_level,
                final_status=_o_fs,
                model_probability=_o_prob,
            )
            if _o_risk:
                components.status_banner(
                    "⚖️", _o_risk.banner_title,
                    _o_risk.banner_body,
                    _o_risk.banner_level,
                )
            else:
                components.status_banner("⚖️", "Obesity — No Data",
                    "Run the analysis to see the obesity prediction.", "info")

        # ── Kidney Disease ────────────────────────────────────────────
        with pred_cols[2]:
            _k_risk  = kidney_risk
            _k_level = _k_risk.card_level        if _k_risk else "low"
            _k_badge = _k_risk.card_risk_text    if _k_risk else "Unknown"
            _k_prob  = _k_risk.model_probability if _k_risk else kidney_confidence
            _k_fs    = _k_risk.final_status      if _k_risk else "Unknown"

            components.prediction_card(
                icon="🫘", name="Kidney Disease",
                pred_class=kidney_label,
                risk=_k_badge,
                level=_k_level,
                final_status=_k_fs,
                model_probability=_k_prob,
            )
            if _k_risk:
                components.status_banner(
                    "🫘", _k_risk.banner_title,
                    _k_risk.banner_body,
                    _k_risk.banner_level,
                )
            else:
                components.status_banner("🫘", "Kidney Disease — No Data",
                    "Run the analysis to see the kidney disease prediction.", "info")

        # ── Overall summary ───────────────────────────────────────────
        components.section_header("📋", "Overall Risk Summary")
        # Count conditions with genuine clinical or model+clinical signal
        _flagged = [
            r for r in [diabetes_risk, obesity_risk, kidney_risk]
            if r is not None and r.final_status in ("High Risk", "Moderate Risk")
        ]
        _model_flags = [
            r for r in [diabetes_risk, obesity_risk, kidney_risk]
            if r is not None and r.final_status == "Model Flag"
        ]
        if not _flagged and not _model_flags:
            components.status_banner("🎉", "All Clear!",
                "No significant disease risk detected. Maintain your healthy lifestyle.", "ok")
        elif _flagged:
            _flagged_names = ", ".join(
                n for n, r in [("Diabetes", diabetes_risk),
                               ("Obesity",  obesity_risk),
                               ("Kidney Disease", kidney_risk)]
                if r is not None and r.final_status in ("High Risk", "Moderate Risk")
            )
            components.status_banner("⚠️", "Elevated Risk Detected",
                f"Identified: <strong>{_flagged_names}</strong>. "
                "Personalised meal and nutrition recommendations are in the next tab.",
                "warning")
        else:
            _flag_names = ", ".join(
                n for n, r in [("Diabetes", diabetes_risk),
                               ("Obesity",  obesity_risk),
                               ("Kidney Disease", kidney_risk)]
                if r is not None and r.final_status == "Model Flag"
            )
            components.status_banner("🔵", "Screening Flags — Clinical Markers Normal",
                f"The model flagged: <strong>{_flag_names}</strong>. "
                "All measured clinical markers are within normal reference ranges. "
                "These are model screening signals, not confirmed diagnoses.",
                "info")

        # ── XAI — Explainable AI section ──────────────────────────────
        # Retrieve the exact medical params that were fed to the models,
        # stored in session_state at analysis time for consistency.
        _xai_hba1c      = a.get("hba1c",      6.5)
        _xai_glucose    = a.get("glucose",     120.0)
        _xai_bp         = a.get("bp",          120)
        _xai_sodium     = a.get("sodium",      138.0)
        _xai_potassium  = a.get("potassium",   4.5)
        _xai_creatinine = a.get("creatinine",  1.0)

        if XAI_AVAILABLE:
            components.section_header("🔬", "Explainable AI — Why These Predictions?")

            # ── One full-width tab per disease ────────────────────────
            # Using st.tabs() instead of st.columns() so each panel
            # gets the full page width — fixes the expander arrow /
            # label overlap that occurs inside narrow columns.
            xai_tabs = st.tabs([
                "🩸 Diabetes",
                "⚖️ Obesity",
                "🫘 Kidney Disease",
            ])

            with xai_tabs[0]:
                # Header row: prediction label + confidence chip
                _conf_diab = (
                    f"&nbsp;&nbsp;<span style='"
                    f"background:#EFF6FF;color:#1E40AF;"
                    f"border-radius:99px;padding:0.2rem 0.7rem;"
                    f"font-size:0.8rem;font-weight:700;'>"
                    f"Confidence: {diabetes_confidence}%</span>"
                    if diabetes_confidence is not None else ""
                )
                st.markdown(
                    f"<div class='xai-tab-header'>"
                    f"<span style='font-size:1.5rem;'>🩸</span>"
                    f"<span>Diabetes &nbsp;—&nbsp; "
                    f"<em>{diabetes_label}</em>{_conf_diab}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                _xai_diab = explain_diabetes(
                    age=age, gender=gender, bmi=bmi,
                    hba1c=_xai_hba1c, glucose=_xai_glucose,
                    label=diabetes_label,
                )
                components.xai_explanation_panel(_xai_diab, "Diabetes")

            with xai_tabs[1]:
                _conf_ob = (
                    f"&nbsp;&nbsp;<span style='"
                    f"background:#FFFBEB;color:#92400E;"
                    f"border-radius:99px;padding:0.2rem 0.7rem;"
                    f"font-size:0.8rem;font-weight:700;'>"
                    f"Confidence: {obesity_confidence}%</span>"
                    if obesity_confidence is not None else ""
                )
                st.markdown(
                    f"<div class='xai-tab-header'>"
                    f"<span style='font-size:1.5rem;'>⚖️</span>"
                    f"<span>Obesity &nbsp;—&nbsp; "
                    f"<em>{obesity_label}</em>{_conf_ob}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                _xai_ob = explain_obesity(
                    age=age, gender=gender, bmi=bmi,
                    label=obesity_label,
                )
                components.xai_explanation_panel(_xai_ob, "Obesity")

            with xai_tabs[2]:
                _conf_kid = (
                    f"&nbsp;&nbsp;<span style='"
                    f"background:#FFF1F2;color:#9F1239;"
                    f"border-radius:99px;padding:0.2rem 0.7rem;"
                    f"font-size:0.8rem;font-weight:700;'>"
                    f"Confidence: {kidney_confidence}%</span>"
                    if kidney_confidence is not None else ""
                )
                st.markdown(
                    f"<div class='xai-tab-header'>"
                    f"<span style='font-size:1.5rem;'>🫘</span>"
                    f"<span>Kidney Disease &nbsp;—&nbsp; "
                    f"<em>{kidney_label}</em>{_conf_kid}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                _xai_kid = explain_kidney(
                    age=age, gender=gender, bmi=bmi,
                    sodium=_xai_sodium, potassium=_xai_potassium,
                    bp=_xai_bp, creatinine=_xai_creatinine,
                    label=kidney_label,
                )
                components.xai_explanation_panel(_xai_kid, "Kidney Disease")
        else:
            st.info(
                "ℹ️ Explainable AI module is not available. "
                "Check that backend/xai.py is present and all models are loaded."
            )

    # ══════════════════════════════════════════════════════════════════
    #  TAB 3 — Recommendations
    # ══════════════════════════════════════════════════════════════════
    with tabs[3]:

        # ── NCF AI recommendations ────────────────────────────────────
        if use_ncf and "ncf_recommendations" in st.session_state:
            components.section_header("🤖", "AI-Powered Recommendations (NCF)")
            components.status_banner(
                "🤖", "Neural Collaborative Filtering Active",
                st.session_state.get("ncf_explanation",
                                     "Personalised recommendations from your profile."),
                "info",
            )
            ncf_df = st.session_state.ncf_recommendations
            if not ncf_df.empty:
                st.dataframe(ncf_df, use_container_width=True, hide_index=True)
            detected = st.session_state.get("ncf_diseases", [])
            if detected and detected != ["normal"]:
                st.markdown(
                    "**Detected conditions:** "
                    + ", ".join(d.replace("_", " ").title() for d in detected)
                )
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # ── Meal Plan ─────────────────────────────────────────────────
        components.section_header("🍽️", "Personalised Daily Meal Plan")
        meal_plan = recommendations.get("meal_plan", {})
        if meal_plan:
            for meal_name, meal_data in meal_plan.items():
                components.meal_tag(meal_name.capitalize())
                if isinstance(meal_data, pd.DataFrame) and not meal_data.empty:
                    st.dataframe(meal_data, use_container_width=True, hide_index=True)
                elif isinstance(meal_data, list) and meal_data:
                    st.dataframe(
                        pd.DataFrame(meal_data),
                        use_container_width=True, hide_index=True,
                    )
                else:
                    st.info(f"No {meal_name.lower()} data available.")
        else:
            st.info("Meal plan data is not available.")

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # ── Foods to Avoid ────────────────────────────────────────────
        components.section_header("🚫", "Foods to Avoid")
        avoid = recommendations.get("foods_to_avoid", [])
        if avoid:
            if isinstance(avoid[0], dict):
                avoid_df = pd.DataFrame(avoid)
                # Rename columns for cleaner display
                avoid_df.columns = [c.title() for c in avoid_df.columns]
            else:
                avoid_df = pd.DataFrame({"Food": avoid})
            st.dataframe(avoid_df, use_container_width=True, hide_index=True)
        else:
            components.status_banner("✅", "No Restrictions",
                "No specific foods to avoid based on your current health profile.", "ok")

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # ── Macronutrient chart (shown when data available) ───────────
        macro_data = (
            recommendations.get("macronutrients")
            or recommendations.get("macros")
        )
        if macro_data:
            components.section_header("🥗", "Macronutrient Distribution")
            if isinstance(macro_data, dict):
                macro_df = pd.DataFrame([
                    {"macro": k.replace("_g", "").capitalize(), "grams": v}
                    for k, v in macro_data.items()
                    if isinstance(v, (int, float))
                ])
            elif isinstance(macro_data, pd.DataFrame):
                macro_df = macro_data
            else:
                macro_df = pd.DataFrame()

            if not macro_df.empty:
                st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                components.chart_macronutrient(macro_df)
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # ── Nutrition Tips ────────────────────────────────────────────
        components.section_header("💡", "Daily Nutrition Tips")
        tips = recommendations.get("nutrition_tips", [])
        if tips:
            components.tip_list(tips)
        else:
            st.info("No nutrition tips available for your current profile.")

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # ── Export Report ─────────────────────────────────────────────
        components.section_header("📥", "Export Your Health Report")
        report_data = st.session_state.get("report_data")
        if report_data:
            d1, d2, d3 = st.columns(3)
            with d1:
                components.download_button(
                    report_data, filename="health_report", format="json"
                )
            with d2:
                try:
                    components.download_button(
                        report_data, filename="health_report", format="pdf"
                    )
                except Exception as e:
                    st.error(f"PDF download error: {e}")
            with d3:
                st.markdown(
                    "<div style='padding:0.55rem 0;color:var(--c-muted);font-size:0.85rem;'>"
                    "💾 Report auto-saved to local database.</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("Run an analysis first to enable report export.")
