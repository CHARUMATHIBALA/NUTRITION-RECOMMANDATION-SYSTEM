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
except ImportError:
    NCF_AVAILABLE = False

# ── Backend helpers ──────────────────────────────────────────────────
from backend.auth import authenticate, logout
from backend.database import save_analysis, get_history  # wired below

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
        "Patient Name", value=st.session_state.get("pat_name", "")
    )
    age = st.number_input(
        "Age", min_value=1, max_value=120,
        value=st.session_state.get("age", 30)
    )
    gender = st.selectbox(
        "Gender", ["Male", "Female"],
        index=0 if st.session_state.get("gender", "Male") == "Male" else 1,
    )
    height = st.number_input(
        "Height (cm)", min_value=100, max_value=250,
        value=st.session_state.get("height", 170)
    )
    weight = st.number_input(
        "Weight (kg)", min_value=30, max_value=300,
        value=st.session_state.get("weight", 70)
    )
    activity = st.selectbox(
        "Activity Level",
        ["Sedentary", "Light", "Moderate", "Active", "Very Active"],
        index=["Sedentary", "Light", "Moderate", "Active", "Very Active"].index(
            st.session_state.get("activity", "Sedentary")
        ),
    )

    # ── Medical Parameters ───────────────────────────────────────────
    st.subheader("Medical Parameters")
    hba1c = st.number_input(
        "HbA1c (%)", min_value=3.0, max_value=15.0,
        value=float(st.session_state.get("hba1c", 6.5)), step=0.1
    )
    glucose = st.number_input(
        "Blood Glucose (mg/dL)", min_value=50, max_value=500,
        value=int(st.session_state.get("glucose", 120))
    )
    bp = st.number_input(
        "Systolic BP (mmHg)", min_value=80, max_value=200,
        value=int(st.session_state.get("bp", 120))
    )
    creatinine = st.number_input(
        "Serum Creatinine", min_value=0.1, max_value=5.0,
        value=float(st.session_state.get("creatinine", 1.0)), step=0.1
    )
    sodium = st.number_input(
        "Sodium (mEq/L)", min_value=80.0, max_value=200.0,
        value=float(st.session_state.get("sodium", 138.0)), step=0.5
    )
    potassium = st.number_input(
        "Potassium (mEq/L)", min_value=2.0, max_value=7.0,
        value=float(st.session_state.get("potassium", 4.5)), step=0.1
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
#  ANALYSIS  — triggered by "Analyse Health"
# ════════════════════════════════════════════════════════════════════
if analyze:
    try:
        # ── Core calculations ────────────────────────────────────────────
        bmi     = calculate_bmi(weight, height)
        bmi_cat = bmi_category(bmi)
        bmr     = calculate_bmr(age, gender, weight, height)
        tdee    = calculate_tdee(bmr, ACTIVITY_FACTOR[activity])

        # ── Disease predictions ──────────────────────────────────────────
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
            if str(diabetes_label).lower() == "diabetes":
                diseases.append("Diabetes")
            if str(kidney_label).lower() == "kidney disease":
                diseases.append("Kidney Disease")
            if str(obesity_label).lower() in [
                "obese class i", "obese class ii", "obese class iii", "overweight"
            ]:
                diseases.append("Obesity")
            if not diseases:
                diseases = ["Normal"]

        # ── NCF recommendations (optional) ──────────────────────────────
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

        # ── Rule-based recommendations ───────────────────────────────────
        recommendations = generate_comprehensive_recommendations(
            diseases=diseases, age=age, gender=gender,
            height=height, weight=weight, bmi=bmi,
            activity_level=activity, daily_calories=tdee,
            hba1c=hba1c, glucose=glucose, bp=bp,
            sodium=sodium, potassium=potassium, creatinine=creatinine,
        )

        # ── Build report_data (used for PDF/JSON export) ─────────────────
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

        # ── Persist to SQLite (wired!) ───────────────────────────────────
        try:
            save_analysis(
                user_id=None,         # anonymous until user table is populated
                patient_name=pat_name or username or "unknown",
                data=report_data,
            )
        except Exception:
            pass  # database errors must never crash the UI

        # ── Store everything in session state ────────────────────────────
        st.session_state.report_data = report_data
        st.session_state.analysis = {
            "bmi": bmi, "bmi_cat": bmi_cat, "bmr": bmr, "tdee": tdee,
            "obesity": obesity, "diabetes": diabetes, "kidney": kidney,
            "obesity_label": obesity_label, "obesity_confidence": obesity_confidence,
            "diabetes_label": diabetes_label, "diabetes_confidence": diabetes_confidence,
            "kidney_label": kidney_label, "kidney_confidence": kidney_confidence,
            "diseases": diseases, "recommendations": recommendations,
            "use_ncf": use_ncf,
            # Store all input variables for display in tabs
            "pat_name": pat_name, "age": age, "gender": gender,
            "height": height, "weight": weight, "activity": activity,
            "region": region, "goal": goal,
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
    # Retrieve input variables
    pat_name = a.get("pat_name", "")
    age      = a.get("age", 30)
    gender   = a.get("gender", "Male")
    height   = a.get("height", 170)
    weight   = a.get("weight", 70)
    activity = a.get("activity", "Sedentary")
    region   = a.get("region", "Andhra Pradesh")
    goal     = a.get("goal", "Weight Loss")

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
            is_diabetic = (
                "diabetes" in str(diabetes_label).lower()
            )
            components.prediction_card(
                icon="🩸", name="Diabetes",
                pred_class=diabetes_label,
                risk="High" if is_diabetic else "Low",
                level="high" if is_diabetic else "low",
                confidence=diabetes_confidence,
            )
            if is_diabetic:
                components.status_banner("🩸", "Diabetes Detected",
                    "HbA1c and blood glucose levels suggest diabetes. "
                    "Please consult an endocrinologist.", "danger")
            else:
                components.status_banner("✅", "No Diabetes Detected",
                    "Your blood sugar markers are within acceptable limits.", "ok")

        # ── Obesity ──────────────────────────────────────────────────
        with pred_cols[1]:
            is_obese = any(
                x in str(obesity_label).lower()
                for x in ["obese", "overweight"]
            )
            components.prediction_card(
                icon="⚖️", name="Obesity",
                pred_class=obesity_label,
                risk="High" if is_obese else "Normal",
                level="medium" if is_obese else "low",
                confidence=obesity_confidence,
            )
            if is_obese:
                components.status_banner("⚖️", "Weight Concern",
                    "Your BMI and body metrics indicate elevated weight. "
                    "A structured diet and exercise plan is advised.", "warning")
            else:
                components.status_banner("✅", "Healthy Weight Status",
                    "Your weight is within the normal range.", "ok")

        # ── Kidney Disease ────────────────────────────────────────────
        with pred_cols[2]:
            is_kidney = "kidney disease" in str(kidney_label).lower()
            components.prediction_card(
                icon="🫘", name="Kidney Disease",
                pred_class=kidney_label,
                risk="High" if is_kidney else "Low",
                level="high" if is_kidney else "low",
                confidence=kidney_confidence,
            )
            if is_kidney:
                components.status_banner("🫘", "Kidney Disease Detected",
                    "Creatinine, sodium, or potassium levels suggest kidney dysfunction. "
                    "Seek nephrology consultation.", "danger")
            else:
                components.status_banner("✅", "Kidney Health Normal",
                    "Your renal markers are within healthy limits.", "ok")

        # ── Overall summary ───────────────────────────────────────────
        components.section_header("📋", "Overall Risk Summary")
        if diseases == ["Normal"]:
            components.status_banner("🎉", "All Clear!",
                "No significant disease risk detected. Maintain your healthy lifestyle.", "ok")
        else:
            components.status_banner("⚠️", "Conditions Detected",
                f"Identified: <strong>{', '.join(diseases)}</strong>. "
                f"Personalised meal and nutrition recommendations are in the next tab.",
                "warning")

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
