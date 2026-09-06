"""app.py — Smart Health Dashboard main entry point.

Flow
----
1. Page config + CSS injection
2. Theme initialisation (light / dark)
3. Authentication gate  — stops here if not logged in
4. Sidebar  — branding, user info, dark mode, reset, logout  (NO patient inputs)
5. Main page — st.form with all patient / medical inputs + Analyse button
6. Analysis — triggered by form submission (same logic as before)
7. Main area — 4 tabs: Patient Profile · Health Summary · Predictions · Recommendations
"""

import streamlit as st
import pandas as pd

# Ensure analyze variable is defined
analyze = False


# ── Local utilities ──────────────────────────────────────────────────
from utils import calculate_bmi, bmi_category, calculate_bmr, calculate_tdee
from predict import predict_diabetes, predict_kidney, predict_obesity
from meal_planner import generate_comprehensive_recommendations
import components
from recommendation import IntelligentNutritionRecommender

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
    initial_sidebar_state="collapsed",
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

if auth_status is not True:
    # Show styled login landing. The login widget itself is rendered
    # by streamlit-authenticator inside authenticate() above.
    st.markdown(
        """
        <script>document.documentElement.classList.add('login-view');</script>
        <style>
          body, .stApp, section.main {
            background: linear-gradient(135deg, #EFF6FF 0%, #F0F9FF 55%, #F8FAFC 100%) !important;
            min-height: 100vh !important;
          }
        </style>
        <div class='login-wrapper'>
            <span class='login-logo'>🩺</span>
            <div class='login-accent-bar'></div>
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
#  SIDEBAR  — compact controls (dark mode, reset, logout)
# ════════════════════════════════════════════════════════════════════
_display_name = name or username or "User"

with st.sidebar:
    components.sidebar_user_chip(_display_name)
    st.markdown("<hr style='margin:0.4rem 0;'/>", unsafe_allow_html=True)

    st.subheader("Appearance")
    dark_mode = st.checkbox(
        "🌙 Dark Mode",
        value=st.session_state.get("theme", "light") == "dark",
        key="sidebar_dark_mode",
    )
    st.session_state.theme = "dark" if dark_mode else "light"

    st.markdown("<hr style='margin:0.4rem 0;'/>", unsafe_allow_html=True)

    if st.button("♻️ Reset Results", use_container_width=True, key="sidebar_reset"):
        keep = {"theme", "authenticator", "authentication_status", "name", "username"}
        for key in list(st.session_state.keys()):
            if key not in keep:
                del st.session_state[key]
        st.rerun()

    if st.button("🚪 Logout", use_container_width=True, key="sidebar_logout"):
        logout()
        st.rerun()

# ════════════════════════════════════════════════════════════════════
#  Re-apply theme (sidebar may have just changed it)
# ════════════════════════════════════════════════════════════════════
st.markdown(
    f"<script>document.documentElement.setAttribute('data-theme',"
    f"'{st.session_state.theme}');</script>",
    unsafe_allow_html=True,
)

# ════════════════════════════════════════════════════════════════════
#  HERO BANNER  (single top header — profile icon embedded inside)
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

_ACTIVITY_OPTIONS = ["Sedentary", "Light", "Moderate", "Active", "Very Active"]

_INDIAN_STATES = [
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

# ════════════════════════════════════════════════════════════════════
#  INPUT VALIDATION  (unchanged)
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
            try:
                if dbp >= int(bp):
                    errors.append(
                        f"❗ **Diastolic BP** ({bp_diastolic} mmHg) must be lower than "
                        f"**Systolic BP** ({bp} mmHg)."
                    )
            except (TypeError, ValueError):
                pass
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


def _get_bmi_range_info(bmi):
    """Get BMI healthy range information."""
    if bmi < 18.5:
        return {
            'range': '18.5 – 24.9',
            'status': 'Below reference range',
            'status_class': 'status-below'
        }
    elif bmi < 25:
        return {
            'range': '18.5 – 24.9',
            'status': 'Within reference range',
            'status_class': 'status-normal'
        }
    elif bmi < 30:
        return {
            'range': '18.5 – 24.9',
            'status': 'Above reference range',
            'status_class': 'status-above'
        }
    else:
        return {
            'range': '18.5 – 24.9',
            'status': 'Above reference range',
            'status_class': 'status-above'
        }


def _generate_meal_explanation(food_items: list, diseases: list) -> str:
    """Generate explanation for a balanced meal."""
    if not food_items:
        return "Nutritionally balanced meal."
    
    parts = []
    
    # Analyze nutritional profile
    total_protein = sum(
        item.get("Protein (g)", item.get("Protein", item.get("protein", 0)))
        for item in food_items if isinstance(item, dict)
    )
    total_fiber = sum(
        item.get("Fibre (g)", item.get("Fibre", item.get("fiber", 0)))
        for item in food_items if isinstance(item, dict)
    )
    
    if total_fiber >= 5:
        parts.append("high-fibre")
    if total_protein >= 10:
        parts.append("protein-rich")
    
    has_diabetes = any('diabetes' in d.lower() for d in diseases)
    has_obesity = any('obesity' in d.lower() or 'overweight' in d.lower() for d in diseases)
    
    if has_diabetes:
        parts.append("controlled-carbohydrate")
    if has_obesity:
        parts.append("calorie-conscious")
    
    if not parts:
        parts.append("balanced")
    
    explanation = f"{' '.join(parts)} meal suitable for your health profile."
    
    return explanation


def _display_balanced_meal(food_items: list, meal_name: str, diseases: list, daily_calories: float):
    """Display a balanced meal with main + protein + vegetable format."""
    if not food_items:
        st.info(f"No {meal_name.lower()} data available.")
        return
    
    # Calculate total calories
    total_calories = sum(
        item.get("Calories (kcal)", item.get("Calories", item.get("calories", 0)))
        for item in food_items if isinstance(item, dict)
    )
    
    # Generate meal explanation
    explanation = _generate_meal_explanation(food_items, diseases)
    
    # Display meal as a combined card
    st.markdown(
        f"""
        <div style='background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 1rem; margin-bottom: 1rem;'>
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;'>
                <span style='font-size: 1.1rem; font-weight: 600; color: #0F172A;'>
                    {' + '.join([item.get('Dish Name', 'Unknown') for item in food_items[:3] if isinstance(item, dict)])}
                </span>
                <span style='color: #2563EB; font-weight: 500; background: #EFF6FF; padding: 0.3rem 0.6rem; border-radius: 6px;'>
                    ~{total_calories:.0f} kcal
                </span>
            </div>
            <div style='color: #64748B; font-size: 0.9rem; margin-bottom: 0.75rem;'>
                <strong>Why:</strong> {explanation}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Display individual food items with nutrition
    for idx, food_item in enumerate(food_items):
        if isinstance(food_item, dict):
            food_name = food_item.get("Dish Name", "Unknown Food")
            calories = food_item.get("Calories (kcal)", food_item.get("Calories", food_item.get("calories", 0)))
            protein = food_item.get("Protein (g)", food_item.get("Protein", food_item.get("protein", None)))
            carbs = food_item.get("Carbohydrates (g)", food_item.get("Carbohydrates", food_item.get("carbs", None)))
            fat = food_item.get("Fats (g)", food_item.get("Fats", food_item.get("fat", None)))
            fiber = food_item.get("Fibre (g)", food_item.get("Fibre", food_item.get("fiber", None)))
            
            # Display food item card
            components.food_item_card(
                food_name=food_name,
                calories=calories,
                protein=protein,
                carbs=carbs,
                fat=fat,
                fiber=fiber,
                meal_type=meal_name,
                food_index=idx
            )
            
            st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)


def _get_parameter_range_info(param_name, value):
    """Get healthy range information for medical parameters."""
    ranges = {
        'hba1c': {
            'normal_range': 'below 5.7%',
            'prediabetes_range': '5.7% – 6.4%',
            'diabetes_range': '6.5% or higher',
            'check_normal': lambda v: v < 5.7,
            'check_prediabetes': lambda v: 5.7 <= v < 6.5,
            'check_diabetes': lambda v: v >= 6.5
        },
        'glucose': {
            'normal_range': '70 – 99 mg/dL',
            'prediabetes_range': '100 – 125 mg/dL',
            'diabetes_range': '126 mg/dL or higher',
            'check_normal': lambda v: 70 <= v <= 99,
            'check_prediabetes': lambda v: 100 <= v <= 125,
            'check_diabetes': lambda v: v >= 126
        },
        'bp_systolic': {
            'normal_range': 'below 120 mmHg',
            'elevated_range': '120 – 129 mmHg',
            'high_range': '130 mmHg or higher',
            'check_normal': lambda v: v < 120,
            'check_elevated': lambda v: 120 <= v < 130,
            'check_high': lambda v: v >= 130
        },
        'sodium': {
            'normal_range': '135 – 145 mEq/L',
            'check_normal': lambda v: 135 <= v <= 145,
            'check_low': lambda v: v < 135,
            'check_high': lambda v: v > 145
        },
        'potassium': {
            'normal_range': '3.5 – 5.0 mEq/L',
            'check_normal': lambda v: 3.5 <= v <= 5.0,
            'check_low': lambda v: v < 3.5,
            'check_high': lambda v: v > 5.0
        },
        'creatinine': {
            'normal_range': '0.7 – 1.2 mg/dL (varies by gender/age)',
            'check_normal': lambda v: 0.7 <= v <= 1.2,
            'check_low': lambda v: v < 0.7,
            'check_high': lambda v: v > 1.2
        }
    }
    
    if param_name not in ranges:
        return None
    
    param_info = ranges[param_name]
    
    # Determine status based on parameter type
    if param_name == 'hba1c':
        if param_info['check_normal'](value):
            return {
                'range': param_info['normal_range'],
                'status': 'Within reference range',
                'status_class': 'status-normal'
            }
        elif param_info['check_prediabetes'](value):
            return {
                'range': param_info['prediabetes_range'],
                'status': 'Above reference range',
                'status_class': 'status-above'
            }
        else:
            return {
                'range': param_info['diabetes_range'],
                'status': 'Above reference range',
                'status_class': 'status-above'
            }
    
    elif param_name == 'glucose':
        if param_info['check_normal'](value):
            return {
                'range': param_info['normal_range'],
                'status': 'Within reference range',
                'status_class': 'status-normal'
            }
        elif param_info['check_prediabetes'](value):
            return {
                'range': param_info['prediabetes_range'],
                'status': 'Above reference range',
                'status_class': 'status-above'
            }
        else:
            return {
                'range': param_info['diabetes_range'],
                'status': 'Above reference range',
                'status_class': 'status-above'
            }
    
    elif param_name == 'bp_systolic':
        if param_info['check_normal'](value):
            return {
                'range': param_info['normal_range'],
                'status': 'Within reference range',
                'status_class': 'status-normal'
            }
        elif param_info['check_elevated'](value):
            return {
                'range': param_info['elevated_range'],
                'status': 'Above reference range',
                'status_class': 'status-above'
            }
        else:
            return {
                'range': param_info['high_range'],
                'status': 'Above reference range',
                'status_class': 'status-above'
            }
    
    else:
        # For sodium, potassium, creatinine
        if param_info['check_normal'](value):
            return {
                'range': param_info['normal_range'],
                'status': 'Within reference range',
                'status_class': 'status-normal'
            }
        elif param_info['check_low'](value):
            return {
                'range': param_info['normal_range'],
                'status': 'Below reference range',
                'status_class': 'status-below'
            }
        else:
            return {
                'range': param_info['normal_range'],
                'status': 'Above reference range',
                'status_class': 'status-above'
            }


# ════════════════════════════════════════════════════════════════════
#  MAIN-PAGE FORM  — 4-step wizard health assessment form.
#  All variable names match what the existing analysis block expects.
# ════════════════════════════════════════════════════════════════════
_has_results = "analysis" in st.session_state

# Initialize wizard step in session state
if "wizard_step" not in st.session_state:
    st.session_state.wizard_step = 1

# Reset wizard step when editing existing results
if _has_results and st.session_state.wizard_step != 1:
    st.session_state.wizard_step = 1

# Initialize edit-form visibility toggle
if "show_edit_form" not in st.session_state:
    st.session_state.show_edit_form = False

# When results exist, show a slim inline toggle instead of an expander box
if _has_results:
    st.markdown("<div class='edit-form-toggle-row'>", unsafe_allow_html=True)
    if st.button(
        "✏️ Edit Patient Details & Re-run Analysis" if not st.session_state.show_edit_form
        else "✖ Close Edit Form",
        key="toggle_edit_form",
        use_container_width=False,
    ):
        st.session_state.show_edit_form = not st.session_state.show_edit_form
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

_show_form = (not _has_results) or st.session_state.get("show_edit_form", False)

if _show_form:
    _form_ctx = st.container()
else:
    _form_ctx = None

if _show_form:

    # ── Page header (shown only when no results yet) ─────────────────
    if not _has_results:
        st.markdown(
            """
            <div class='assessment-header animate-in'>
                <h1>🩺 Patient Health Assessment</h1>
                <div class='accent-line'></div>
                <p class='sub'>
                    Enter your health information below to receive personalised
                    disease risk predictions, nutrition recommendations, and
                    a tailored meal plan.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Wizard Progress Indicator ─────────────────────────────────────
    def render_wizard_progress(current_step):
        """Render visual progress indicator for wizard steps."""
        steps = [
            ("1", "Personal Information", "👤"),
            ("2", "Lifestyle", "🏃"),
            ("3", "Medical Parameters", "🩺"),
            ("4", "Review & Analyse", "✅"),
        ]

        # Step counter display
        step_counter_html = f"""
            <div class='wizard-step-counter'>
                <span class='step-counter-text'>Step {current_step} of 4</span>
            </div>
        """
        st.markdown(step_counter_html, unsafe_allow_html=True)

        progress_html = "<div class='wizard-progress'>"
        for i, (num, label, icon) in enumerate(steps):
            step_num = i + 1
            is_completed = step_num < current_step
            is_current = step_num == current_step

            status_class = ""
            if is_completed:
                status_class = "completed"
            elif is_current:
                status_class = "current"
            else:
                status_class = "pending"

            # Show checkmark for completed steps, number for others
            icon_display = "✓" if is_completed else num

            progress_html += f"""
                <div class='wizard-step {status_class}' aria-label='{label} - Step {step_num} of 4, {"Completed" if is_completed else ("Current" if is_current else "Upcoming")}'>
                    <div class='wizard-step-icon'>{icon_display}</div>
                    <div class='wizard-step-label'>{label}</div>
                </div>
            """
            if i < len(steps) - 1:
                progress_html += "<div class='wizard-step-connector'></div>"
        progress_html += "</div>"

        st.markdown(progress_html, unsafe_allow_html=True)

    render_wizard_progress(st.session_state.wizard_step)

    # ── Wizard Step Navigation Buttons ───────────────────────────────────
    def render_wizard_navigation(current_step):
        """Render Back/Next buttons based on current step."""
        col_back, col_next = st.columns([1, 1])
        
        with col_back:
            if current_step > 1:
                if st.button("← Back", use_container_width=True, key="wizard_back"):
                    st.session_state.wizard_step = current_step - 1
                    st.rerun()
        
        with col_next:
            if current_step < 4:
                if st.button(f"Step {current_step + 1} →", use_container_width=True, key="wizard_next", type="primary"):
                    st.session_state.wizard_step = current_step + 1
                    st.rerun()

    # ── Step 1: Personal Information ─────────────────────────────────
    if st.session_state.wizard_step == 1:
        st.markdown(
            "<div class='form-section-card'>"
            "<div class='form-section-title'>"
            "  <div class='fst-icon'>👤</div>"
            "  <h3>Personal Information</h3>"
            "  <span class='fst-badge'>Step 1 of 4</span>"
            "</div>",
            unsafe_allow_html=True,
        )

        # Row 1 — Name spans full width for comfortable entry
        pat_name = st.text_input(
            "Patient Name",
            value=st.session_state.get("pat_name", ""),
            max_chars=100,
            placeholder="Enter the patient's full name",
            help="Required — minimum 2 characters.",
            key="step1_pat_name",
        )

        # Row 2 — Age · Gender · Height · Weight  (4 equal columns)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            age = st.number_input(
                "Age (years)",
                min_value=1, max_value=120,
                value=int(st.session_state.get("age", 30)),
                step=1,
                help="1 – 120 years.",
                key="step1_age",
            )
        with c2:
            gender = st.selectbox(
                "Gender",
                ["Male", "Female"],
                index=0 if st.session_state.get("gender", "Male") == "Male" else 1,
                help="Used for BMR & model calculations.",
                key="step1_gender",
            )
        with c3:
            height = st.number_input(
                "Height (cm)",
                min_value=100, max_value=250,
                value=int(st.session_state.get("height", 170)),
                step=1,
                help="100 – 250 cm.",
                key="step1_height",
            )
        with c4:
            weight = st.number_input(
                "Weight (kg)",
                min_value=30, max_value=300,
                value=int(st.session_state.get("weight", 70)),
                step=1,
                help="30 – 300 kg.",
                key="step1_weight",
            )

        st.markdown("</div>", unsafe_allow_html=True)   # close form-section-card

        # Save step 1 values to session state
        st.session_state.pat_name = str(pat_name).strip() if pat_name else ""
        st.session_state.age = int(age)
        st.session_state.gender = gender
        st.session_state.height = int(height)
        st.session_state.weight = int(weight)

        # Real-time BMI and BMR calculation display
        if height > 0 and weight > 0:
            try:
                # Calculate BMI
                calculated_bmi = calculate_bmi(weight, height)
                bmi_cat = bmi_category(calculated_bmi)
                
                # Calculate BMR if age and gender are available
                calculated_bmr = None
                if age > 0 and gender:
                    calculated_bmr = calculate_bmr(age, gender, weight, height)
                
                # Display metrics in a clean card
                st.markdown(
                    """
                    <div class='form-section-card'>
                    <div class='form-section-title'>
                      <div class='fst-icon'>📊</div>
                      <h3>Health Metrics</h3>
                      <span class='fst-badge'>Auto-calculated</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                
                # Display BMI and BMR in two columns
                metrics_col1, metrics_col2 = st.columns(2)
                
                with metrics_col1:
                    # Determine BMI category class for color coding
                    bmi_class = f"bmi-{bmi_cat.lower().replace(' ', '-')}"
                    
                    # Get BMI healthy range information
                    bmi_range_info = _get_bmi_range_info(calculated_bmi)
                    
                    st.markdown(
                        f"""
                        <div class='metric-card {bmi_class}'>
                            <div class='metric-icon'>⚖️</div>
                            <div class='metric-label'>BMI</div>
                            <div class='metric-value'>{calculated_bmi:.2f}</div>
                            <div class='metric-category'>{bmi_cat}</div>
                            <div class='metric-range'>
                                <span class='range-label'>Reference: {bmi_range_info['range']}</span>
                            </div>
                            <div class='metric-status {bmi_range_info['status_class']}'>
                                {bmi_range_info['status']}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                
                with metrics_col2:
                    if calculated_bmr:
                        st.markdown(
                            f"""
                            <div class='metric-card'>
                                <div class='metric-icon'>🔥</div>
                                <div class='metric-label'>BMR</div>
                                <div class='metric-value'>{calculated_bmr:.0f}</div>
                                <div class='metric-unit'>kcal/day</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            """
                            <div class='metric-card' style='opacity: 0.6;'>
                                <div class='metric-icon'>🔥</div>
                                <div class='metric-label'>BMR</div>
                                <div class='metric-value'>—</div>
                                <div class='metric-unit'>Enter age & gender</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                
                st.markdown("</div>", unsafe_allow_html=True)   # close form-section-card
                
            except Exception:
                # Handle calculation errors gracefully
                pass
        else:
            # Show placeholder when values are incomplete
            st.markdown(
                """
                <div class='form-section-card' style='opacity: 0.7;'>
                    <div class='form-section-title'>
                      <div class='fst-icon'>📊</div>
                      <h3>Health Metrics</h3>
                      <span class='fst-badge'>Enter height & weight</span>
                    </div>
                    <p style='color: var(--c-muted); font-size: 0.9rem; margin: 0.5rem 0;'>
                        Enter your height and weight to automatically calculate BMI and BMR.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Validate step 1 before allowing next
        step1_errors = _validate_inputs(
            pat_name=pat_name, age=age, height=height, weight=weight,
            hba1c=5.5, glucose=90, bp=120, bp_diastolic=80,
            creatinine=1.0, sodium=138.0, potassium=4.5,
        )
        step1_errors = [e for e in step1_errors if "Patient Name" in e or "Age" in e or "Height" in e or "Weight" in e]

        if step1_errors:
            st.error("### ⚠️ Please fix the following before proceeding:")
            for err in step1_errors:
                st.markdown(err)

        render_wizard_navigation(1)

    # ── Step 2: Lifestyle ───────────────────────────────────────────────
    elif st.session_state.wizard_step == 2:
        st.markdown(
            "<div class='form-section-card'>"
            "<div class='form-section-title'>"
            "  <div class='fst-icon'>🏃</div>"
            "  <h3>Lifestyle</h3>"
            "  <span class='fst-badge'>Step 2 of 4</span>"
            "</div>",
            unsafe_allow_html=True,
        )

        activity = st.selectbox(
            "Activity Level",
            _ACTIVITY_OPTIONS,
            index=_ACTIVITY_OPTIONS.index(
                st.session_state.get("activity", "Sedentary")
            ),
            help="Your typical daily physical activity level.",
            key="step2_activity",
        )

        st.markdown("</div>", unsafe_allow_html=True)   # close form-section-card

        # Save step 2 values to session state
        st.session_state.activity = activity

        render_wizard_navigation(2)

    # ── Step 3: Medical Parameters ─────────────────────────────────────
    elif st.session_state.wizard_step == 3:
        st.markdown(
            "<div class='form-section-card'>"
            "<div class='form-section-title'>"
            "  <div class='fst-icon'>🩺</div>"
            "  <h3>Medical Parameters</h3>"
            "  <span class='fst-badge'>Step 3 of 4</span>"
            "</div>",
            unsafe_allow_html=True,
        )

        # Row 1 — HbA1c · Glucose · Systolic BP · Diastolic BP
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            hba1c = st.number_input(
                "HbA1c (%)",
                min_value=3.0, max_value=15.0,
                value=float(st.session_state.get("hba1c", 5.5)),
                step=0.1, format="%.1f",
                help="Normal < 5.7 %  |  Pre-diabetes 5.7–6.4 %  |  Diabetes ≥ 6.5 %",
                key="step3_hba1c",
            )
        with m2:
            glucose = st.number_input(
                "Blood Glucose (mg/dL)",
                min_value=50, max_value=500,
                value=int(st.session_state.get("glucose", 90)),
                step=1,
                help="Fasting: Normal 70–99  |  Pre-diabetes 100–125  |  Diabetes ≥ 126",
                key="step3_glucose",
            )
        with m3:
            bp = st.number_input(
                "Systolic BP (mmHg)",
                min_value=80, max_value=200,
                value=int(st.session_state.get("bp", 120)),
                step=1,
                help="Upper blood pressure value. Normal < 120 mmHg.",
                key="step3_bp",
            )
        with m4:
            bp_diastolic = st.number_input(
                "Diastolic BP (mmHg)",
                min_value=40, max_value=140,
                value=int(st.session_state.get("bp_diastolic", 80)),
                step=1,
                help="Lower blood pressure value. Normal < 80 mmHg.",
                key="step3_bp_diastolic",
            )

        # Row 2 — Creatinine · Sodium · Potassium  (3 columns, balanced)
        n1, n2, n3, _n4 = st.columns(4)
        with n1:
            creatinine = st.number_input(
                "Serum Creatinine (mg/dL)",
                min_value=0.1, max_value=15.0,
                value=float(st.session_state.get("creatinine", 1.0)),
                step=0.1, format="%.1f",
                help="Kidney marker. Normal 0.7–1.2 (M)  |  0.5–1.0 (F) mg/dL.",
                key="step3_creatinine",
            )
        with n2:
            sodium = st.number_input(
                "Sodium (mEq/L)",
                min_value=115.0, max_value=170.0,
                value=float(st.session_state.get("sodium", 138.0)),
                step=0.5, format="%.1f",
                help="Normal 135–145 mEq/L.",
                key="step3_sodium",
            )
        with n3:
            potassium = st.number_input(
                "Potassium (mEq/L)",
                min_value=2.0, max_value=7.0,
                value=float(st.session_state.get("potassium", 4.5)),
                step=0.1, format="%.1f",
                help="Normal 3.5–5.0 mEq/L.",
                key="step3_potassium",
            )
        with _n4:
            st.empty()  # balanced spacer

        st.markdown("</div>", unsafe_allow_html=True)   # close form-section-card

        # Save step 3 values to session state
        st.session_state.hba1c = float(hba1c)
        st.session_state.glucose = int(glucose)
        st.session_state.bp = int(bp)
        st.session_state.bp_diastolic = int(bp_diastolic)
        st.session_state.creatinine = float(creatinine)
        st.session_state.sodium = float(sodium)
        st.session_state.potassium = float(potassium)

        # Real-time healthy range comparison for medical parameters
        st.markdown(
            """
            <div class='form-section-card'>
            <div class='form-section-title'>
              <div class='fst-icon'>📊</div>
              <h3>Parameter Reference Ranges</h3>
              <span class='fst-badge'>Real-time comparison</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Display reference ranges for key parameters
        param_grid_col1, param_grid_col2 = st.columns(2)
        
        with param_grid_col1:
            # HbA1c comparison
            hba1c_range = _get_parameter_range_info('hba1c', hba1c)
            if hba1c_range:
                st.markdown(
                    f"""
                    <div class='param-range-card'>
                        <div class='param-name'>HbA1c</div>
                        <div class='param-value'>{hba1c:.1f}%</div>
                        <div class='param-reference'>Reference: {hba1c_range['range']}</div>
                        <div class='param-status {hba1c_range['status_class']}'>{hba1c_range['status']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            
            # Blood Glucose comparison
            glucose_range = _get_parameter_range_info('glucose', glucose)
            if glucose_range:
                st.markdown(
                    f"""
                    <div class='param-range-card'>
                        <div class='param-name'>Blood Glucose</div>
                        <div class='param-value'>{glucose} mg/dL</div>
                        <div class='param-reference'>Reference: {glucose_range['range']}</div>
                        <div class='param-status {glucose_range['status_class']}'>{glucose_range['status']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            
            # Systolic BP comparison
            bp_range = _get_parameter_range_info('bp_systolic', bp)
            if bp_range:
                st.markdown(
                    f"""
                    <div class='param-range-card'>
                        <div class='param-name'>Systolic BP</div>
                        <div class='param-value'>{bp} mmHg</div>
                        <div class='param-reference'>Reference: {bp_range['range']}</div>
                        <div class='param-status {bp_range['status_class']}'>{bp_range['status']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        
        with param_grid_col2:
            # Sodium comparison
            sodium_range = _get_parameter_range_info('sodium', sodium)
            if sodium_range:
                st.markdown(
                    f"""
                    <div class='param-range-card'>
                        <div class='param-name'>Sodium</div>
                        <div class='param-value'>{sodium:.1f} mEq/L</div>
                        <div class='param-reference'>Reference: {sodium_range['range']}</div>
                        <div class='param-status {sodium_range['status_class']}'>{sodium_range['status']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            
            # Potassium comparison
            potassium_range = _get_parameter_range_info('potassium', potassium)
            if potassium_range:
                st.markdown(
                    f"""
                    <div class='param-range-card'>
                        <div class='param-name'>Potassium</div>
                        <div class='param-value'>{potassium:.1f} mEq/L</div>
                        <div class='param-reference'>Reference: {potassium_range['range']}</div>
                        <div class='param-status {potassium_range['status_class']}'>{potassium_range['status']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            
            # Creatinine comparison
            creatinine_range = _get_parameter_range_info('creatinine', creatinine)
            if creatinine_range:
                st.markdown(
                    f"""
                    <div class='param-range-card'>
                        <div class='param-name'>Serum Creatinine</div>
                        <div class='param-value'>{creatinine:.1f} mg/dL</div>
                        <div class='param-reference'>Reference: {creatinine_range['range']}</div>
                        <div class='param-status {creatinine_range['status_class']}'>{creatinine_range['status']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Medical disclaimer
        st.markdown(
            """
            <div class='medical-disclaimer'>
                <span class='disclaimer-icon'>ℹ️</span>
                <span class='disclaimer-text'>Reference ranges are general guidelines and may vary based on individual circumstances. Consult a qualified healthcare professional for medical interpretation.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)   # close form-section-card

        # Validate step 3 before allowing next
        step3_errors = _validate_inputs(
            pat_name="Test", age=30, height=170, weight=70,
            hba1c=hba1c, glucose=glucose, bp=bp, bp_diastolic=bp_diastolic,
            creatinine=creatinine, sodium=sodium, potassium=potassium,
        )
        step3_errors = [e for e in step3_errors if "Patient Name" not in e and "Age" not in e and "Height" not in e and "Weight" not in e]

        if step3_errors:
            st.error("### ⚠️ Please fix the following before proceeding:")
            for err in step3_errors:
                st.markdown(err)

        render_wizard_navigation(3)

    
# ── Step 4: Goals & Review ─────────────────────────────────────────
    elif st.session_state.wizard_step == 4:
        st.markdown(
            "<div class='form-section-card'>"
            "<div class='form-section-title'>"
            "  <div class='fst-icon'>🎯</div>"
            "  <h3>Goals &amp; Recommendation Settings</h3>"
            "  <span class='fst-badge'>Step 4 of 4</span>"
            "</div>",
            unsafe_allow_html=True,
        )

        g1, g2, g3, g4 = st.columns(4)

        _goal_options = ["Weight Loss", "Weight Gain", "Weight Maintenance"]
        with g1:
            goal = st.selectbox(
                "Health Goal",
                _goal_options,
                index=_goal_options.index(
                    st.session_state.get("goal", "Weight Loss")
                ),
                help="Your primary weight management objective.",
                key="step4_goal",
            )
        with g2:
            region = st.selectbox(
                "State / Union Territory",
                _INDIAN_STATES,
                index=_INDIAN_STATES.index(
                    st.session_state.get("region", "Andhra Pradesh")
                ) if st.session_state.get("region", "Andhra Pradesh") in _INDIAN_STATES else 0,
                help="Used to tailor regional food recommendations.",
                key="step4_region",
            )
        with g3:
            auto_predict = st.checkbox(
                "Auto Disease Prediction",
                value=st.session_state.get("auto_predict", True),
                help="Use AI models to predict diseases automatically.",
                key="step4_auto_predict",
            )
            manual_override = st.checkbox(
                "Manual Disease Override",
                value=st.session_state.get("manual_override", False),
                help="Manually select diseases instead of AI prediction.",
                key="step4_manual_override",
            )
        with g4:
            if NCF_AVAILABLE:
                use_ncf = st.checkbox(
                    "AI Food Recommendations (NCF)",
                    value=st.session_state.get("use_ncf", False),
                    help="Enable Neural Collaborative Filtering for personalised food recommendations.",
                    key="step4_use_ncf",
                )
            else:
                use_ncf = False
                st.caption("NCF unavailable — rule-based recommendations active.")

        manual_diseases: list = []
        if manual_override:
            manual_diseases = st.multiselect(
                "Select Diseases (Manual Override)",
                ["Diabetes", "Obesity", "Kidney Disease"],
                default=st.session_state.get("manual_diseases", []),
                help="These will replace the AI predictions.",
                key="step4_manual_diseases",
            )

        st.markdown("</div>", unsafe_allow_html=True)   # close form-section-card

        # Save step 4 values to session state
        st.session_state.goal = goal
        st.session_state.region = region
        st.session_state.auto_predict = auto_predict
        st.session_state.manual_override = manual_override
        st.session_state.use_ncf = use_ncf
        st.session_state.manual_diseases = manual_diseases

        # ── Review Summary ───────────────────────────────────────────────
        st.markdown(
            "<div class='form-section-card'>"
            "<div class='form-section-title'>"
            "  <div class='fst-icon'>📋</div>"
            "  <h3>Review Your Information</h3>"
            "  <span class='fst-badge'>Confirm before analysis</span>"
            "</div>",
            unsafe_allow_html=True,
        )

        # Display summary of all entered values
        review_col1, review_col2 = st.columns(2)
        
        with review_col1:
            st.markdown("**Personal Information**")
            st.markdown(f"- **Name:** {st.session_state.get('pat_name', '—')}")
            st.markdown(f"- **Age:** {st.session_state.get('age', '—')} years")
            st.markdown(f"- **Gender:** {st.session_state.get('gender', '—')}")
            st.markdown(f"- **Height:** {st.session_state.get('height', '—')} cm")
            st.markdown(f"- **Weight:** {st.session_state.get('weight', '—')} kg")
            st.markdown(f"- **Activity Level:** {st.session_state.get('activity', '—')}")
        
        with review_col2:
            st.markdown("**Medical Parameters**")
            st.markdown(f"- **HbA1c:** {st.session_state.get('hba1c', '—')} %")
            st.markdown(f"- **Blood Glucose:** {st.session_state.get('glucose', '—')} mg/dL")
            st.markdown(f"- **Systolic BP:** {st.session_state.get('bp', '—')} mmHg")
            st.markdown(f"- **Diastolic BP:** {st.session_state.get('bp_diastolic', '—')} mmHg")
            st.markdown(f"- **Serum Creatinine:** {st.session_state.get('creatinine', '—')} mg/dL")
            st.markdown(f"- **Sodium:** {st.session_state.get('sodium', '—')} mEq/L")
            st.markdown(f"- **Potassium:** {st.session_state.get('potassium', '—')} mEq/L")

        st.markdown("**Goals & Settings**")
        g_review_col1, g_review_col2 = st.columns(2)
        with g_review_col1:
            st.markdown(f"- **Health Goal:** {st.session_state.get('goal', '—')}")
            st.markdown(f"- **Region:** {st.session_state.get('region', '—')}")
        with g_review_col2:
            st.markdown(f"- **Auto Prediction:** {'✓' if st.session_state.get('auto_predict', True) else '✗'}")
            st.markdown(f"- **Manual Override:** {'✓' if st.session_state.get('manual_override', False) else '✗'}")
            st.markdown(f"- **NCF Recommendations:** {'✓' if st.session_state.get('use_ncf', False) else '✗'}")
        
        if st.session_state.get('manual_override', False):
            st.markdown(f"- **Manual Diseases:** {', '.join(st.session_state.get('manual_diseases', []))}")

        st.markdown("</div>", unsafe_allow_html=True)   # close form-section-card

        # Navigation buttons for Step 4
        col_back, col_analyze = st.columns([1, 2])
        
        with col_back:
            if st.button("← Back", use_container_width=True, key="wizard_back_step4"):
                st.session_state.wizard_step = 3
                st.rerun()
        
        with col_analyze:
            # Check if analysis is in progress
            if st.session_state.get("analysis_in_progress", False):
                analyze = st.button(
                    "⏳ Analysing...",
                    use_container_width=True,
                    type="primary",
                    key="wizard_analyze",
                    disabled=True,
                )
            else:
                analyze = st.button(
                    "🔍 Analyse My Health",
                    use_container_width=True,
                    type="primary",
                    key="wizard_analyze",
                )

# ════════════════════════════════════════════════════════════════════
#  ANALYSIS  — triggered by Analyse button on Step 4.
#  All variable names are read from session state.
# ════════════════════════════════════════════════════════════════════
if analyze:

    # Set loading state
    st.session_state.analysis_in_progress = True
    st.session_state.analysis_error = None
    st.session_state.loading_step = 1
    st.rerun()

# Show loading state if analysis is in progress
if st.session_state.get("analysis_in_progress", False):
    
    # Get current loading step
    loading_step = st.session_state.get("loading_step", 1)
    
    # Loading UI with dynamic progress
    step_messages = {
        1: "Validating your information...",
        2: "Analysing health parameters...",
        3: "Generating personalized recommendations...",
        4: "Preparing your results..."
    }
    
    current_message = step_messages.get(loading_step, "Processing...")
    
    st.markdown(
        f"""
        <div class='analysis-loading-card'>
            <div class='loading-spinner'></div>
            <div class='loading-content'>
                <h3 class='loading-title'>Analysing your health profile...</h3>
                <p class='loading-subtitle'>Please wait while we generate your personalized results.</p>
                <div class='loading-progress'>
                    <span class='loading-message'>{current_message}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Perform the actual analysis based on loading step
    try:
        # Step 1: Validation
        if loading_step == 1:
            # ── Read all values from session state ───────────────────────────────
            pat_name = st.session_state.get("pat_name", "")
            age = st.session_state.get("age", 30)
            gender = st.session_state.get("gender", "Male")
            height = st.session_state.get("height", 170)
            weight = st.session_state.get("weight", 70)
            activity = st.session_state.get("activity", "Sedentary")
            hba1c = st.session_state.get("hba1c", 5.5)
            glucose = st.session_state.get("glucose", 90)
            bp = st.session_state.get("bp", 120)
            bp_diastolic = st.session_state.get("bp_diastolic", 80)
            creatinine = st.session_state.get("creatinine", 1.0)
            sodium = st.session_state.get("sodium", 138.0)
            potassium = st.session_state.get("potassium", 4.5)
            goal = st.session_state.get("goal", "Weight Loss")
            region = st.session_state.get("region", "Andhra Pradesh")
            use_ncf = st.session_state.get("use_ncf", False)
            auto_predict = st.session_state.get("auto_predict", True)
            manual_override = st.session_state.get("manual_override", False)
            manual_diseases = st.session_state.get("manual_diseases", [])

            # ── Run validation before any calculation ────────────────────────
            _errors = _validate_inputs(
                pat_name=pat_name, age=age, height=height, weight=weight,
                hba1c=hba1c, glucose=glucose, bp=bp, bp_diastolic=bp_diastolic,
                creatinine=creatinine, sodium=sodium, potassium=potassium,
            )

            if _errors:
                st.session_state.analysis_in_progress = False
                st.session_state.analysis_error = "validation"
                st.session_state.validation_errors = _errors
                st.rerun()
            else:
                # Store validated data for next step
                st.session_state.analysis_data = {
                    "pat_name": pat_name, "age": age, "gender": gender,
                    "height": height, "weight": weight, "activity": activity,
                    "hba1c": hba1c, "glucose": glucose, "bp": bp,
                    "bp_diastolic": bp_diastolic, "creatinine": creatinine,
                    "sodium": sodium, "potassium": potassium, "goal": goal,
                    "region": region, "use_ncf": use_ncf, "auto_predict": auto_predict,
                    "manual_override": manual_override, "manual_diseases": manual_diseases
                }
                st.session_state.loading_step = 2
                st.rerun()
        
        # Step 2: Core calculations and predictions
        elif loading_step == 2:
            data = st.session_state.analysis_data
            
            # ── Core calculations ─────────────────────────────────────────
            bmi     = calculate_bmi(data["weight"], data["height"])
            bmi_cat = bmi_category(bmi)
            bmr     = calculate_bmr(data["age"], data["gender"], data["weight"], data["height"])
            tdee    = calculate_tdee(bmr, ACTIVITY_FACTOR[data["activity"]])

            # ── Disease predictions ───────────────────────────────────────
            obesity_result  = predict_obesity(data["age"], data["gender"], bmi)
            diabetes_result = predict_diabetes(data["age"], data["gender"], bmi, data["hba1c"], data["glucose"])
            kidney_result   = predict_kidney(data["age"], data["gender"], bmi, data["sodium"], data["potassium"], data["bp"], data["creatinine"])

            obesity_label      = obesity_result["label"]      if isinstance(obesity_result,  dict) else obesity_result
            obesity_confidence = obesity_result["confidence"] if isinstance(obesity_result,  dict) else None
            diabetes_label     = diabetes_result["label"]     if isinstance(diabetes_result, dict) else diabetes_result
            diabetes_confidence= diabetes_result["confidence"]if isinstance(diabetes_result, dict) else None
            kidney_label       = kidney_result["label"]       if isinstance(kidney_result,   dict) else kidney_result
            kidney_confidence  = kidney_result["confidence"]  if isinstance(kidney_result,   dict) else None

            # Apply manual override if selected
            if data["manual_override"] and data["manual_diseases"]:
                obesity  = {"label": "Obesity (Manual)",        "confidence": None} if "Obesity"       in data["manual_diseases"] else obesity_result
                diabetes = {"label": "Diabetes (Manual)",       "confidence": None} if "Diabetes"      in data["manual_diseases"] else diabetes_result
                kidney   = {"label": "Kidney Disease (Manual)", "confidence": None} if "Kidney Disease" in data["manual_diseases"] else kidney_result
                diseases = list(data["manual_diseases"])
            else:
                obesity  = obesity_result
                diabetes = diabetes_result
                kidney   = kidney_result
                diseases = []
                diabetes_str = str(diabetes_label).lower() if diabetes_label is not None else ""
                kidney_str   = str(kidney_label).lower()   if kidney_label   is not None else ""
                obesity_str  = str(obesity_label).lower()  if obesity_label  is not None else ""
                if diabetes_str == "diabetes":
                    diseases.append("Diabetes")
                if kidney_str == "kidney disease":
                    diseases.append("Kidney Disease")
                if obesity_str in ["obese class i", "obese class ii", "obese class iii", "overweight"]:
                    diseases.append("Obesity")
                if not diseases:
                    diseases = ["Normal"]

            # ── Derive severity from RiskResult.final_status ─────────────
            # Each predict_* function returns {"label": ..., "confidence": ..., "risk": RiskResult}
            # RiskResult.final_status is the functional severity proxy:
            #   "High Risk"     → "severe"
            #   "Moderate Risk" → "moderate"
            #   "Borderline" /
            #   "Model Flag"    → "mild"
            #   "Low Risk"      → None  (no disease restrictions)
            # Manual overrides have no RiskResult → fall back to "moderate" (safe default)
            try:
                from backend.services.severity_rules import build_severity_dict

                def _get_final_status(result):
                    """Safely extract final_status from a predict_* result dict."""
                    if isinstance(result, dict):
                        risk = result.get("risk")
                        if risk is not None and hasattr(risk, "final_status"):
                            return risk.final_status
                    return None

                _diab_fs   = _get_final_status(diabetes_result)
                _kidney_fs = _get_final_status(kidney_result)
                _obese_fs  = _get_final_status(obesity_result)

                # If manual override is active, there is no RiskResult → pass None
                # (build_severity_dict maps None → None → safe default in recommender)
                if data["manual_override"] and data["manual_diseases"]:
                    _diab_fs   = _get_final_status(diabetes_result)   if "Diabetes"      not in data["manual_diseases"] else None
                    _kidney_fs = _get_final_status(kidney_result)     if "Kidney Disease" not in data["manual_diseases"] else None
                    _obese_fs  = _get_final_status(obesity_result)    if "Obesity"        not in data["manual_diseases"] else None

                severity_dict = build_severity_dict(_diab_fs, _kidney_fs, _obese_fs)
            except Exception:
                severity_dict = {}   # safe fallback — recommender uses SEVERITY_DEFAULT

            # Store intermediate results
            st.session_state.analysis_data.update({
                "bmi": bmi, "bmi_cat": bmi_cat, "bmr": bmr, "tdee": tdee,
                "obesity_result": obesity_result, "diabetes_result": diabetes_result, "kidney_result": kidney_result,
                "obesity": obesity, "diabetes": diabetes, "kidney": kidney,
                "obesity_label": obesity_label, "obesity_confidence": obesity_confidence,
                "diabetes_label": diabetes_label, "diabetes_confidence": diabetes_confidence,
                "kidney_label": kidney_label, "kidney_confidence": kidney_confidence,
                "diseases": diseases,
                "severity": severity_dict,   # ← new: per-disease severity levels
            })
            st.session_state.loading_step = 3
            st.rerun()
        
        # Step 3: Recommendations
        elif loading_step == 3:
            data = st.session_state.analysis_data
            
            # ── NCF recommendations (optional) ────────────────────────────
            _use_ncf = data["use_ncf"] and NCF_AVAILABLE
            if _use_ncf:
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
                        user_id=user_id, age=data["age"], gender=data["gender"], bmi=data["bmi"],
                        hba1c=data["hba1c"], glucose=data["glucose"], sodium=data["sodium"],
                        potassium=data["potassium"], bp=data["bp"], creatinine=data["creatinine"], top_n=20,
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
                    _use_ncf = False

            # ── Rule-based recommendations ────────────────────────────────
            recommendations = generate_comprehensive_recommendations(
                diseases=data["diseases"], age=data["age"], gender=data["gender"],
                height=data["height"], weight=data["weight"], bmi=data["bmi"],
                activity_level=data["activity"], daily_calories=data["tdee"],
                hba1c=data["hba1c"], glucose=data["glucose"], bp=data["bp"],
                sodium=data["sodium"], potassium=data["potassium"], creatinine=data["creatinine"],
                goal=data.get("goal"), region=data.get("region"),
                severity=data.get("severity", {}),   # ← severity dict from Step 2
            )

            st.session_state.analysis_data.update({
                "recommendations": recommendations, "use_ncf": _use_ncf
            })
            st.session_state.loading_step = 4
            st.rerun()
        
        # Step 4: Finalize and store results
        elif loading_step == 4:
            data = st.session_state.analysis_data
            
            # ── Build report_data ─────────────────────────────────────────
            report_data = {
                "personal": {
                    "name":     data["pat_name"] or "—",
                    "age":      data["age"],
                    "gender":   data["gender"],
                    "height":   f"{data['height']} cm",
                    "weight":   f"{data['weight']} kg",
                    "activity": data["activity"],
                    "region":   data["region"],
                },
                "metrics": {
                    "BMI":                     round(data["bmi"], 2),
                    "BMI Category":            data["bmi_cat"],
                    "BMR (kcal)":              round(data["bmr"], 2),
                    "TDEE (kcal)":             round(data["tdee"], 2),
                    "Water Intake (L)":        data["recommendations"].get("water_intake"),
                    "Protein Requirement (g)": data["recommendations"].get("protein_requirement"),
                },
                "predictions": {
                    "Obesity": {
                        "label":      data["obesity_label"] if isinstance(data["obesity"], dict) else str(data["obesity"]),
                        "confidence": data["obesity_confidence"] if isinstance(data["obesity"], dict) else None,
                    },
                    "Diabetes": {
                        "label":      data["diabetes_label"] if isinstance(data["diabetes"], dict) else str(data["diabetes"]),
                        "confidence": data["diabetes_confidence"] if isinstance(data["diabetes"], dict) else None,
                    },
                    "Kidney Disease": {
                        "label":      data["kidney_label"] if isinstance(data["kidney"], dict) else str(data["kidney"]),
                        "confidence": data["kidney_confidence"] if isinstance(data["kidney"], dict) else None,
                    },
                },
                "nutrition_tips": data["recommendations"].get("nutrition_tips", []),
                "foods_to_avoid": data["recommendations"].get("foods_to_avoid", []),
            }

            # ── Persist to SQLite ─────────────────────────────────────────
            try:
                save_analysis(
                    user_id=None,
                    patient_name=data["pat_name"] or username or "unknown",
                    data=report_data,
                )
            except Exception:
                pass

            # ── Store everything in session state ─────────────────────────
            st.session_state.report_data = report_data
            st.session_state.analysis = {
                "bmi": data["bmi"], "bmi_cat": data["bmi_cat"], "bmr": data["bmr"], "tdee": data["tdee"],
                "obesity":  data["obesity"],  "diabetes":  data["diabetes"],  "kidney": data["kidney"],
                "obesity_label":       data["obesity_label"],
                "obesity_confidence":  data["obesity_confidence"],
                "diabetes_label":      data["diabetes_label"],
                "diabetes_confidence": data["diabetes_confidence"],
                "kidney_label":        data["kidney_label"],
                "kidney_confidence":   data["kidney_confidence"],
                # RiskResult objects
                "obesity_risk":  data["obesity_result"].get("risk")  if isinstance(data["obesity_result"],  dict) else None,
                "diabetes_risk": data["diabetes_result"].get("risk") if isinstance(data["diabetes_result"], dict) else None,
                "kidney_risk":   data["kidney_result"].get("risk")   if isinstance(data["kidney_result"],   dict) else None,
                "diseases":         data["diseases"],
                "recommendations":  data["recommendations"],
                "use_ncf":          data["use_ncf"],
                # Input values for Profile tab display
                "pat_name":    data["pat_name"],    "age":      data["age"],
                "gender":      data["gender"],      "height":   data["height"],
                "weight":      data["weight"],      "activity": data["activity"],
                "region":      data["region"],      "goal":     data["goal"],
                "bp_diastolic":data["bp_diastolic"],
                # Medical params for XAI consistency
                "hba1c":      float(data["hba1c"]),
                "glucose":    float(data["glucose"]),
                "bp":         int(data["bp"]),
                "sodium":     float(data["sodium"]),
                "potassium":  float(data["potassium"]),
                "creatinine": float(data["creatinine"]),
            }

            # Clear loading state and temporary data
            if "analysis_data" in st.session_state:
                del st.session_state.analysis_data
            if "loading_step" in st.session_state:
                del st.session_state.loading_step
            st.session_state.analysis_in_progress = False
            st.rerun()   # re-render so the dashboard tabs appear immediately

    except Exception as e:
        # Handle error with user-friendly message
        st.session_state.analysis_in_progress = False
        st.session_state.analysis_error = "general"
        st.session_state.error_message = str(e)
        # Clean up temporary data
        if "analysis_data" in st.session_state:
            del st.session_state.analysis_data
        if "loading_step" in st.session_state:
            del st.session_state.loading_step
        st.rerun()

# Show error state if analysis failed
if st.session_state.get("analysis_error") and not st.session_state.get("analysis_in_progress", False):
    
    error_type = st.session_state.analysis_error
    
    if error_type == "validation":
        st.error("### ⚠️ Please fix the following before running the analysis:")
        for _err in st.session_state.get("validation_errors", []):
            st.markdown(_err)
        
        if st.button("🔄 Try Again", use_container_width=True, type="primary", key="retry_validation"):
            st.session_state.analysis_error = None
            st.session_state.validation_errors = None
            st.rerun()
    
    else:
        st.markdown(
            """
            <div class='analysis-error-card'>
                <div class='error-icon'>⚠️</div>
                <div class='error-content'>
                    <h3 class='error-title'>Unable to analyse your profile right now</h3>
                    <p class='error-subtitle'>Please try again.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        if st.button("🔄 Try Again", use_container_width=True, type="primary", key="retry_analysis"):
            st.session_state.analysis_error = None
            st.session_state.error_message = None
            st.rerun()

# ════════════════════════════════════════════════════════════════════
#  MAIN CONTENT — dashboard tabs (shown only after analysis)
# ════════════════════════════════════════════════════════════════════

if "analysis" not in st.session_state:
    # Welcome screen shown until the user submits the form
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
    obesity_label      = a.get("obesity_label",      str(obesity)  if not isinstance(obesity,  dict) else obesity.get("label",  ""))
    obesity_confidence = a.get("obesity_confidence", None)
    diabetes_label     = a.get("diabetes_label",     str(diabetes) if not isinstance(diabetes, dict) else diabetes.get("label", ""))
    diabetes_confidence= a.get("diabetes_confidence",None)
    kidney_label       = a.get("kidney_label",       str(kidney)   if not isinstance(kidney,   dict) else kidney.get("label",   ""))
    kidney_confidence  = a.get("kidney_confidence",  None)
    obesity_risk  = a.get("obesity_risk",  None)
    diabetes_risk = a.get("diabetes_risk", None)
    kidney_risk   = a.get("kidney_risk",   None)
    pat_name     = a.get("pat_name",  "")
    age          = a.get("age",       30)
    gender       = a.get("gender",    "Male")
    height       = a.get("height",    170)
    weight       = a.get("weight",    70)
    activity     = a.get("activity",  "Sedentary")
    region       = a.get("region",    "Andhra Pradesh")
    goal         = a.get("goal",      "Weight Loss")
    bp_diastolic = a.get("bp_diastolic", 80)
    manual_override  = st.session_state.get("manual_override", False)
    manual_diseases  = st.session_state.get("manual_diseases", [])

    # ── Quick-stats strip ─────────────────────────────────────────────
    components.dashboard_stats_strip(bmi, bmi_cat, tdee, diseases)

    # ══════════════════════════════════════════════════════════════════
    #  SECTION — Patient Profile
    # ══════════════════════════════════════════════════════════════════
    components.section_header("👤", "Patient Profile")

    col_a, col_b = st.columns(2, gap="large")
    left_items = [
        ("🧑", "Name",          pat_name or "—"),
        ("📅", "Age",           f"{age} years"),
        ("⚥",  "Gender",        gender),
    ]
    right_items = [
        ("📏", "Height",         f"{height} cm"),
        ("⚖️", "Weight",         f"{weight} kg"),
        ("🏃", "Activity Level", activity),
    ]
    with col_a:
        for ico, lbl, val in left_items:
            components.profile_card(ico, lbl, val)
    with col_b:
        for ico, lbl, val in right_items:
            components.profile_card(ico, lbl, val)

    components.section_header("🩺", "Blood Pressure")
    bp_col1, bp_col2 = st.columns(2, gap="large")
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

    st.markdown("<div class='page-divider'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════
    #  SECTION — Health Summary
    # ══════════════════════════════════════════════════════════════════
    components.section_header("📊", "Key Health Metrics")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        components.metric_card("BMI", f"{bmi:.1f}", "⚖️", "#2563EB")
    with c2:
        cat_color = (
            "#16A34A" if "normal"     in bmi_cat.lower() else
            "#D97706" if "overweight" in bmi_cat.lower() else
            "#DC2626"
        )
        components.metric_card("BMI Category", bmi_cat, "🏷️", cat_color)
    with c3:
        components.metric_card("Daily Calories", f"{tdee:.0f} kcal", "🔥", "#D97706")
    with c4:
        components.metric_card("BMR", f"{bmr:.0f} kcal", "⚡", "#7C3AED")

    c5, c6 = st.columns(2)
    with c5:
        components.metric_card(
            "Water Intake",
            f"{recommendations.get('water_intake', '—')} L",
            "💧", "#0284C7",
        )
    with c6:
        components.metric_card(
            "Protein Requirement",
            f"{recommendations.get('protein_requirement', '—')} g",
            "🥩", "#16A34A",
        )

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

    components.section_header("📈", "Visual Analytics")
    ch1, ch2 = st.columns(2, gap="large")

    with ch1:
        components.chart_bmi_gauge(bmi)

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
        if meal_cals:
            components.chart_calorie_breakdown(pd.DataFrame(meal_cals))
        else:
            st.info("Calorie breakdown chart will appear after meal plan is generated.")

    st.markdown("<div class='page-divider'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════
    #  SECTION — Disease Risk Predictions
    # ══════════════════════════════════════════════════════════════════
    components.section_header("🧬", "Disease Risk Predictions")

    if manual_override and manual_diseases:
        components.status_banner(
            "🔧", "Manual Override Active",
            f"Showing results for manually selected conditions: "
            f"<strong>{', '.join(manual_diseases)}</strong>. "
            f"AI predictions have been replaced.",
            "info",
        )

    pred_cols = st.columns(3, gap="large")

    with pred_cols[0]:
        _d_risk  = diabetes_risk
        _d_level = _d_risk.card_level        if _d_risk else "low"
        _d_badge = _d_risk.card_risk_text    if _d_risk else "Unknown"
        _d_prob  = _d_risk.model_probability if _d_risk else diabetes_confidence
        _d_fs    = _d_risk.final_status      if _d_risk else "Unknown"
        components.prediction_card(
            icon="🩸", name="Diabetes",
            pred_class=diabetes_label,
            risk=_d_badge, level=_d_level,
            final_status=_d_fs, model_probability=_d_prob,
        )
        if _d_risk:
            components.status_banner("🩸", _d_risk.banner_title, _d_risk.banner_body, _d_risk.banner_level)
        else:
            components.status_banner("🩸", "Diabetes — No Data", "Run the analysis to see the diabetes prediction.", "info")

    with pred_cols[1]:
        _o_risk  = obesity_risk
        _o_level = _o_risk.card_level        if _o_risk else "low"
        _o_badge = _o_risk.card_risk_text    if _o_risk else "Unknown"
        _o_prob  = _o_risk.model_probability if _o_risk else obesity_confidence
        _o_fs    = _o_risk.final_status      if _o_risk else "Unknown"
        components.prediction_card(
            icon="⚖️", name="Obesity",
            pred_class=obesity_label,
            risk=_o_badge, level=_o_level,
            final_status=_o_fs, model_probability=_o_prob,
        )
        if _o_risk:
            components.status_banner("⚖️", _o_risk.banner_title, _o_risk.banner_body, _o_risk.banner_level)
        else:
            components.status_banner("⚖️", "Obesity — No Data", "Run the analysis to see the obesity prediction.", "info")

    with pred_cols[2]:
        _k_risk  = kidney_risk
        _k_level = _k_risk.card_level        if _k_risk else "low"
        _k_badge = _k_risk.card_risk_text    if _k_risk else "Unknown"
        _k_prob  = _k_risk.model_probability if _k_risk else kidney_confidence
        _k_fs    = _k_risk.final_status      if _k_risk else "Unknown"
        components.prediction_card(
            icon="🫘", name="Kidney Disease",
            pred_class=kidney_label,
            risk=_k_badge, level=_k_level,
            final_status=_k_fs, model_probability=_k_prob,
        )
        if _k_risk:
            components.status_banner("🫘", _k_risk.banner_title, _k_risk.banner_body, _k_risk.banner_level)
        else:
            components.status_banner("🫘", "Kidney Disease — No Data", "Run the analysis to see the kidney disease prediction.", "info")

    components.section_header("📋", "Overall Risk Summary")
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
            n for n, r in [("Diabetes", diabetes_risk), ("Obesity", obesity_risk), ("Kidney Disease", kidney_risk)]
            if r is not None and r.final_status in ("High Risk", "Moderate Risk")
        )
        components.status_banner("⚠️", "Elevated Risk Detected",
            f"Identified: <strong>{_flagged_names}</strong>. "
            "Personalised meal and nutrition recommendations are below.", "warning")
    else:
        _flag_names = ", ".join(
            n for n, r in [("Diabetes", diabetes_risk), ("Obesity", obesity_risk), ("Kidney Disease", kidney_risk)]
            if r is not None and r.final_status == "Model Flag"
        )
        components.status_banner("🔵", "Screening Flags — Clinical Markers Normal",
            f"The model flagged: <strong>{_flag_names}</strong>. "
            "All measured clinical markers are within normal reference ranges. "
            "These are model screening signals, not confirmed diagnoses.", "info")

    # ── XAI (Explainable AI) ──────────────────────────────────────────
    _xai_hba1c      = a.get("hba1c",      6.5)
    _xai_glucose    = a.get("glucose",     120.0)
    _xai_bp         = a.get("bp",          120)
    _xai_sodium     = a.get("sodium",      138.0)
    _xai_potassium  = a.get("potassium",   4.5)
    _xai_creatinine = a.get("creatinine",  1.0)

    if XAI_AVAILABLE:
        components.section_header("🔬", "Explainable AI — Why These Predictions?")

        # Diabetes XAI
        _conf_diab = (
            f"&nbsp;&nbsp;<span class='xai-conf-chip'>Confidence:&nbsp;{diabetes_confidence}%</span>"
            if diabetes_confidence is not None else ""
        )
        st.markdown(
            f"<div class='xai-tab-header'><span style='font-size:1.4rem;'>🩸</span>"
            f"<span>Diabetes &nbsp;—&nbsp; <em>{diabetes_label}</em>{_conf_diab}</span></div>",
            unsafe_allow_html=True,
        )
        _xai_diab = explain_diabetes(age=age, gender=gender, bmi=bmi, hba1c=_xai_hba1c, glucose=_xai_glucose, label=diabetes_label)
        components.xai_explanation_panel(_xai_diab, "Diabetes")

        st.markdown("<div class='page-divider'></div>", unsafe_allow_html=True)

        # Obesity XAI
        _conf_ob = (
            f"&nbsp;&nbsp;<span class='xai-conf-chip'>Confidence:&nbsp;{obesity_confidence}%</span>"
            if obesity_confidence is not None else ""
        )
        st.markdown(
            f"<div class='xai-tab-header'><span style='font-size:1.4rem;'>⚖️</span>"
            f"<span>Obesity &nbsp;—&nbsp; <em>{obesity_label}</em>{_conf_ob}</span></div>",
            unsafe_allow_html=True,
        )
        _xai_ob = explain_obesity(age=age, gender=gender, bmi=bmi, label=obesity_label)
        components.xai_explanation_panel(_xai_ob, "Obesity")

        st.markdown("<div class='page-divider'></div>", unsafe_allow_html=True)

        # Kidney XAI
        _conf_kid = (
            f"&nbsp;&nbsp;<span class='xai-conf-chip'>Confidence:&nbsp;{kidney_confidence}%</span>"
            if kidney_confidence is not None else ""
        )
        st.markdown(
            f"<div class='xai-tab-header'><span style='font-size:1.4rem;'>🫘</span>"
            f"<span>Kidney Disease &nbsp;—&nbsp; <em>{kidney_label}</em>{_conf_kid}</span></div>",
            unsafe_allow_html=True,
        )
        _xai_kid = explain_kidney(
            age=age, gender=gender, bmi=bmi,
            sodium=_xai_sodium, potassium=_xai_potassium,
            bp=_xai_bp, creatinine=_xai_creatinine, label=kidney_label,
        )
        components.xai_explanation_panel(_xai_kid, "Kidney Disease")
    else:
        st.info("ℹ️ Explainable AI module is not available. Check that backend/xai.py is present and all models are loaded.")

    st.markdown("<div class='page-divider'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════
    #  SECTION — Nutrition Recommendations & Meal Plan
    # ══════════════════════════════════════════════════════════════════

    if use_ncf and "ncf_recommendations" in st.session_state:
        components.section_header("🤖", "AI-Powered Recommendations (NCF)")
        components.status_banner(
            "🤖", "Neural Collaborative Filtering Active",
            st.session_state.get("ncf_explanation", "Personalised recommendations from your profile."),
            "info",
        )
        ncf_df = st.session_state.ncf_recommendations
        if not ncf_df.empty:
            st.dataframe(ncf_df, use_container_width=True, hide_index=True)
        detected = st.session_state.get("ncf_diseases", [])
        if detected and detected != ["normal"]:
            st.markdown("**Detected conditions:** " + ", ".join(d.replace("_", " ").title() for d in detected))

    components.section_header("🍽️", "Personalised Daily Meal Plan")
    
    # Get the original meal plan from recommendations
    meal_plan = recommendations.get("meal_plan", {})
    
    # Check if enhanced recommender is being used
    use_enhanced = recommendations.get("use_enhanced_recommender", False)
    use_improved_planner = recommendations.get("use_improved_planner", False)
    meal_validation = recommendations.get("meal_validation", {})
    
    # Display calorie targeting info if using enhanced recommender
    if use_enhanced:
        target_calories = recommendations.get("target_calories", tdee)
        total_calories = recommendations.get("total_calories", 0)
        calorie_diff = recommendations.get("calorie_difference", 0)
        
        # Calorie summary card
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Daily Target", 
                f"{target_calories:.0f} kcal",
                help="Your recommended daily calorie intake based on TDEE"
            )
        with col2:
            st.metric(
                "Recommended", 
                f"{total_calories:.0f} kcal",
                f"{calorie_diff:+.0f} kcal" if abs(calorie_diff) > 10 else "On target"
            )
        with col3:
            nutrition_summary = recommendations.get("nutrition_summary", {})
            st.metric(
                "Total Protein", 
                f"{nutrition_summary.get('protein', 0):.1f}g"
            )
        
        st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
    
    # Display validation warnings if any
    if meal_validation.get('warnings'):
        for warning in meal_validation['warnings']:
            st.warning(f"⚠️ {warning}")
    
    if meal_plan:
        meal_explanations = recommendations.get("meal_explanations", {})
        
        # Initialize swap state if not exists
        if "swap_state" not in st.session_state:
            st.session_state.swap_state = {
                "active_swap": None,  # (meal_type, food_index, current_food)
                "alternatives": None,
                "selected_alternative": None
            }
        
        for meal_name, meal_data in meal_plan.items():
            components.meal_tag(meal_name.capitalize())
            
            # Display meal explanation if available
            if use_enhanced and meal_name in meal_explanations:
                st.info(f"💡 {meal_explanations[meal_name]}")
            
            if isinstance(meal_data, pd.DataFrame) and not meal_data.empty:
                # Display each food item
                for idx, row in meal_data.iterrows():
                    food_name = row.get("Dish Name", "Unknown Food")
                    calories = row.get("Calories (kcal)", 0)
                    protein = row.get("Protein (g)", None)
                    carbs = row.get("Carbohydrates (g)", None)
                    fat = row.get("Fats (g)", None)
                    fiber = row.get("Fibre (g)", None)
                    
                    # Check if this food is being swapped
                    is_swapping = (st.session_state.swap_state["active_swap"] == 
                                  (meal_name, idx, food_name))
                    
                    # Display food item card with swap button
                    swap_clicked = components.food_item_card(
                        food_name=food_name,
                        calories=calories,
                        protein=protein,
                        carbs=carbs,
                        fat=fat,
                        fiber=fiber,
                        meal_type=meal_name,
                        food_index=idx,
                        show_swap=use_enhanced and not is_swapping
                    )
                    
                    if swap_clicked and use_enhanced:
                        st.session_state.swap_state["active_swap"] = (meal_name, idx, food_name)
                        st.rerun()
                    
                    st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)
                
            elif isinstance(meal_data, list) and meal_data:
                # Display balanced meal format
                if (use_improved_planner or use_enhanced) and len(meal_data) <= 3:
                    # This is likely a balanced meal (main + protein + vegetable)
                    _display_balanced_meal(meal_data, meal_name, diseases, tdee)
                else:
                    # Display each food item from list format (fallback)
                    for idx, food_item in enumerate(meal_data):
                        if isinstance(food_item, dict):
                            food_name = food_item.get("Dish Name", food_item.get("food_name", "Unknown Food"))
                            calories = food_item.get("Calories (kcal)", food_item.get("Calories", food_item.get("calories", 0)))
                            protein = food_item.get("Protein (g)", food_item.get("Protein", food_item.get("protein", None)))
                            carbs = food_item.get("Carbohydrates (g)", food_item.get("Carbohydrates", food_item.get("carbs", None)))
                            fat = food_item.get("Fats (g)", food_item.get("Fats", food_item.get("fat", None)))
                            fiber = food_item.get("Fibre (g)", food_item.get("Fibre", food_item.get("fiber", None)))
                            
                            # Check if this food is being swapped
                            is_swapping = (st.session_state.swap_state["active_swap"] == 
                                          (meal_name, idx, food_name))
                            
                            # Display food item card with swap button
                            swap_clicked = components.food_item_card(
                                food_name=food_name,
                                calories=calories,
                                protein=protein,
                                carbs=carbs,
                                fat=fat,
                                fiber=fiber,
                                meal_type=meal_name,
                                food_index=idx,
                                show_swap=use_enhanced and not is_swapping
                            )
                            
                            if swap_clicked and use_enhanced:
                                st.session_state.swap_state["active_swap"] = (meal_name, idx, food_name)
                                st.rerun()
                            
                            st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)
            else:
                st.info(f"No {meal_name.lower()} data available.")
            
            st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
        
        # Handle active swap - show alternatives
        if use_enhanced and st.session_state.swap_state["active_swap"]:
            meal_type, food_index, current_food = st.session_state.swap_state["active_swap"]
            
            st.markdown("<div class='page-divider'></div>", unsafe_allow_html=True)
            components.section_header("🔄", "Food Swap Alternatives")
            
            # Get alternatives using enhanced recommender
            if st.session_state.swap_state["alternatives"] is None:
                try:
                    from backend.services.enhanced_recommender import EnhancedNutritionRecommender
                    
                    # Build user profile
                    user_profile = {
                        'diseases': diseases,
                        'age': age,
                        'gender': gender,
                        'height': height,
                        'weight': weight,
                        'bmi': bmi,
                        'activity_level': activity,
                        'daily_calories': tdee,
                        'hba1c': hba1c,
                        'glucose': glucose,
                        'bp': bp,
                        'sodium': sodium,
                        'potassium': potassium,
                        'creatinine': creatinine,
                        'goal': st.session_state.analysis_data.get("goal", "Weight Loss"),
                        'region': st.session_state.analysis_data.get("region", "Andhra Pradesh")
                    }
                    
                    recommender = EnhancedNutritionRecommender()
                    alternatives = recommender.get_swap_alternatives(
                        current_food=current_food,
                        meal_type=meal_type,
                        user_profile=user_profile,
                        limit=3
                    )
                    
                    st.session_state.swap_state["alternatives"] = alternatives
                    st.rerun()
                except Exception as e:
                    st.error(f"Error finding alternatives: {e}")
                    if st.button("Cancel", key="swap_error_cancel"):
                        st.session_state.swap_state["active_swap"] = None
                        st.session_state.swap_state["alternatives"] = None
                        st.rerun()
            else:
                # Show alternatives
                alternatives = st.session_state.swap_state["alternatives"]
                
                if not alternatives:
                    st.info("No suitable alternatives found.")
                else:
                    st.info(f"Choose an alternative to replace **{current_food}**:")
                    
                    for alt_idx, alt_food in enumerate(alternatives):
                        if isinstance(alt_food, pd.Series):
                            alt_name = alt_food.get("Dish Name", "Unknown")
                            alt_cal = alt_food.get("Calories (kcal)", 0)
                            alt_prot = alt_food.get("Protein (g)", None)
                            alt_carbs = alt_food.get("Carbohydrates (g)", None)
                            alt_fat = alt_food.get("Fats (g)", None)
                            alt_fiber = alt_food.get("Fibre (g)", None)
                        else:
                            alt_name = alt_food.get("Dish Name", alt_food.get("food_name", "Unknown"))
                            alt_cal = alt_food.get("Calories (kcal)", alt_food.get("Calories", alt_food.get("calories", 0)))
                            alt_prot = alt_food.get("Protein (g)", alt_food.get("Protein", alt_food.get("protein", None)))
                            alt_carbs = alt_food.get("Carbohydrates (g)", alt_food.get("Carbohydrates", alt_food.get("carbs", None)))
                            alt_fat = alt_food.get("Fats (g)", alt_food.get("Fats", alt_food.get("fat", None)))
                            alt_fiber = alt_food.get("Fibre (g)", alt_food.get("Fibre", alt_food.get("fiber", None)))
                        
                        # Display alternative card
                        components.food_item_card(
                            food_name=alt_name,
                            calories=alt_cal,
                            protein=alt_prot,
                            carbs=alt_carbs,
                            fat=alt_fat,
                            fiber=alt_fiber,
                            meal_type=meal_type,
                            food_index=alt_idx,
                            show_swap=False
                        )
                        
                        # Select button
                        if st.button(f"Select {alt_name}", key=f"select_alt_{alt_idx}"):
                            # Update meal plan with selected alternative
                            if isinstance(meal_plan[meal_type], list):
                                meal_plan[meal_type][food_index] = alt_food.to_dict() if isinstance(alt_food, pd.Series) else alt_food
                            
                            st.success(f"Replaced {current_food} with {alt_name}!")
                            st.session_state.swap_state["active_swap"] = None
                            st.session_state.swap_state["alternatives"] = None
                            st.rerun()
                        
                        st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)
                
                # Cancel button
                if st.button("Cancel Swap", key="swap_cancel"):
                    st.session_state.swap_state["active_swap"] = None
                    st.session_state.swap_state["alternatives"] = None
                    st.rerun()
    else:
        st.info("Meal plan data is not available.")


def _display_recommendation_charts(meal_plan: dict, nutrition_summary: dict, target_calories: float):
    """Display dynamic charts for recommendations."""
    import plotly.graph_objects as go
    import plotly.express as px
    
    # Calculate meal calorie distribution
    meal_calories = {}
    for meal_name, foods in meal_plan.items():
        if isinstance(foods, list):
            meal_calories[meal_name] = sum(
                f.get('Calories (kcal)', f.get('Calories', f.get('calories', 0)))
                for f in foods if isinstance(f, dict)
            )
    
    # Chart 1: Meal Calorie Distribution (Bar Chart)
    if meal_calories:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Meal Calorie Distribution")
            meal_names = list(meal_calories.keys())
            cal_values = list(meal_calories.values())
            
            fig = go.Figure(data=[
                go.Bar(
                    x=meal_names,
                    y=cal_values,
                    marker_color=['#3B82F6', '#10B981', '#F59E0B', '#8B5CF6'],
                    text=[f"{v:.0f} kcal" for v in cal_values],
                    textposition='outside'
                )
            ])
            
            fig.update_layout(
                title="Calories per Meal",
                xaxis_title="Meal",
                yaxis_title="Calories (kcal)",
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Chart 2: Macronutrient Distribution (Donut Chart)
        with col2:
            st.subheader("🥗 Macronutrient Distribution")
            macros = nutrition_summary
            if macros and sum(macros.values()) > 0:
                fig = go.Figure(data=[go.Pie(
                    labels=['Protein', 'Carbohydrates', 'Fat', 'Fiber'],
                    values=[macros.get('protein', 0), macros.get('carbohydrates', 0), 
                           macros.get('fat', 0), macros.get('fiber', 0)],
                    hole=0.4,
                    marker=dict(colors=['#3B82F6', '#10B981', '#F59E0B', '#8B5CF6'])
                )])
                
                fig.update_layout(
                    title="Daily Macronutrients",
                    height=300,
                    margin=dict(l=20, r=20, t=40, b=20),
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("<div class='page-divider'></div>", unsafe_allow_html=True)

    components.section_header("🚫", "Foods to Avoid")
    avoid = recommendations.get("foods_to_avoid", [])
    if avoid:
        if isinstance(avoid[0], dict):
            avoid_df = pd.DataFrame(avoid)
            avoid_df.columns = [c.title() for c in avoid_df.columns]
        else:
            avoid_df = pd.DataFrame({"Food": avoid})
        st.dataframe(avoid_df, use_container_width=True, hide_index=True)
    else:
        components.status_banner("✅", "No Restrictions",
            "No specific foods to avoid based on your current health profile.", "ok")

    macro_data = recommendations.get("macronutrients") or recommendations.get("macros")
    if macro_data:
        components.section_header("🥗", "Macronutrient Distribution")
        if isinstance(macro_data, dict):
            macro_df = pd.DataFrame([
                {"macro": k.replace("_g", "").capitalize(), "grams": v}
                for k, v in macro_data.items() if isinstance(v, (int, float))
            ])
        elif isinstance(macro_data, pd.DataFrame):
            macro_df = macro_data
        else:
            macro_df = pd.DataFrame()
        if not macro_df.empty:
            components.chart_macronutrient(macro_df)

    components.section_header("💡", "Daily Nutrition Tips")
    tips = recommendations.get("nutrition_tips", [])
    if tips:
        components.tip_list(tips)
    else:
        st.info("No nutrition tips available for your current profile.")

    st.markdown("<div class='page-divider'></div>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════
    #  SECTION — Export Report
    # ════════════════════════════════════════════════════════════════════
    components.section_header("📥", "Export Your Health Report")
    report_data = st.session_state.get("report_data")
    if report_data:
        d1, d2, d3 = st.columns(3)
        with d1:
            components.download_button(report_data, filename="health_report", format="json")
        with d2:
            try:
                components.download_button(report_data, filename="health_report", format="pdf")
            except Exception as e:
                st.error(f"PDF download error: {e}")
        with d3:
            st.markdown(
                "<div class='report-saved-note'>💾&nbsp;Report auto-saved to local database.</div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("Run an analysis first to enable report export.")

components.page_footer()
