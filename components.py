"""components.py
Premium Blue & White UI component library — Smart Health Dashboard v4.0

All public functions (signatures unchanged for full backward compatibility):
    load_css()
    hero_banner(title, subtitle, username)
    section_header(icon, title)
    metric_card(label, value, icon, color)
    profile_card(icon, label, value)
    badge(label, color, variant)
    status_banner(icon, title, body, level)
    prediction_card(icon, name, pred_class, risk, level,
                    final_status, model_probability, confidence)
    tip_list(tips)
    meal_tag(label)
    welcome_screen()
    download_button(data, filename, format)
    sidebar_user_chip(display_name)
    form_progress_steps()
    dashboard_stats_strip(bmi, bmi_cat, tdee, diseases)
    page_footer()
    chart_bmi_gauge(bmi)
    chart_disease_risk(score, title)
    chart_macronutrient(df)
    chart_calorie_breakdown(df)
    chart_xai_feature_importance(feature_rows, disease_name)
    xai_explanation_panel(xai_result, disease_name)
    food_item_card(food_name, calories, protein, carbs, fat, fiber,
                   meal_type, food_index, show_swap)
"""

import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ════════════════════════════════════════════════════════════════════
#  CSS LOADER
# ════════════════════════════════════════════════════════════════════

def load_css():
    """Inject assets/style.css into the Streamlit page."""
    try:
        with open("assets/style.css", "r", encoding="utf-8") as fh:
            st.markdown(f"<style>{fh.read()}</style>", unsafe_allow_html=True)
    except Exception as exc:
        st.warning(f"Could not load stylesheet: {exc}")


# ════════════════════════════════════════════════════════════════════
#  HERO BANNER
# ════════════════════════════════════════════════════════════════════

def hero_banner(
    title: str = "Smart Health Dashboard",
    subtitle: str = "AI-Powered Personalised Nutrition & Disease Risk Analysis",
    username: str = "",
):
    """Full-width gradient header banner."""
    if username:
        _initial = username.strip()[0].upper() if username.strip() else "U"
        chip_html = (
            f"<div class='hero-profile-chip'>"
            f"  <div class='hpc-avatar'>{_initial}</div>"
            f"  <span class='hpc-name'>{username}</span>"
            f"</div>"
        )
    else:
        chip_html = ""

    st.markdown(
        f"<div class='hero-banner'>"
        f"  <div class='hero-ring'></div>"
        f"  <div class='hero-left'>"
        f"    <p class='hero-title'>&#129338;&nbsp;{title}</p>"
        f"    <p class='hero-subtitle'>{subtitle}</p>"
        f"  </div>"
        f"  {chip_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════
#  SECTION HEADER
# ════════════════════════════════════════════════════════════════════

def section_header(icon: str, title: str):
    """Underlined section divider with icon and title."""
    st.markdown(
        f"<div class='section-header'>"
        f"  <span class='sh-icon'>{icon}</span>"
        f"  <h3>{title}</h3>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════
#  METRIC CARD
# ════════════════════════════════════════════════════════════════════

def metric_card(label: str, value: str, icon: str = "", color: str = "#1D4ED8"):
    """Centred KPI card: icon · label · big value."""
    _icon_class = {
        "#1D4ED8": "icon-blue",
        "#2563EB": "icon-blue",
        "#047857": "icon-green",
        "#16A34A": "icon-green",
        "#B45309": "icon-amber",
        "#D97706": "icon-amber",
        "#7C3AED": "icon-purple",
        "#0369A1": "icon-sky",
        "#0284C7": "icon-sky",
        "#B91C1C": "icon-red",
        "#DC2626": "icon-red",
    }.get(color, "icon-blue")

    st.markdown(
        f"<div class='card animate-in'>"
        f"  <div class='metric-icon {_icon_class}'>{icon}</div>"
        f"  <div class='metric-label'>{label}</div>"
        f"  <div class='metric-value' style='color:{color};'>{value}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════
#  PROFILE ROW CARD
# ════════════════════════════════════════════════════════════════════

def profile_card(icon: str, label: str, value: str):
    """Horizontal card for Patient Profile."""
    st.markdown(
        f"<div class='card card-row'>"
        f"  <span class='card-row-icon'>{icon}</span>"
        f"  <span class='profile-label'>{label}</span>"
        f"  <span class='profile-value'>{value}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════
#  BADGE
# ════════════════════════════════════════════════════════════════════

def badge(label: str, color: str = "#2563EB", variant: str = ""):
    """Pill-shaped coloured badge."""
    if variant:
        html = f"<span class='badge badge-{variant}'>{label}</span>"
    else:
        html = f"<span class='badge' style='background:{color};'>{label}</span>"
    st.markdown(html, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
#  STATUS / ALERT BANNER
# ════════════════════════════════════════════════════════════════════

def status_banner(icon: str, title: str, body: str, level: str = "info"):
    """Coloured alert banner. level: 'ok' | 'warning' | 'danger' | 'info'"""
    cls = {
        "ok":      "status-ok",
        "warning": "status-warning",
        "danger":  "status-danger",
        "info":    "status-info",
    }.get(level, "status-info")

    st.markdown(
        f"<div class='status-banner {cls}'>"
        f"  <span class='sb-icon'>{icon}</span>"
        f"  <div class='sb-body'>"
        f"    <strong class='sb-title'>{title}</strong>"
        f"    <span class='sb-text'>&nbsp;{body}</span>"
        f"  </div>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════
#  PREDICTION CARD
# ════════════════════════════════════════════════════════════════════

def prediction_card(
    icon: str,
    name: str,
    pred_class: str,
    risk: str = "Low",
    level: str = "low",
    final_status: str = "",
    model_probability: "float | None" = None,
    confidence: "float | None" = None,
):
    """Disease prediction card with stripe, icon, result and risk badge."""
    # Stripe gradient per risk level
    stripe = {
        "high":   "background:linear-gradient(90deg,#DC2626,#B91C1C);",
        "medium": "background:linear-gradient(90deg,#F59E0B,#D97706);",
        "low":    "background:linear-gradient(90deg,#059669,#047857);",
    }.get(level, "background:linear-gradient(90deg,#059669,#047857);")

    icon_bg = {
        "high":   "background:#FEF2F2;border:2px solid #FECACA;",
        "medium": "background:#FFFBEB;border:2px solid #FDE68A;",
        "low":    "background:#ECFDF5;border:2px solid #A7F3D0;",
    }.get(level, "background:#ECFDF5;border:2px solid #A7F3D0;")

    badge_cls = {
        "high":   "pred-risk-high",
        "medium": "pred-risk-medium",
        "low":    "pred-risk-low",
    }.get(level, "pred-risk-low")

    dot = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(level, "🟢")

    # Badge text
    _no_suffix = {"Screening Flag", "Unavailable", "Borderline", "Model Flag", "Normal"}
    badge_text = risk if (risk in _no_suffix or risk.endswith("Risk")) else f"{risk} Risk"

    # Model probability display
    _prob = model_probability if model_probability is not None else confidence
    prob_html = ""
    if _prob is not None:
        prob_html = (
            f"<div class='pred-confidence'>"
            f"Model Probability:&nbsp;<strong>{_prob:.1f}%</strong>"
            f"</div>"
        )

    # Final-status sub-label
    status_html = ""
    if final_status and final_status not in ("", "Unknown", risk):
        _status_styles = {
            "High Risk":     ("background:#FEF2F2;color:#991B1B;border:1px solid #FECACA;", "🔴"),
            "Moderate Risk": ("background:#FFFBEB;color:#92400E;border:1px solid #FDE68A;", "🟡"),
            "Model Flag":    ("background:#EFF6FF;color:#1E40AF;border:1px solid #BFDBFE;", "🔵"),
            "Low Risk":      ("background:#ECFDF5;color:#065F46;border:1px solid #A7F3D0;", "🟢"),
            "Borderline":    ("background:#FFFBEB;color:#92400E;border:1px solid #FDE68A;", "🟡"),
            "Unavailable":   ("background:#F8FAFC;color:#64748B;border:1px solid #E2E8F0;", "⚪"),
        }
        _sc, _sdot = _status_styles.get(
            final_status,
            ("background:#EFF6FF;color:#1E40AF;border:1px solid #BFDBFE;", "ℹ️"),
        )
        status_html = (
            f"<div style='margin-top:0.45rem;padding:0.22rem 0.65rem;"
            f"border-radius:999px;font-size:0.70rem;font-weight:700;"
            f"display:inline-block;{_sc}'>"
            f"{_sdot}&nbsp;{final_status}"
            f"</div>"
        )

    st.markdown(
        f"<div class='pred-card'>"
        f"  <div class='pred-card-stripe' style='{stripe}'></div>"
        f"  <div class='pred-icon-wrap' style='{icon_bg}'>{icon}</div>"
        f"  <div class='pred-name'>{name}</div>"
        f"  <div class='pred-result'>{pred_class}</div>"
        f"  <span class='pred-risk-badge {badge_cls}'>{dot}&nbsp;{badge_text}</span>"
        f"  {status_html}"
        f"  {prob_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════
#  NUTRITION TIPS LIST
# ════════════════════════════════════════════════════════════════════

def tip_list(tips: list):
    """Numbered, styled nutrition tip list."""
    if not tips:
        return
    items = "".join(
        f"<div class='tip-item'>"
        f"  <span class='tip-num'>{i}</span>"
        f"  <span class='tip-text'>{tip}</span>"
        f"</div>"
        for i, tip in enumerate(tips, start=1)
    )
    st.markdown(items, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
#  MEAL TAG
# ════════════════════════════════════════════════════════════════════

def meal_tag(label: str):
    """Gradient pill tag for meal sections."""
    icons = {
        "breakfast":     "🌅",
        "mid-morning":   "☕",
        "morning":       "☕",
        "lunch":         "☀️",
        "afternoon":     "🍱",
        "evening snack": "🍎",
        "evening":       "🍎",
        "snack":         "🍎",
        "snacks":        "🍎",
        "dinner":        "🌙",
    }
    ico = icons.get(label.lower(), "🍽️")
    st.markdown(
        f"<div class='meal-tag'>{ico}&nbsp;{label}</div>",
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════
#  WELCOME SCREEN
# ════════════════════════════════════════════════════════════════════

def welcome_screen():
    """Landing screen shown before the first analysis."""
    st.markdown(
        "<div class='welcome-banner animate-in'>"
        "  <span class='welcome-icon'>&#128075;</span>"
        "  <div class='welcome-content'>"
        "    <p class='welcome-title'>Welcome to Smart Health Dashboard</p>"
        "    <p class='welcome-body'>"
        "      Complete the health assessment form, then press "
        "      <strong>Analyse My Health</strong> to receive your "
        "      personalised disease risk assessment, nutrition plan, and meal recommendations."
        "    </p>"
        "  </div>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='how-it-works'>"
        "  <div class='hiw-step'>"
        "    <span class='hiw-num'>1</span>"
        "    <div class='hiw-icon'>&#128221;</div>"
        "    <p class='hiw-title'>Enter Details</p>"
        "    <p class='hiw-body'>Fill in personal info, medical parameters &amp; health goals</p>"
        "  </div>"
        "  <div class='hiw-step'>"
        "    <span class='hiw-num'>2</span>"
        "    <div class='hiw-icon'>&#129504;</div>"
        "    <p class='hiw-title'>Run Analysis</p>"
        "    <p class='hiw-body'>AI models evaluate disease risk &amp; calculate your nutrition needs</p>"
        "  </div>"
        "  <div class='hiw-step'>"
        "    <span class='hiw-num'>3</span>"
        "    <div class='hiw-icon'>&#128202;</div>"
        "    <p class='hiw-title'>View Results</p>"
        "    <p class='hiw-body'>Explore predictions, charts, meal plans &amp; export your report</p>"
        "  </div>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="large")
    tiles = [
        ("🧬", "Disease Risk Analysis",
         "AI-powered screening for Diabetes, Kidney Disease &amp; Obesity"),
        ("🍽️", "Personalised Meal Plans",
         "Tailored daily nutrition plans matched to your health profile"),
        ("📊", "Health Metrics &amp; Charts",
         "BMI · BMR · TDEE · Calorie breakdown · Macronutrients"),
    ]
    for col, (ico, ttl, body) in zip([c1, c2, c3], tiles):
        with col:
            st.markdown(
                f"<div class='feature-tile animate-in'>"
                f"  <div class='feature-tile-icon'>{ico}</div>"
                f"  <p class='feature-tile-title'>{ttl}</p>"
                f"  <p class='feature-tile-body'>{body}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )


# ════════════════════════════════════════════════════════════════════
#  SIDEBAR USER CHIP
# ════════════════════════════════════════════════════════════════════

def sidebar_user_chip(display_name: str):
    """Styled user identity chip for the sidebar."""
    initial = display_name.strip()[0].upper() if display_name.strip() else "U"
    st.markdown(
        f"<div class='user-chip'>"
        f"  <div class='user-chip-avatar'>{initial}</div>"
        f"  <div class='user-chip-info'>"
        f"    <p class='user-chip-label'>Signed in as</p>"
        f"    <p class='user-chip-name'>{display_name}</p>"
        f"  </div>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════
#  FORM PROGRESS STEPPER (legacy — kept for compat)
# ════════════════════════════════════════════════════════════════════

def form_progress_steps():
    """Visual 3-step guide above the health assessment form."""
    st.markdown(
        "<div class='form-steps animate-in'>"
        "  <div class='form-step active'>"
        "    <span class='form-step-num'>1</span>"
        "    <span class='form-step-label'>Personal Info</span>"
        "  </div>"
        "  <div class='form-step-connector'></div>"
        "  <div class='form-step active'>"
        "    <span class='form-step-num'>2</span>"
        "    <span class='form-step-label'>Medical Data</span>"
        "  </div>"
        "  <div class='form-step-connector'></div>"
        "  <div class='form-step active'>"
        "    <span class='form-step-num'>3</span>"
        "    <span class='form-step-label'>Goals &amp; Settings</span>"
        "  </div>"
        "</div>",
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════
#  DASHBOARD QUICK-STATS STRIP
# ════════════════════════════════════════════════════════════════════

def dashboard_stats_strip(
    bmi: float,
    bmi_cat: str,
    tdee: float,
    diseases: list,
):
    """Four-pill horizontal summary bar above the dashboard results."""
    _risk_text = "Healthy" if diseases == ["Normal"] else ", ".join(diseases)
    _risk_icon = "✅" if diseases == ["Normal"] else "⚠️"
    _risk_bg   = "#ECFDF5" if diseases == ["Normal"] else "#FFFBEB"

    _bmi_color = (
        "#047857" if "normal"     in bmi_cat.lower() else
        "#B45309" if "overweight" in bmi_cat.lower() else
        "#B91C1C" if "obese"      in bmi_cat.lower() else
        "#0369A1"
    )

    st.markdown(
        f"<div class='stats-strip animate-in'>"

        # BMI pill
        f"<div class='stat-pill'>"
        f"  <div class='stat-pill-icon' style='background:#EBF0FF;'>⚖️</div>"
        f"  <div class='stat-pill-body'>"
        f"    <p class='stat-pill-label'>BMI</p>"
        f"    <p class='stat-pill-value' style='color:{_bmi_color};'>{bmi:.1f}</p>"
        f"  </div>"
        f"</div>"

        # Category pill
        f"<div class='stat-pill'>"
        f"  <div class='stat-pill-icon' style='background:#EBF0FF;'>🏷️</div>"
        f"  <div class='stat-pill-body'>"
        f"    <p class='stat-pill-label'>Category</p>"
        f"    <p class='stat-pill-value'>{bmi_cat}</p>"
        f"  </div>"
        f"</div>"

        # Calories pill
        f"<div class='stat-pill'>"
        f"  <div class='stat-pill-icon' style='background:#FFFBEB;'>🔥</div>"
        f"  <div class='stat-pill-body'>"
        f"    <p class='stat-pill-label'>Daily Calories</p>"
        f"    <p class='stat-pill-value'>{tdee:.0f} kcal</p>"
        f"  </div>"
        f"</div>"

        # Risk pill
        f"<div class='stat-pill'>"
        f"  <div class='stat-pill-icon' style='background:{_risk_bg};'>{_risk_icon}</div>"
        f"  <div class='stat-pill-body'>"
        f"    <p class='stat-pill-label'>Risk Status</p>"
        f"    <p class='stat-pill-value'>{_risk_text}</p>"
        f"  </div>"
        f"</div>"

        f"</div>",
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════
#  PAGE FOOTER
# ════════════════════════════════════════════════════════════════════

def page_footer():
    """Minimal branded footer."""
    st.markdown(
        "<div class='page-footer'>"
        "  <p class='page-footer-text'>"
        "    Smart Health Dashboard &mdash; AI-Powered Nutrition &amp; Disease Risk Analysis"
        "  </p>"
        "  <span class='page-footer-badge'>&#129338; Healthcare AI</span>"
        "</div>",
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════
#  DOWNLOAD BUTTON
# ════════════════════════════════════════════════════════════════════

def download_button(data, filename: str = "report", format: str = "json"):
    """Render a styled download button for JSON, CSV, or PDF."""

    def _serialize(obj):
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict("records")
        if isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_serialize(i) for i in obj]
        return obj

    # ── JSON ──────────────────────────────────────────────────────────
    if format == "json":
        payload = json.dumps(_serialize(data), indent=2)
        st.download_button(
            label="📄 Download JSON Report",
            data=payload,
            file_name=f"{filename}.json",
            mime="application/json",
            use_container_width=True,
        )

    # ── CSV ───────────────────────────────────────────────────────────
    elif format == "csv":
        csv_str = (
            data.to_csv(index=False)
            if isinstance(data, pd.DataFrame)
            else pd.DataFrame(data).to_csv(index=False)
        )
        st.download_button(
            label="📊 Download CSV Report",
            data=csv_str,
            file_name=f"{filename}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # ── PDF ───────────────────────────────────────────────────────────
    elif format == "pdf":
        from fpdf import FPDF

        def _safe(text: str) -> str:
            return (
                str(text)
                .replace("\u2014", "-").replace("\u2013", "-")
                .replace("\u2018", "'").replace("\u2019", "'")
                .replace("\u201c", '"').replace("\u201d", '"')
                .replace("\u2022", "*").replace("\u2026", "...")
                .encode("latin-1", errors="replace").decode("latin-1")
            )

        def _flatten(obj, max_list: int = 8):
            if isinstance(obj, pd.DataFrame):
                rows = obj.head(max_list).to_dict("records")
                return [_flatten(r) for r in rows]
            if isinstance(obj, dict):
                return {str(k): _flatten(v, max_list) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                items = [_flatten(i, max_list) for i in list(obj)[:max_list]]
                if len(obj) > max_list:
                    items.append(f"... and {len(obj) - max_list} more")
                return items
            s = str(obj)
            return (s[:100] + "...") if len(s) > 100 else s

        def _write(pdf_obj, d: dict, indent: int = 0):
            indent = min(indent, 3)
            lm = 10 + indent * 5
            for k, v in d.items():
                if pdf_obj.get_y() > pdf_obj.h - pdf_obj.b_margin - 15:
                    pdf_obj.add_page()
                pdf_obj.set_left_margin(lm)
                pdf_obj.set_x(lm)
                if isinstance(v, dict):
                    pdf_obj.set_font("Arial", "B", 11)
                    pdf_obj.set_fill_color(239, 246, 255)
                    pdf_obj.multi_cell(0, 8, _safe(str(k)), fill=(indent == 0))
                    pdf_obj.set_font("Arial", size=10)
                    _write(pdf_obj, v, indent + 1)
                elif isinstance(v, list):
                    pdf_obj.set_font("Arial", "B", 10)
                    pdf_obj.multi_cell(0, 7, _safe(f"{k}:"))
                    pdf_obj.set_font("Arial", size=10)
                    for item in v:
                        if pdf_obj.get_y() > pdf_obj.h - pdf_obj.b_margin - 15:
                            pdf_obj.add_page()
                        pdf_obj.set_left_margin(lm + 5)
                        pdf_obj.set_x(lm + 5)
                        if isinstance(item, dict):
                            _write(pdf_obj, item, indent + 2)
                        else:
                            pdf_obj.multi_cell(0, 6, _safe(f"- {item}"))
                    pdf_obj.set_left_margin(lm)
                else:
                    pdf_obj.set_font("Arial", size=10)
                    pdf_obj.multi_cell(0, 7, _safe(f"{k}: {v}"))
            pdf_obj.set_left_margin(10)

        pdf = FPDF()
        pdf.set_margins(left=10, top=15, right=10)
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        pdf.set_font("Arial", "B", 18)
        pdf.cell(0, 12, _safe("Smart Health Dashboard — Health Report"), ln=True, align="C")
        pdf.set_font("Arial", size=10)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 8, _safe("AI-Powered Personalised Nutrition & Disease Risk Analysis"), ln=True, align="C")
        pdf.set_text_color(0, 0, 0)
        pdf.ln(4)
        pdf.set_draw_color(226, 232, 240)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(6)

        safe_data = _flatten(data) if isinstance(data, dict) else _safe(str(data))
        if isinstance(safe_data, dict):
            _write(pdf, safe_data)
        else:
            pdf.set_font("Arial", size=10)
            pdf.multi_cell(0, 7, safe_data)

        try:
            pdf_output = pdf.output(dest='S')
            pdf_bytes = pdf_output.encode('latin-1', errors='replace')
        except Exception:
            pdf_bytes = bytes(pdf.output())

        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_bytes,
            file_name=f"{filename}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        st.error(f"Unsupported format: {format}")


# ════════════════════════════════════════════════════════════════════
#  PLOTLY HELPERS
# ════════════════════════════════════════════════════════════════════

# Blue-based chart palette — consistent across all charts
_BLUE_PALETTE = [
    "#1A56DB", "#0EA5E9", "#38BDF8", "#7DD3FC", "#BAE6FD",
    "#1E3A8A", "#1D4ED8", "#3B82F6", "#60A5FA", "#93C5FD",
]


def _theme(fig, height: int = 260):
    """Apply clean transparent background, Inter font & professional text color to a Plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Plus Jakarta Sans, sans-serif",
                  size=13, color="#0B1120"),
        margin=dict(t=36, b=28, l=16, r=16),
        height=height,
        title_font=dict(size=13, color="#334155"),
        legend_title_font=dict(size=12, color="#334155"),
    )
    return fig


def chart_bmi_gauge(bmi: float):
    """Gauge chart visualising BMI against WHO thresholds with WCAG-AA colors."""
    if bmi < 18.5:
        color = "#0369A1"
    elif bmi < 25:
        color = "#047857"
    elif bmi < 30:
        color = "#B45309"
    else:
        color = "#B91C1C"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=bmi,
        delta={
            "reference": 22,
            "valueformat": ".1f",
            "increasing": {"color": "#B91C1C"},
            "decreasing": {"color": "#047857"},
        },
        number={
            "valueformat": ".1f",
            "font": {"size": 34, "color": "#0B1120", "family": "Inter, sans-serif"},
        },
        title={
            "text": "Body Mass Index (BMI)",
            "font": {"size": 13, "color": "#334155", "family": "Inter, sans-serif"},
        },
        gauge={
            "axis": {
                "range": [10, 40],
                "tickwidth": 1.5,
                "tickcolor": "#CBD5F0",
                "tickvals": [10, 18.5, 25, 30, 40],
                "ticktext": ["10", "18.5", "25", "30", "40"],
                "tickfont": {"size": 11, "color": "#475569"},
            },
            "bar": {"color": color, "thickness": 0.24},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [10,   18.5], "color": "rgba(3,105,161,.12)"},
                {"range": [18.5, 25],   "color": "rgba(4,120,87,.12)"},
                {"range": [25,   30],   "color": "rgba(180,83,9,.12)"},
                {"range": [30,   40],   "color": "rgba(185,28,28,.12)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4.5},
                "thickness": 0.78,
                "value": bmi,
            },
        },
    ))
    fig = _theme(fig, height=270)
    st.plotly_chart(fig, use_container_width=True)


def chart_disease_risk(score: float, title: str = "Risk Score"):
    """0-100% gauge for generic disease risk with WCAG-AA colors."""
    pct = score * 100
    color = "#047857" if pct < 33 else ("#B45309" if pct < 66 else "#B91C1C")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={
            "suffix": "%",
            "valueformat": ".0f",
            "font": {"size": 30, "color": "#0B1120", "family": "Inter, sans-serif"},
        },
        title={"text": title, "font": {"size": 13, "color": "#334155", "family": "Inter, sans-serif"}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1.5,
                "tickcolor": "#CBD5F0",
                "tickfont": {"size": 11, "color": "#475569"},
            },
            "bar": {"color": color, "thickness": 0.24},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  33], "color": "rgba(4,120,87,.12)"},
                {"range": [33, 66], "color": "rgba(180,83,9,.12)"},
                {"range": [66,100], "color": "rgba(185,28,28,.12)"},
            ],
        },
    ))
    fig = _theme(fig, height=240)
    st.plotly_chart(fig, use_container_width=True)


def chart_macronutrient(df: pd.DataFrame):
    """Donut chart for macronutrient split. Columns: ['macro','grams']."""
    fig = px.pie(
        df, names="macro", values="grams", hole=0.52,
        color_discrete_sequence=_BLUE_PALETTE,
    )
    fig.update_traces(
        textposition="outside",
        textinfo="label+percent",
        textfont_size=11,
        pull=[0.03] * len(df),
        marker=dict(line=dict(color="#FFFFFF", width=2)),
    )
    fig = _theme(fig, height=315)
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",
            y=-0.20,
            font=dict(size=11),
            itemsizing="constant",
        ),
    )
    st.plotly_chart(fig, use_container_width=True)


def chart_calorie_breakdown(df: pd.DataFrame):
    """Horizontal bar chart. Columns: ['meal','calories']."""
    fig = px.bar(
        df, x="calories", y="meal", orientation="h",
        text="calories", color="meal",
        color_discrete_sequence=_BLUE_PALETTE,
    )
    fig.update_traces(
        texttemplate="%{text:.0f} kcal",
        textposition="outside",
        textfont_size=11,
        marker_line_width=0,
        cliponaxis=False,
    )
    fig.update_layout(
        xaxis=dict(
            title="Calories (kcal)",
            showgrid=True,
            gridcolor="rgba(203,213,225,.35)",
            zeroline=False,
        ),
        yaxis=dict(title="", categoryorder="total ascending"),
        showlegend=False,
    )
    fig = _theme(fig, height=260)
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
#  XAI — FEATURE IMPORTANCE CHART
# ════════════════════════════════════════════════════════════════════

def chart_xai_feature_importance(feature_rows: list, disease_name: str = "") -> None:
    """Full-width horizontal Plotly bar chart of feature importances."""
    if not feature_rows:
        return

    _impact_colour = {
        "High":   "#DC2626",
        "Medium": "#F59E0B",
        "Low":    "#059669",
    }

    rows_display = list(reversed(feature_rows))
    labels   = [r.feature for r in rows_display]
    scores   = [round(r.importance * 100, 2) for r in rows_display]
    colours  = [_impact_colour.get(r.impact, "#1A56DB") for r in rows_display]
    hover    = [
        f"<b>{r.feature}</b><br>Value: {r.value}<br>"
        f"Status: {r.direction}<br>Impact: {r.impact}"
        for r in rows_display
    ]
    out_text = [f"{r.value}" for r in rows_display]

    fig = go.Figure(go.Bar(
        x=scores,
        y=labels,
        orientation="h",
        marker_color=colours,
        marker_line_width=0,
        hovertemplate=hover,
        hoverinfo="text",
        text=out_text,
        textposition="outside",
        textfont=dict(size=11, color="#475569"),
        cliponaxis=False,
    ))

    max_score = max(scores) if scores else 1
    fig.update_layout(
        xaxis=dict(
            title="Contribution Score",
            range=[0, max_score * 1.55],
            showgrid=True,
            gridcolor="rgba(203,213,225,.38)",
            zeroline=False,
            tickfont=dict(size=10),
        ),
        yaxis=dict(tickfont=dict(size=11), automargin=True),
        showlegend=False,
        bargap=0.28,
        margin=dict(l=8, r=12, t=8, b=28),
    )
    fig = _theme(fig, height=max(155, len(feature_rows) * 44))
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
#  XAI — EXPLANATION PANEL
# ════════════════════════════════════════════════════════════════════

def xai_explanation_panel(xai_result, disease_name: str = "") -> None:
    """Render the XAI explanation panel for one disease — full-width, no expander.

    Layout:
      1. Method note
      2. Impact legend
      3. HTML feature-bar rows (pure CSS)
      4. Plotly horizontal bar chart
      5. Patient-friendly summary box
      6. Disclaimer
    """
    if not xai_result.available:
        st.markdown(
            f"<div class='xai-method-note'>ℹ️ Explanation unavailable for "
            f"<strong>{disease_name}</strong>. {xai_result.error or ''}</div>",
            unsafe_allow_html=True,
        )
        return

    rows = xai_result.feature_rows
    if not rows:
        st.info("No feature data available.")
        return

    # 1. Method note
    st.markdown(
        f"<div class='xai-method-note'>"
        f"<strong>Method:</strong> {xai_result.method}. "
        "Each score = model feature weight × deviation from the clinical reference range."
        "</div>",
        unsafe_allow_html=True,
    )

    # 2. Legend
    st.markdown(
        "<div class='xai-legend'>"
        "  <span class='xai-legend-item'><span class='xai-legend-dot high'></span>High impact</span>"
        "  <span class='xai-legend-item'><span class='xai-legend-dot medium'></span>Medium impact</span>"
        "  <span class='xai-legend-item'><span class='xai-legend-dot low'></span>Low impact</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    # 3. HTML feature rows
    max_score = max(r.importance for r in rows) or 1.0
    _dir_class = {
        "↑ Above normal": "above",
        "↓ Below normal": "below",
        "✓ Normal range": "normal",
    }
    _dir_label = {
        "↑ Above normal": "↑ Above",
        "↓ Below normal": "↓ Below",
        "✓ Normal range": "✓ Normal",
    }

    row_html = ""
    for r in rows:
        pct     = round(r.importance / max_score * 100, 1)
        imp_cls = r.impact.lower()
        dir_cls = _dir_class.get(r.direction, "normal")
        dir_lbl = _dir_label.get(r.direction, r.direction)
        row_html += (
            f"<div class='xai-feature-row'>"
            f"  <span class='xai-feat-name'>{r.feature}</span>"
            f"  <div class='xai-bar-wrap'>"
            f"    <div class='xai-bar-fill {imp_cls}' style='width:{pct}%'></div>"
            f"  </div>"
            f"  <span class='xai-value-chip'>{r.value}</span>"
            f"  <span class='xai-dir {dir_cls}'>{dir_lbl}</span>"
            f"</div>"
        )

    st.markdown(
        f"<div class='xai-panel'>{row_html}</div>",
        unsafe_allow_html=True,
    )

    # 4. Plotly chart
    chart_xai_feature_importance(rows, disease_name)

    # 5. Summary
    if xai_result.summary:
        st.markdown(
            f"<div class='xai-summary'>💡 {xai_result.summary}</div>",
            unsafe_allow_html=True,
        )

    # 6. Disclaimer
    st.markdown(
        "<div class='xai-disclaimer'>"
        "⚠️ This is a screening result only. It does not constitute a medical "
        "diagnosis. Please consult a qualified healthcare professional."
        "</div>",
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════
#  FOOD ITEM CARD
# ════════════════════════════════════════════════════════════════════

def food_item_card(
    food_name: str,
    calories: float,
    protein: float = None,
    carbs: float = None,
    fat: float = None,
    fiber: float = None,
    meal_type: str = "",
    food_index: int = 0,
    show_swap: bool = False,
) -> bool:
    """Render a food item card with nutrition info. Returns True if swap clicked.

    Uses a clean HTML card for the name/calories row to avoid overlap,
    then Streamlit columns only for the swap button.
    """
    # Build nutrition chips HTML
    chips = []
    if protein is not None and protein > 0:
        chips.append(f"<span class='nutrient-item'>🥩&nbsp;Protein: {protein:.1f}g</span>")
    if carbs is not None and carbs > 0:
        chips.append(f"<span class='nutrient-item'>🍞&nbsp;Carbs: {carbs:.1f}g</span>")
    if fat is not None and fat > 0:
        chips.append(f"<span class='nutrient-item'>🥑&nbsp;Fat: {fat:.1f}g</span>")
    if fiber is not None and fiber > 0:
        chips.append(f"<span class='nutrient-item'>🌾&nbsp;Fibre: {fiber:.1f}g</span>")
    nutrition_html = (
        f"<div class='food-item-nutrition'>{''.join(chips)}</div>"
        if chips else ""
    )

    if show_swap:
        # Use columns only when there's a swap button
        col_card, col_btn = st.columns([5, 1], gap="small")
        with col_card:
            st.markdown(
                f"<div class='food-item-card'>"
                f"  <div class='food-item-header'>"
                f"    <span class='food-item-name'>{food_name}</span>"
                f"    <span class='food-item-calories'>{calories:.0f} kcal</span>"
                f"  </div>"
                f"  {nutrition_html}"
                f"</div>",
                unsafe_allow_html=True,
            )
        with col_btn:
            # Vertical padding to align button with card
            st.markdown("<div style='margin-top:0.55rem'></div>", unsafe_allow_html=True)
            button_key = (
                f"swap_{meal_type}_{food_index}_{food_name.replace(' ', '_')[:20]}"
            )
            if st.button("🔄", key=button_key, help=f"Find alternatives for {food_name}"):
                return True
    else:
        # Pure HTML card — no columns, no overlap risk
        st.markdown(
            f"<div class='food-item-card'>"
            f"  <div class='food-item-header'>"
            f"    <span class='food-item-name'>{food_name}</span>"
            f"    <span class='food-item-calories'>{calories:.0f} kcal</span>"
            f"  </div>"
            f"  {nutrition_html}"
            f"</div>",
            unsafe_allow_html=True,
        )

    return False


# ════════════════════════════════════════════════════════════════════
#  SECTION DIVIDER HELPER (convenience)
# ════════════════════════════════════════════════════════════════════

def section_divider():
    """Renders a subtle full-width horizontal rule between sections."""
    st.markdown("<div class='page-divider'></div>", unsafe_allow_html=True)
