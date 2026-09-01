"""components.py
Premium UI component library for Smart Health Dashboard.

All public functions:
    load_css()
    hero_banner(title, subtitle, username)
    section_header(icon, title)
    metric_card(label, value, icon, color)
    profile_card(icon, label, value)
    badge(label, color, variant)
    status_banner(icon, title, body, level)
    prediction_card(icon, name, pred_class, risk, level)
    tip_list(tips)
    meal_tag(label)
    welcome_screen()
    download_button(data, filename, format)

    chart_bmi_gauge(bmi)
    chart_disease_risk(score, title)
    chart_macronutrient(df)
    chart_calorie_breakdown(df)
"""

import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ════════════════════════════════════════════════════════════════════
#  CSS
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
    """Gradient header banner shown at the top of every page."""
    chip = (
        f"<span class='hero-user'>&#128100; {username}</span>"
        if username else ""
    )
    st.markdown(
        f"""
        <div class='hero-banner'>
            <div class='hero-left'>
                <p class='hero-title'>&#129338; {title}</p>
                <p class='hero-subtitle'>{subtitle}</p>
            </div>
            {chip}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════
#  SECTION HEADER
# ════════════════════════════════════════════════════════════════════

def section_header(icon: str, title: str):
    """Underlined section divider with icon and title."""
    st.markdown(
        f"<div class='section-header'>"
        f"<span class='sh-icon'>{icon}</span>"
        f"<h3>{title}</h3>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════
#  METRIC CARD
# ════════════════════════════════════════════════════════════════════

def metric_card(label: str, value: str, icon: str = "", color: str = "#2563EB"):
    """Centred KPI card: icon · label · big value."""
    st.markdown(
        f"""
        <div class='card'>
            <div class='metric-icon' style='color:{color};'>{icon}</div>
            <div class='metric-label'>{label}</div>
            <div class='metric-value' style='color:{color};'>{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════
#  PROFILE ROW CARD
# ════════════════════════════════════════════════════════════════════

def profile_card(icon: str, label: str, value: str):
    """Horizontal card for Patient Profile tab."""
    st.markdown(
        f"""
        <div class='card card-row'>
            <span class='card-row-icon'>{icon}</span>
            <span class='profile-label'>{label}</span>
            <span class='profile-value'>{value}</span>
        </div>
        """,
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
    """Coloured alert banner.

    Parameters
    ----------
    level : 'ok' | 'warning' | 'danger' | 'info'
    """
    cls = {
        "ok":      "status-ok",
        "warning": "status-warning",
        "danger":  "status-danger",
        "info":    "status-info",
    }.get(level, "status-info")

    st.markdown(
        f"""
        <div class='status-banner {cls}'>
            <span class='sb-icon'>{icon}</span>
            <div class='sb-body'>
                <strong class='sb-title'>{title}</strong>
                <span class='sb-text'>{body}</span>
            </div>
        </div>
        """,
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
    confidence: float = None,
):
    """Tall disease prediction card.

    Parameters
    ----------
    icon       : Emoji (e.g. '🩸')
    name       : Disease display name
    pred_class : Raw model output label
    risk       : 'High' | 'Moderate' | 'Low' | 'Normal'
    level      : 'high' | 'medium' | 'low'  — controls colour scheme
    confidence : Prediction confidence percentage (0-100)
    """
    stripe = {
        "high":   "background:linear-gradient(90deg,#EF4444,#F43F5E);",
        "medium": "background:linear-gradient(90deg,#F59E0B,#FBBF24);",
        "low":    "background:linear-gradient(90deg,#10B981,#06B6D4);",
    }.get(level, "background:linear-gradient(90deg,#10B981,#06B6D4);")

    icon_bg = {
        "high":   "background:#FEE2E2;",
        "medium": "background:#FEF3C7;",
        "low":    "background:#D1FAE5;",
    }.get(level, "background:#D1FAE5;")

    badge_cls = {
        "high":   "pred-risk-high",
        "medium": "pred-risk-medium",
        "low":    "pred-risk-low",
    }.get(level, "pred-risk-low")

    dot = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(level, "🟢")

    # Add confidence display if available
    confidence_html = ""
    if confidence is not None:
        confidence_html = f"<div class='pred-confidence'>Confidence: {confidence}%</div>"

    st.markdown(
        f"""
        <div class='pred-card'>
            <div class='pred-card-stripe' style='{stripe}'></div>
            <div class='pred-icon-wrap' style='{icon_bg}'>{icon}</div>
            <div class='pred-name'>{name}</div>
            <div class='pred-result'>{pred_class}</div>
            <span class='pred-risk-badge {badge_cls}'>{dot}&nbsp;{risk} Risk</span>
            {confidence_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════
#  NUTRITION TIPS LIST
# ════════════════════════════════════════════════════════════════════

def tip_list(tips: list):
    """Numbered, styled nutrition tip list."""
    items = "".join(
        f"<div class='tip-item'>"
        f"<span class='tip-num'>{i}</span>"
        f"<span class='tip-text'>{tip}</span>"
        f"</div>"
        for i, tip in enumerate(tips, start=1)
    )
    st.markdown(items, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
#  MEAL TAG
# ════════════════════════════════════════════════════════════════════

def meal_tag(label: str):
    """Gradient pill tag for meal sections (Breakfast, Lunch, etc.)."""
    icons = {
        "breakfast": "🌅", "lunch": "☀️", "dinner": "🌙",
        "snack": "🍎", "snacks": "🍎",
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
        """
        <div class='welcome-banner'>
            <span class='welcome-icon'>👋</span>
            <div>
                <p class='welcome-title'>Welcome to Smart Health Dashboard</p>
                <p class='welcome-body'>
                    Enter your patient details and medical parameters in the
                    left sidebar, then press
                    <strong>Analyse Health</strong> to receive your
                    personalised disease risk assessment and nutrition plan.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3, gap="large")
    tiles = [
        ("🧬", "Disease Risk Analysis",
         "AI-powered screening for Diabetes, Kidney Disease & Obesity"),
        ("🍽️", "Personalised Meal Plans",
         "Tailored daily nutrition plans matched to your health profile"),
        ("📊", "Health Metrics & Charts",
         "BMI · BMR · TDEE · Calorie breakdown · Macronutrients"),
    ]
    for col, (ico, ttl, body) in zip([c1, c2, c3], tiles):
        with col:
            st.markdown(
                f"""
                <div class='feature-tile'>
                    <span class='feature-tile-icon'>{ico}</span>
                    <p class='feature-tile-title'>{ttl}</p>
                    <p class='feature-tile-body'>{body}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ════════════════════════════════════════════════════════════════════
#  PLOTLY HELPERS
# ════════════════════════════════════════════════════════════════════

def _theme(fig, height: int = 260):
    """Apply transparent background and Inter font to a Plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12, color="#475569"),
        margin=dict(t=40, b=25, l=20, r=20),
        height=height,
    )
    return fig


def chart_bmi_gauge(bmi: float):
    """Gauge chart visualising BMI against WHO thresholds."""
    if bmi < 18.5:
        color = "#6366F1"
    elif bmi < 25:
        color = "#10B981"
    elif bmi < 30:
        color = "#F59E0B"
    else:
        color = "#EF4444"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=bmi,
        delta={"reference": 22, "valueformat": ".1f",
               "increasing": {"color": "#EF4444"},
               "decreasing": {"color": "#10B981"}},
        number={"valueformat": ".1f",
                "font": {"size": 34, "color": "#1E293B"}},
        title={"text": "Body Mass Index (BMI)",
               "font": {"size": 13, "color": "#64748B"}},
        gauge={
            "axis": {
                "range": [10, 40],
                "tickwidth": 1, "tickcolor": "#CBD5E1",
                "tickvals": [10, 18.5, 25, 30, 40],
                "ticktext": ["10", "18.5", "25", "30", "40"],
                "tickfont": {"size": 10},
            },
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
            "steps": [
                {"range": [10,   18.5], "color": "rgba(99,102,241,.10)"},
                {"range": [18.5, 25],   "color": "rgba(16,185,129,.10)"},
                {"range": [25,   30],   "color": "rgba(245,158,11,.10)"},
                {"range": [30,   40],   "color": "rgba(239,68,68,.10)"},
            ],
            "threshold": {"line": {"color": color, "width": 4},
                          "thickness": 0.8, "value": bmi},
        },
    ))
    fig = _theme(fig, height=270)
    st.plotly_chart(fig, use_container_width=True)


def chart_disease_risk(score: float, title: str = "Risk Score"):
    """0-100% gauge for generic disease risk."""
    pct = score * 100
    color = "#10B981" if pct < 33 else ("#F59E0B" if pct < 66 else "#EF4444")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "valueformat": ".0f",
                "font": {"size": 30, "color": "#1E293B"}},
        title={"text": title, "font": {"size": 13, "color": "#64748B"}},
        gauge={
            "axis": {"range": [0, 100],
                     "tickwidth": 1, "tickcolor": "#CBD5E1"},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
            "steps": [
                {"range": [0,  33], "color": "rgba(16,185,129,.10)"},
                {"range": [33, 66], "color": "rgba(245,158,11,.10)"},
                {"range": [66,100], "color": "rgba(239,68,68,.10)"},
            ],
        },
    ))
    fig = _theme(fig, height=240)
    st.plotly_chart(fig, use_container_width=True)


def chart_macronutrient(df: pd.DataFrame):
    """Donut chart for macronutrient split. Columns: ['macro','grams']."""
    colors = ["#2563EB", "#10B981", "#F59E0B", "#6366F1", "#EF4444"]
    fig = px.pie(df, names="macro", values="grams", hole=0.5,
                 color_discrete_sequence=colors)
    fig.update_traces(
        textposition="outside",
        textinfo="label+percent",
        textfont_size=12,
        pull=[0.03] * len(df),
    )
    fig = _theme(fig, height=320)
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", y=-0.18, font=dict(size=12)),
    )
    st.plotly_chart(fig, use_container_width=True)


def chart_calorie_breakdown(df: pd.DataFrame):
    """Horizontal bar chart. Columns: ['meal','calories']."""
    colors = ["#2563EB", "#10B981", "#F59E0B", "#6366F1"]
    fig = px.bar(df, x="calories", y="meal", orientation="h",
                 text="calories", color="meal",
                 color_discrete_sequence=colors)
    fig.update_traces(
        texttemplate="%{text} kcal", textposition="outside",
        textfont_size=11,
    )
    fig.update_layout(
        xaxis_title="Calories (kcal)", yaxis_title="",
        showlegend=False,
        yaxis={"categoryorder": "total ascending"},
    )
    fig = _theme(fig, height=260)
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
#  DOWNLOAD BUTTON
# ════════════════════════════════════════════════════════════════════

def download_button(data, filename: str = "report", format: str = "json"):
    """Render a download button for JSON, CSV, or PDF formats.

    Parameters
    ----------
    data     : dict or pd.DataFrame
    filename : base filename without extension
    format   : 'json' | 'csv' | 'pdf'
    """

    # ── serialiser (handles DataFrames inside dicts) ─────────────────
    def _serialize(obj):
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict("records")
        if isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_serialize(i) for i in obj]
        return obj

    # ── JSON ─────────────────────────────────────────────────────────
    if format == "json":
        payload = json.dumps(_serialize(data), indent=2)
        st.download_button(
            label="📄 Download JSON Report",
            data=payload,
            file_name=f"{filename}.json",
            mime="application/json",
            use_container_width=True,
        )

    # ── CSV ──────────────────────────────────────────────────────────
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

    # ── PDF ──────────────────────────────────────────────────────────
    elif format == "pdf":
        from fpdf import FPDF

        # ── unicode sanitiser ────────────────────────────────────────
        def _safe(text: str) -> str:
            return (
                str(text)
                .replace("\u2014", "-").replace("\u2013", "-")
                .replace("\u2018", "'").replace("\u2019", "'")
                .replace("\u201c", '"').replace("\u201d", '"')
                .replace("\u2022", "*").replace("\u2026", "...")
                .encode("latin-1", errors="replace").decode("latin-1")
            )

        # ── flatten nested structure (removes DataFrames safely) ─────
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

        # ── recursive writer ─────────────────────────────────────────
        def _write(pdf_obj, d: dict, indent: int = 0):
            indent = min(indent, 3)          # cap indent depth
            lm = 10 + indent * 5             # left margin per level

            for k, v in d.items():
                if pdf_obj.get_y() > pdf_obj.h - pdf_obj.b_margin - 15:
                    pdf_obj.add_page()
                pdf_obj.set_left_margin(lm)
                pdf_obj.set_x(lm)

                if isinstance(v, dict):
                    pdf_obj.set_font("Arial", "B", 11)
                    pdf_obj.set_fill_color(241, 245, 249)
                    pdf_obj.multi_cell(0, 8, _safe(str(k)),
                                       fill=(indent == 0))
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

        # ── build PDF ────────────────────────────────────────────────
        pdf = FPDF()
        pdf.set_margins(left=10, top=15, right=10)
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # Title block
        pdf.set_font("Arial", "B", 18)
        pdf.cell(0, 12,
                 _safe("Smart Health Dashboard - Health Report"),
                 ln=True, align="C")
        pdf.set_font("Arial", size=10)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 8,
                 _safe("AI-Powered Personalised Nutrition & Disease Risk Analysis"),
                 ln=True, align="C")
        pdf.set_text_color(0, 0, 0)
        pdf.ln(4)
        pdf.set_draw_color(226, 232, 240)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(6)

        # Write flattened data
        safe_data = _flatten(data) if isinstance(data, dict) else _safe(str(data))
        if isinstance(safe_data, dict):
            _write(pdf, safe_data)
        else:
            pdf.set_font("Arial", size=10)
            pdf.multi_cell(0, 7, safe_data)

        # Get PDF output as bytes
        pdf_output = pdf.output(dest='S')
        if isinstance(pdf_output, str):
            pdf_bytes = pdf_output.encode('latin-1')
        else:
            pdf_bytes = pdf_output
        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_bytes,
            file_name=f"{filename}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    else:
        st.error(f"Unsupported format: {format}")
