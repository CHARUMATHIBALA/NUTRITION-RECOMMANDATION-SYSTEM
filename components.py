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
    # ── new parameters (replace old `confidence`) ────────────────────
    final_status: str = "",
    model_probability: "float | None" = None,
    # kept for backward-compat; ignored when model_probability is provided
    confidence: "float | None" = None,
):
    """Tall disease prediction card.

    Parameters
    ----------
    icon              : Emoji (e.g. '🩸')
    name              : Disease display name
    pred_class        : Human-readable model output label
    risk              : Badge text — comes from RiskResult.card_risk_text
    level             : 'high' | 'medium' | 'low' — controls colour stripe
    final_status      : RiskResult.final_status — shown as sub-label on card
    model_probability : RiskResult.model_probability (0–100 float)
                        Displayed as "Model Probability: X%" NOT "Confidence: X%"
                        to correctly separate ML probability from clinical risk.
    confidence        : Legacy parameter — ignored when model_probability given.

    Design note
    -----------
    The card NEVER derives risk from pred_class.  All risk information
    comes pre-computed from backend.risk.classify_risk() via the caller.
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

    # ── Badge text — use card_risk_text, never append " Risk" blindly ─
    # "Screening Flag" and "Unavailable" should not get " Risk" appended.
    _no_suffix = {"Screening Flag", "Unavailable", "Borderline",
                  "Model Flag", "Normal"}
    badge_text = risk if (risk in _no_suffix or risk.endswith("Risk")) else f"{risk} Risk"

    # ── Probability display ───────────────────────────────────────────
    # Use model_probability when available (new path).
    # Fall back to legacy confidence for backward compat.
    _prob = model_probability if model_probability is not None else confidence
    prob_html = ""
    if _prob is not None:
        # Label it "Model Probability" to make clear this is NOT clinical risk.
        prob_html = (
            f"<div class='pred-confidence'>"
            f"Model Probability: {_prob:.1f}%"
            f"</div>"
        )

    # ── Final-status sub-label (shown below the badge) ────────────────
    # Only rendered when final_status differs meaningfully from badge_text.
    status_html = ""
    if final_status and final_status not in ("", "Unknown", risk):
        # Color-coded sub-label so the user can immediately see the distinction
        # between "Screening Flag" (blue) and "High Risk" (red).
        _status_colors = {
            "High Risk":      ("background:#FEE2E2;color:#991B1B;",   "🔴"),
            "Moderate Risk":  ("background:#FEF3C7;color:#92400E;",   "🟡"),
            "Model Flag":     ("background:#EFF6FF;color:#1E40AF;",   "🔵"),
            "Low Risk":       ("background:#D1FAE5;color:#065F46;",   "🟢"),
            "Borderline":     ("background:#FEF3C7;color:#92400E;",   "🟡"),
            "Unavailable":    ("background:#F1F5F9;color:#64748B;",   "⚪"),
        }
        _sc, _sdot = _status_colors.get(
            final_status,
            ("background:#F1F5F9;color:#64748B;", "ℹ️"),
        )
        status_html = (
            f"<div style='margin-top:0.5rem;padding:0.25rem 0.7rem;"
            f"border-radius:99px;font-size:0.72rem;font-weight:700;"
            f"display:inline-block;{_sc}'>"
            f"{_sdot}&nbsp;{final_status}"
            f"</div>"
        )

    st.markdown(
        f"""
        <div class='pred-card'>
            <div class='pred-card-stripe' style='{stripe}'></div>
            <div class='pred-icon-wrap' style='{icon_bg}'>{icon}</div>
            <div class='pred-name'>{name}</div>
            <div class='pred-result'>{pred_class}</div>
            <span class='pred-risk-badge {badge_cls}'>{dot}&nbsp;{badge_text}</span>
            {status_html}
            {prob_html}
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

        # Get PDF output as bytes — fpdf2 >= 2.2.0 returns bytearray from output()
        # bytes() wraps bytearray cleanly; never call .encode() on bytearray
        # Get PDF output as bytes. Use dest='S' to get a string representation and encode safely.
        try:
            pdf_output = pdf.output(dest='S')
            # Ensure we have a bytes object; encode with latin-1 and replace unknown chars.
            pdf_bytes = pdf_output.encode('latin-1', errors='replace')
        except Exception:
            # Fallback for older fpdf versions that return a bytearray directly.
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
#  XAI — FEATURE IMPORTANCE CHART  (full-width Plotly, no expander)
# ════════════════════════════════════════════════════════════════════

def chart_xai_feature_importance(feature_rows: list, disease_name: str = "") -> None:
    """Render a full-width horizontal Plotly bar chart of feature importances.

    Must be called at full page width (not inside a narrow column) so the
    outside text labels have room to render without overlapping the bars.
    """
    if not feature_rows:
        return

    _impact_colour = {
        "High":   "#EF4444",
        "Medium": "#F59E0B",
        "Low":    "#10B981",
    }

    # Build lists in display order (highest importance last = top of chart)
    rows_display = list(reversed(feature_rows))
    labels  = [r.feature for r in rows_display]
    scores  = [round(r.importance * 100, 2) for r in rows_display]
    colours = [_impact_colour.get(r.impact, "#6366F1") for r in rows_display]
    hover   = [
        f"<b>{r.feature}</b><br>Value: {r.value}<br>"
        f"Status: {r.direction}<br>Impact: {r.impact}"
        for r in rows_display
    ]
    outside_text = [f"{r.value}" for r in rows_display]

    fig = go.Figure(go.Bar(
        x=scores,
        y=labels,
        orientation="h",
        marker_color=colours,
        marker_line_width=0,
        hovertemplate=hover,
        hoverinfo="text",
        text=outside_text,
        textposition="outside",
        textfont=dict(size=11, color="#475569"),
        cliponaxis=False,
    ))

    max_score = max(scores) if scores else 1
    fig.update_layout(
        xaxis=dict(
            title="Contribution Score",
            range=[0, max_score * 1.60],
            showgrid=True,
            gridcolor="rgba(203,213,225,0.4)",
            zeroline=False,
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            tickfont=dict(size=12),
            automargin=True,
        ),
        showlegend=False,
        bargap=0.30,
        margin=dict(l=10, r=10, t=10, b=30),
    )
    fig = _theme(fig, height=max(160, len(feature_rows) * 46))
    st.plotly_chart(fig, use_container_width=True)


def xai_explanation_panel(xai_result, disease_name: str = "") -> None:
    """Render the XAI explanation panel for one disease — inline, no expander.

    Layout (all full-width, no columns):
      1. Method note
      2. Legend (High / Medium / Low chips)
      3. HTML feature-bar rows  (pure CSS bars — no Plotly, no columns)
      4. Plotly horizontal bar chart
      5. Patient-friendly summary box
      6. Disclaimer

    Using st.expander() caused the chevron arrow to bleed into the label text
    when rendered inside narrow columns — it has been removed entirely.
    The parent app.py now uses st.tabs() per disease which provides the same
    collapsible UX at full page width.
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

    # ── 1. Method note ────────────────────────────────────────────────
    st.markdown(
        f"<div class='xai-method-note'>"
        f"<strong>Method:</strong> {xai_result.method}. "
        "Each score = model feature weight × how far your value deviates from "
        "the clinical reference range."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── 2. Legend ─────────────────────────────────────────────────────
    st.markdown(
        "<div class='xai-legend'>"
        "<span class='xai-legend-item'>"
        "<span class='xai-legend-dot high'></span>High impact</span>"
        "<span class='xai-legend-item'>"
        "<span class='xai-legend-dot medium'></span>Medium impact</span>"
        "<span class='xai-legend-item'>"
        "<span class='xai-legend-dot low'></span>Low impact</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── 3. HTML feature rows (pure CSS bars) ──────────────────────────
    # Normalise scores to 0–100% relative to the highest score
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
        pct        = round(r.importance / max_score * 100, 1)
        imp_cls    = r.impact.lower()   # "high" | "medium" | "low"
        dir_cls    = _dir_class.get(r.direction, "normal")
        dir_lbl    = _dir_label.get(r.direction, r.direction)
        row_html  += (
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

    # ── 4. Plotly chart (full-width, rendered outside the panel div) ──
    chart_xai_feature_importance(rows, disease_name)

    # ── 5. Summary ────────────────────────────────────────────────────
    if xai_result.summary:
        st.markdown(
            f"<div class='xai-summary'>💡 {xai_result.summary}</div>",
            unsafe_allow_html=True,
        )

    # ── 6. Disclaimer ─────────────────────────────────────────────────
    st.markdown(
        "<div class='xai-disclaimer'>"
        "⚠️ This is a screening result only. It does not constitute a medical "
        "diagnosis. Please consult a qualified healthcare professional."
        "</div>",
        unsafe_allow_html=True,
    )
