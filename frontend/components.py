# frontend/components.py
"""Reusable UI components for the healthcare dashboard.
Blue & White professional healthcare theme.

Functions:
- load_css()                                   — inject CSS
- metric_card(label, value, icon, color)       — KPI card
- badge(label, color)                          — pill badge
- prediction_card(name, pred_class, confidence, risk, severity, color)
- chart_macronutrient(df)
- chart_calorie_breakdown(df)
- chart_bmi_gauge(bmi)
- chart_disease_risk(score, title)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ── Blue-based chart palette ──────────────────────────────────────
_BLUE_PALETTE = [
    "#2563EB", "#0EA5E9", "#38BDF8", "#7DD3FC", "#BAE6FD",
    "#1E3A8A", "#1D4ED8", "#3B82F6", "#60A5FA", "#93C5FD",
]

# =================================================================
#  CSS
# =================================================================

def load_css():
    """Load custom CSS from frontend/assets/style.css."""
    css_path = Path(__file__).parent / "assets" / "style.css"
    try:
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Failed to load CSS: {e}")


# =================================================================
#  METRIC CARD
# =================================================================

def metric_card(label: str, value: str, icon: str = "", color: str = "#2563EB"):
    """Render a clean KPI card.

    Parameters
    ----------
    label : str   — metric name
    value : str   — formatted metric value
    icon  : str   — emoji or HTML icon
    color : str   — hex accent colour for the value text
    """
    html = f"""
    <div style="
        background:#FFFFFF;
        border:1px solid #E2E8F0;
        border-radius:12px;
        padding:1.2rem 1.3rem;
        margin:0.4rem 0;
        box-shadow:0 2px 8px rgba(15,23,42,.06);
        display:flex;
        flex-direction:column;
        align-items:center;
        text-align:center;
        transition:transform .18s,box-shadow .18s;
        position:relative;
        overflow:hidden;
    ">
        <div style="
            font-size:1.8rem;
            width:48px;height:48px;
            border-radius:11px;
            background:#EFF6FF;
            display:flex;align-items:center;justify-content:center;
            margin-bottom:0.4rem;
        ">{icon}</div>
        <div style="
            font-size:0.68rem;
            font-weight:600;
            text-transform:uppercase;
            letter-spacing:0.08em;
            color:#64748B;
            margin-bottom:0.28rem;
            margin-top:0.4rem;
            font-family:Inter,sans-serif;
        ">{label}</div>
        <div style="
            font-size:1.75rem;
            font-weight:800;
            letter-spacing:-0.03em;
            color:{color};
            line-height:1.1;
            font-family:Inter,sans-serif;
        ">{value}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# =================================================================
#  BADGE
# =================================================================

def badge(label: str, color: str = "#2563EB"):
    """Render a pill-shaped coloured badge."""
    html = (
        f"<span style='"
        f"display:inline-flex;align-items:center;"
        f"padding:0.26rem 0.75rem;"
        f"border-radius:99px;"
        f"font-size:0.74rem;font-weight:600;"
        f"letter-spacing:0.04em;"
        f"background:{color};color:#fff;"
        f"font-family:Inter,sans-serif;"
        f"'>{label}</span>"
    )
    st.markdown(html, unsafe_allow_html=True)


# =================================================================
#  PREDICTION CARD
# =================================================================

def prediction_card(
    name: str,
    pred_class: str,
    confidence: float,
    risk: str,
    severity: str,
    color: str = "#2563EB",
):
    """Render a disease prediction card.

    Parameters
    ----------
    name       : Disease name
    pred_class : Predicted class/label
    confidence : Confidence score (0–1 float)
    risk       : Human-readable risk level string
    severity   : Severity description
    color      : Left-border accent colour
    """
    # Map colour to semantic risk badge
    _risk_lower = risk.lower()
    if "high" in _risk_lower:
        badge_bg, badge_color = "#FEF2F2", "#991B1B"
        dot = "🔴"
    elif "moderate" in _risk_lower or "medium" in _risk_lower or "warning" in _risk_lower:
        badge_bg, badge_color = "#FFFBEB", "#92400E"
        dot = "🟡"
    else:
        badge_bg, badge_color = "#F0FDF4", "#14532D"
        dot = "🟢"

    conf_pct = f"{confidence:.1%}" if isinstance(confidence, float) else str(confidence)

    html = f"""
    <div style="
        background:#FFFFFF;
        border:1px solid #E2E8F0;
        border-left:4px solid {color};
        border-radius:12px;
        padding:1.3rem 1.2rem;
        margin:0.4rem 0;
        box-shadow:0 2px 8px rgba(15,23,42,.06);
        transition:transform .18s,box-shadow .18s;
    ">
        <div style="
            font-size:1.05rem;font-weight:700;
            color:#0F172A;margin-bottom:0.25rem;
            font-family:Inter,sans-serif;
        ">{name}</div>
        <div style="
            font-size:0.82rem;color:#64748B;
            margin-bottom:0.65rem;
            font-family:Inter,sans-serif;
        ">Class: {pred_class}</div>
        <span style="
            display:inline-flex;align-items:center;gap:0.28rem;
            padding:0.28rem 0.8rem;
            border-radius:99px;
            font-size:0.72rem;font-weight:700;
            letter-spacing:0.05em;text-transform:uppercase;
            background:{badge_bg};color:{badge_color};
            border:1px solid {badge_color}22;
            font-family:Inter,sans-serif;
        ">{dot}&nbsp;{risk}</span>
        <div style="
            margin-top:0.6rem;
            font-size:0.78rem;color:#64748B;
            font-family:Inter,sans-serif;
        ">
            <span style="font-weight:600;">Confidence:</span> {conf_pct}
            &nbsp;·&nbsp;
            <span style="font-weight:600;">Severity:</span> {severity}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# =================================================================
#  PLOTLY HELPERS
# =================================================================

def _theme(fig, height: int = 260):
    """Apply clean transparent background and Inter font."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Inter, Plus Jakarta Sans, sans-serif",
            size=12,
            color="#334155",
        ),
        margin=dict(t=40, b=25, l=20, r=20),
        height=height,
    )
    return fig


def chart_bmi_gauge(bmi: float):
    """Gauge chart visualising BMI against WHO thresholds."""
    if bmi < 18.5:
        color = "#0EA5E9"       # sky blue — underweight
    elif bmi < 25:
        color = "#16A34A"       # green — healthy
    elif bmi < 30:
        color = "#F59E0B"       # amber — overweight
    else:
        color = "#DC2626"       # red — obese

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=bmi,
        delta={
            "reference": 22,
            "valueformat": ".1f",
            "increasing": {"color": "#DC2626"},
            "decreasing": {"color": "#16A34A"},
        },
        number={"valueformat": ".1f",
                "font": {"size": 34, "color": "#0F172A"}},
        title={"text": "Body Mass Index (BMI)",
               "font": {"size": 13, "color": "#64748B"}},
        gauge={
            "axis": {
                "range": [10, 40],
                "tickwidth": 1,
                "tickcolor": "#E2E8F0",
                "tickvals": [10, 18.5, 25, 30, 40],
                "ticktext": ["10", "18.5", "25", "30", "40"],
                "tickfont": {"size": 10},
            },
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [10,   18.5], "color": "rgba(14,165,233,.12)"},
                {"range": [18.5, 25],   "color": "rgba(22,163,74,.12)"},
                {"range": [25,   30],   "color": "rgba(245,158,11,.12)"},
                {"range": [30,   40],   "color": "rgba(220,38,38,.12)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.8,
                "value": bmi,
            },
        },
    ))
    fig = _theme(fig, height=270)
    st.plotly_chart(fig, use_container_width=True)


def chart_disease_risk(score: float, title: str = "Risk"):
    """Gauge chart for a generic disease risk score (0–1)."""
    pct = score * 100
    color = "#16A34A" if pct < 33 else ("#F59E0B" if pct < 66 else "#DC2626")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "valueformat": ".0f",
                "font": {"size": 30, "color": "#0F172A"}},
        title={"text": title, "font": {"size": 13, "color": "#64748B"}},
        gauge={
            "axis": {"range": [0, 100],
                     "tickwidth": 1, "tickcolor": "#E2E8F0"},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  33], "color": "rgba(22,163,74,.12)"},
                {"range": [33, 66], "color": "rgba(245,158,11,.12)"},
                {"range": [66,100], "color": "rgba(220,38,38,.12)"},
            ],
        },
    ))
    fig = _theme(fig, height=240)
    st.plotly_chart(fig, use_container_width=True)


def chart_macronutrient(df: pd.DataFrame):
    """Donut chart showing macronutrient distribution.

    Expected columns: ['macro', 'grams']
    """
    fig = px.pie(
        df, names="macro", values="grams", hole=0.5,
        color_discrete_sequence=_BLUE_PALETTE,
    )
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
    """Horizontal bar chart for calorie breakdown per meal.

    Expected columns: ['meal', 'calories']
    """
    fig = px.bar(
        df, x="calories", y="meal", orientation="h",
        text="calories", color="meal",
        color_discrete_sequence=_BLUE_PALETTE,
    )
    fig.update_traces(
        texttemplate="%{text} kcal",
        textposition="outside",
        textfont_size=11,
    )
    fig.update_layout(
        xaxis_title="Calories (kcal)",
        yaxis_title="",
        showlegend=False,
        yaxis={"categoryorder": "total ascending"},
    )
    fig = _theme(fig, height=260)
    st.plotly_chart(fig, use_container_width=True)
