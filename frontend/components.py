# frontend/components.py
"""Reusable UI components for the healthcare dashboard.

Functions:
- load_css(): inject custom CSS.
- metric_card(label, value, icon="", color="#4caf50")
- badge(label, color="#2196f3")
- prediction_card(name, pred_class, confidence, risk, severity, color)
- chart_macronutrient(df)
- chart_calorie_breakdown(df)
- chart_bmi_gauge(bmi)
- chart_disease_risk(score)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# -------------------------------------------------
# CSS handling
# -------------------------------------------------

def load_css():
    """Load custom CSS from the assets folder.
    The CSS file is located at ``frontend/assets/style.css``.
    """
    css_path = Path(__file__).parent / "assets" / "style.css"
    try:
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Failed to load CSS: {e}")

# -------------------------------------------------
# Metric Card
# -------------------------------------------------

def metric_card(label: str, value: str, icon: str = "", color: str = "#4caf50"):
    """Render a rounded metric card.
    .card {
      background: var(--card-bg);
      border-radius: 12px;
      padding: 1.2rem;
      margin: 0.6rem 0;
      box-shadow: 0 2px 8px rgba(0,0,0,0.05);
      display: flex;
      align-items: flex-start;
      flex-direction: column;
      gap: 0.6rem;
      min-width: 0;
      word-break: break-word;
      overflow-wrap: anywhere;
    }
    Parameters
    ----------
    label : str
        Metric label.
    value : str
        Formatted metric value.
    icon : str, optional
        Emoji or HTML icon.
    color : str, optional
        Hex colour for the icon.
    """
    html = f"""
    <div class='card'>
        <div class='metric-icon' style='color:{color};'>{icon}</div>
        <div class='metric-label'>{label}</div>
        <div class='metric-value'>{value}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# -------------------------------------------------
# Badge
# -------------------------------------------------

def badge(label: str, color: str = "#2196f3"):
    """Render a coloured badge (e.g., BMI category)."""
    html = f"<span class='badge' style='background:{color};'>{label}</span>"
    st.markdown(html, unsafe_allow_html=True)

# -------------------------------------------------
# Prediction Card
# -------------------------------------------------

def prediction_card(name: str, pred_class: str, confidence: float, risk: str, severity: str, color: str = "#ff9800"):
    """Render a disease prediction card.
    Parameters
    ----------
    name : str
        Disease name.
    pred_class : str
        Predicted class/label.
    confidence : float
        Confidence score (0‑1).
    risk : str
        Human‑readable risk level.
    severity : str
        Severity description.
    color : str, optional
        Accent colour for the card.
    """
    html = f"""
    <div class='card prediction-card' style='border-left:4px solid {color};'>
        <div class='metric-icon'>{name}</div>
        <div class='metric-label'>Class: {pred_class}</div>
        <div class='metric-value'>Confidence: {confidence:.2%}</div>
        <div class='metric-label'>Risk: {risk}</div>
        <div class='metric-label'>Severity: {severity}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# -------------------------------------------------
# Plotly Charts
# -------------------------------------------------

def chart_macronutrient(df: pd.DataFrame):
    """Donut chart showing macronutrient distribution.
    Expected columns: ['macro', 'grams']
    """
    fig = px.pie(df, names='macro', values='grams', hole=0.4,
                 color_discrete_sequence=px.colors.sequential.Teal)
    fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), showlegend=True)
    st.plotly_chart(fig, use_container_width=True)


def chart_calorie_breakdown(df: pd.DataFrame):
    """Bar chart for calorie breakdown per meal.
    Expected columns: ['meal', 'calories']
    """
    fig = px.bar(df, x='meal', y='calories', text='calories',
                 color='meal', color_discrete_sequence=px.colors.qualitative.Vivid)
    fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), xaxis_title='Meal', yaxis_title='Calories')
    st.plotly_chart(fig, use_container_width=True)


def chart_bmi_gauge(bmi: float):
    """Gauge chart visualising BMI against standard thresholds."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=bmi,
        title={'text': "BMI"},
        gauge={
            'axis': {'range': [0, 40]},
            'steps': [
                {'range': [0, 18.5], 'color': "#5bc0de"},
                {'range': [18.5, 24.9], 'color': "#5cb85c"},
                {'range': [25, 29.9], 'color': "#f0ad4e"},
                {'range': [30, 40], 'color': "#d9534f"},
            ],
            'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': bmi},
        }
    ))
    st.plotly_chart(fig, use_container_width=True)


def chart_disease_risk(score: float, title: str = "Risk"):
    """Gauge chart for a generic disease risk score (0‑1)."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        title={'text': title},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#2c7a7b"},
            'steps': [
                {'range': [0, 33], 'color': "#5cb85c"},
                {'range': [33, 66], 'color': "#f0ad4e"},
                {'range': [66, 100], 'color': "#d9534f"},
            ],
        }
    ))
    st.plotly_chart(fig, use_container_width=True)
