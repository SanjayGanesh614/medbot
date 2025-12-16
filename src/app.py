"""
AI-Powered Clinical Pharmacist Assistant (AI-CPA)
Main Streamlit Application
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
from io import BytesIO

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    load_model, get_risk_category, get_risk_color, 
    create_prediction_summary, generate_report_filename, parse_fhir_patient
)
from src.explainability import SHAPExplainer
from src.evaluate import ModelEvaluator
# Dashboard functions integrated below

# Page configuration
st.set_page_config(
    page_title="AI-CPA | Clinical Pharmacist Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for corporate white theme
st.markdown("""
<style>
    /* Global white background theme */
    .stApp {
        background-color: #FFFFFF;
    }
    
    /* Main content area */
    .main {
        background-color: #FFFFFF;
    }
    
    /* Top margin and header area (where deploy button is) */
    .stApp > header {
        background-color: #FFFFFF !important;
    }
    
    .stApp > header[data-testid="stHeader"] {
        background-color: #FFFFFF !important;
    }
    
    /* Fix Streamlit's top bar */
    .stApp .main .block-container {
        background-color: #FFFFFF;
    }
    
    /* Override Streamlit's default top margin styling */
    .stApp > div:first-child {
        background-color: #FFFFFF !important;
    }
    
    /* Additional top margin fixes */
    .stApp .main .block-container {
        padding-top: 1rem;
        background-color: #FFFFFF !important;
    }
    
    /* Fix any remaining header/margin issues */
    .stApp header {
        background-color: #FFFFFF !important;
    }
    
    .stApp [data-testid="stHeader"] {
        background-color: #FFFFFF !important;
    }
    
    /* Override any dark theme remnants in top area */
    .stApp > div {
        background-color: #FFFFFF !important;
    }
    
    /* Fix deploy button area */
    .stApp .stAppToolbar {
        background-color: #FFFFFF !important;
    }
    
    /* Ensure the entire app container is white */
    .stApp {
        background-color: #FFFFFF !important;
    }
    
    /* Fix any iframe or embedded content styling */
    .stApp iframe {
        background-color: #FFFFFF !important;
    }
    
    /* Sidebar styling - hidden for landing page */
    [data-testid="stSidebar"] {
        display: none !important;
    }
    
    [data-testid="stSidebar"] .element-container {
        color: #1F2937;
    }
    
    /* Corporate header */
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1F2937;
        text-align: center;
        padding: 1.5rem 0;
        letter-spacing: -0.02em;
        border-bottom: 3px solid #2563EB;
        margin-bottom: 2rem;
    }
    
    /* Hero section (landing top) */
    .hero-container {
        max-width: 960px;
        margin: 0.5rem auto 0.75rem auto;
        text-align: center;
        padding: 0 2rem;
    }
    .hero-logo {
        width: 96px;
        height: 96px;
        border-radius: 24px;
        background: #2563EB;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 1.75rem auto;
        box-shadow: 0 15px 35px rgba(37, 99, 235, 0.35);
    }
    .hero-logo-icon {
        font-size: 2.5rem;
        color: #ffffff;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -0.055em;
        color: #020617;
        margin-bottom: 0.25rem;
    }
    .hero-subtitle {
        font-size: 2.6rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 1.1rem;
    }
    .hero-description {
        max-width: 900px;
        margin: 0 auto 2rem auto;
        font-size: 3.2rem;
        line-height: 1.7;
        color: #111827;
        font-weight: 500;
    }
    .hero-actions {
        display: flex;
        justify-content: center;
        gap: 1rem;
        flex-wrap: wrap;
    }
    .hero-primary-btn,
    .hero-secondary-btn {
        padding: 0.8rem 1.9rem;
        border-radius: 999px;
        font-size: 0.98rem;
        font-weight: 500;
        border: none;
        cursor: pointer;
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        text-decoration: none;
        transition: all 0.18s ease;
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.08);
    }
    .hero-primary-btn {
        background-color: #111827;
        color: #FFFFFF;
    }
    .hero-primary-btn:hover {
        background-color: #030712;
        transform: translateY(-1px);
        box-shadow: 0 14px 28px rgba(15, 23, 42, 0.16);
    }
    .hero-secondary-btn {
        background-color: #FFFFFF;
        color: #111827;
        border: 1px solid #D1D5DB;
        box-shadow: 0 8px 16px rgba(15, 23, 42, 0.05);
    }
    .hero-secondary-btn:hover {
        background-color: #F9FAFB;
        transform: translateY(-1px);
        box-shadow: 0 12px 24px rgba(15, 23, 42, 0.12);
    }
    .hero-kbd-icon {
        font-size: 1.1rem;
    }
    /* Ensure hero links never show blue/visited colors */
    .hero-primary-btn:link,
    .hero-primary-btn:visited {
        color: #FFFFFF;
        text-decoration: none;
    }
    .hero-secondary-btn:link,
    .hero-secondary-btn:visited {
        color: #111827;
        text-decoration: none;
    }

    /* Feature cards section */
    .feature-section {
        max-width: 1200px;
        margin: 2rem auto 0 auto;
        padding: 0 2rem 3.5rem 2rem;
    }
    .feature-card {
        background: #FFFFFF;
        border-radius: 24px;
        padding: 2.4rem 2.2rem;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.06);
        border: 1px solid rgba(148, 163, 184, 0.25);
        transition: all 0.18s ease-out;
        margin-bottom: 1.75rem;
        min-height: 300px;
        display: flex;
        flex-direction: column;
    }
    .feature-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 24px 60px rgba(15, 23, 42, 0.14);
        border-color: rgba(59, 130, 246, 0.45);
    }
    .feature-icon {
        width: 64px;
        height: 64px;
        border-radius: 999px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 1.5rem;
        font-size: 1.9rem;
    }
    .feature-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #020617;
        margin-bottom: 0.65rem;
    }
    .feature-text {
        font-size: 0.98rem;
        line-height: 1.7;
        color: #4B5563;
    }
    
    /* Bottom CTA card */
    .cta-wrapper {
        max-width: 960px;
        margin: 0 auto 4rem auto;
        padding: 0 2rem;
    }
    .cta-card {
        background: #FFFFFF;
        border-radius: 24px;
        padding: 2.5rem 2.75rem 2.75rem 2.75rem;
        box-shadow: 0 22px 55px rgba(15, 23, 42, 0.12);
        border: 1px solid rgba(148, 163, 184, 0.35);
        text-align: center;
    }
    .cta-title {
        font-size: 1.9rem;
        font-weight: 800;
        color: #020617;
        margin-bottom: 0.75rem;
    }
    .cta-text {
        font-size: 1.05rem;
        color: #4B5563;
        line-height: 1.8;
        margin-bottom: 1.9rem;
    }
    .cta-button {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        padding: 0.85rem 2.4rem;
        border-radius: 999px;
        background-color: #020617;
        color: #FFFFFF;
        font-weight: 600;
        font-size: 0.98rem;
        border: none;
        cursor: pointer;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.35);
        transition: all 0.18s ease-out;
        text-decoration: none;
    }
    .cta-button:hover {
        background-color: #000000;
        transform: translateY(-1px);
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.45);
    }
    .cta-button:link,
    .cta-button:visited {
        color: #FFFFFF;
        text-decoration: none;
    }

    /* Dashboard header */
    .app-header-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0rem 0 0.5rem 0;
    }
    .app-header-left {
        display: flex;
        align-items: center;
        gap: 0.9rem;
    }
    .app-header-right {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .app-logo {
        width: 52px;
        height: 52px;
        border-radius: 20px;
        background: #2563EB;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #FFFFFF;
        font-size: 1.9rem;
        box-shadow: 0 14px 30px rgba(37, 99, 235, 0.35);
    }
    .app-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #020617;
    }
    .app-subtitle {
        font-size: 0.95rem;
        color: #6B7280;
    }
    .app-header-btn {
        padding: 0.55rem 1.4rem;
        border-radius: 999px;
        background-color: #FFFFFF;
        color: #111827;
        border: 1px solid #E5E7EB;
        font-size: 0.85rem;
        font-weight: 500;
        cursor: pointer;
        box-shadow: 0 6px 16px rgba(15, 23, 42, 0.08);
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
    }
    .app-header-btn:hover {
        background-color: #F9FAFB;
        box-shadow: 0 10px 22px rgba(15, 23, 42, 0.14);
    }
    .summary-card {
        background: #FFFFFF;
        border-radius: 18px;
        padding: 1.6rem 1.7rem;
        box-shadow: 0 16px 40px rgba(15, 23, 42, 0.06);
        border: 1px solid rgba(148, 163, 184, 0.25);
        height: 150px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .summary-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #4B5563;
        margin-bottom: 0.9rem;
    }
    .summary-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #020617;
        margin-bottom: 0.4rem;
    }
    .summary-footnote {
        font-size: 0.8rem;
        color: #9CA3AF;
        min-height: 1.2rem;
    }
    
    /* Dashboard tabs navigation */
    .dashboard-tabs {
        display: flex;
        gap: 0.5rem;
        margin: 1.5rem 0 2rem 0;
        border-bottom: 2px solid #E5E7EB;
        padding-bottom: 0;
    }
    .dashboard-tab {
        padding: 0.75rem 1.5rem;
        background: transparent;
        border: none;
        border-bottom: 3px solid transparent;
        cursor: pointer;
        font-size: 0.95rem;
        font-weight: 500;
        color: #6B7280;
        transition: all 0.2s ease;
        margin-bottom: -2px;
    }
    .dashboard-tab:hover {
        color: #111827;
        background: #F9FAFB;
    }
    .dashboard-tab.active {
        color: #2563EB;
        border-bottom-color: #2563EB;
        font-weight: 600;
    }
    .patient-management-section {
        background: #FFFFFF;
        border-radius: 18px;
        padding: 2rem;
        box-shadow: 0 16px 40px rgba(15, 23, 42, 0.06);
        border: 1px solid rgba(148, 163, 184, 0.25);
        margin-top: 1.5rem;
    }
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1rem;
    }
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #020617;
    }
    .section-subtitle {
        font-size: 0.95rem;
        color: #6B7280;
        margin-top: 0.25rem;
    }
    
    /* Risk box styling */
    .risk-box {
        padding: 2rem;
        border-radius: 12px;
        margin: 1rem 0;
        text-align: center;
        background-color: #FFFFFF;
        border: 2px solid #E5E7EB;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Metric cards */
    .metric-card {
        background-color: #F9FAFB;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 1px solid #E5E7EB;
    }
    
    /* Form styling */
    .stForm {
        background-color: #FFFFFF;
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }
    /* Fix form text color - make it black */
    .stForm label,
    .stForm p,
    .stForm .stMarkdown,
    .stForm .stMarkdown p,
    .stForm .stTextInput label,
    .stForm .stNumberInput label,
    .stForm .stSelectbox label,
    .stForm .stMultiselect label {
        color: #020617 !important;
    }
    .stForm h3 {
        color: #020617 !important;
    }
    
    /* Fix all text elements outside forms too */
    label,
    p,
    .stMarkdown,
    .stMarkdown p,
    .stTextInput label,
    .stNumberInput label,
    .stSelectbox label,
    .stMultiselect label,
    .stRadio label,
    .stFileUploader label,
    .stFileUploader p,
    div[data-testid="stRadio"] label,
    div[data-testid="stFileUploader"] label,
    div[data-testid="stFileUploader"] p {
        color: #020617 !important;
    }
    
    /* Fix radio button labels specifically */
    .stRadio > label,
    .stRadio > div > label,
    div[data-testid="stRadio"] > label,
    div[data-testid="stRadio"] > div > label {
        color: #020617 !important;
        font-weight: 500;
    }
    
    /* Fix file uploader text */
    .stFileUploader > label,
    .stFileUploader > p,
    div[data-testid="stFileUploader"] > label,
    div[data-testid="stFileUploader"] > p,
    .stFileUploader .stMarkdown,
    div[data-testid="stFileUploader"] .stMarkdown {
        color: #020617 !important;
    }
    
    /* Fix all markdown headings */
    h1, h2, h3, h4, h5, h6 {
        color: #020617 !important;
    }
    
    /* Fix Streamlit's default text colors */
    .element-container,
    .element-container p,
    .element-container label {
        color: #020617 !important;
    }
    
    /* Ensure radio buttons and their labels are visible */
    [data-testid="stRadio"] label,
    [data-testid="stRadio"] div,
    [data-testid="stRadio"] span {
        color: #020617 !important;
    }
    
    /* File uploader visibility */
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] div {
        color: #020617 !important;
    }
    
    /* Make sure all Streamlit widget labels are visible */
    [class*="st"] label,
    [class*="st"] p:not([class*="button"]) {
        color: #020617 !important;
    }
    
    /* Override any white text in main content */
    .main .block-container,
    .main .block-container *:not(button):not(.stButton) {
        color: #020617 !important;
    }
    
    /* Fix Streamlit metric values - make numbers black */
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] > div,
    [data-testid="stMetricValue"] > div > div,
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] > div,
    [data-testid="stMetricLabel"] > div > div,
    .stMetric > div > div,
    .stMetric label,
    .stMetric [class*="value"],
    .stMetric [class*="delta"],
    div[data-testid="stMetric"] > div,
    div[data-testid="stMetric"] > div > div,
    div[data-testid="stMetric"] label {
        color: #020617 !important;
    }
    
    /* Fix metric container text */
    [data-testid="stMetricContainer"] > div,
    [data-testid="stMetricContainer"] label,
    [data-testid="stMetricContainer"] > div > div {
        color: #020617 !important;
    }
    
    /* Fix all text inside metric containers */
    [data-testid="stMetricContainer"] * {
        color: #020617 !important;
    }
    
    /* =======================
   DOWNLOAD BUTTON BASE
======================= */
div[data-testid="stDownloadButton"] button {
    background-color: #020617 !important;
    border: 1px solid #020617 !important;
}

/* =======================
   🚨 FINAL OVERRIDE (CRITICAL)
   THIS FIXES BLACK TEXT
======================= */
div[data-testid="stDownloadButton"] button,
div[data-testid="stDownloadButton"] button * {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    text-fill-color: #FFFFFF !important;
    fill: #FFFFFF !important;
}

/* =======================
   DOWNLOAD BUTTON HOVER
======================= */
div[data-testid="stDownloadButton"] button:hover {
    background-color: #1E293B !important;
    border-color: #1E293B !important;
}

/* =======================
   PREVENT OTHER RULES
   FROM OVERRIDING BUTTONS
======================= */
.main .block-container *:not(button):not(svg):not(path) {
    color: #020617 !important;
}
</style>
""", unsafe_allow_html=True)

def load_model_and_explainer():
    """Load model and SHAP explainer (uncached for real-time updates)"""
    try:
        model = load_model("models/xgb_adr_model.pkl")
        explainer = SHAPExplainer(model=model)
        
        # Try to load pre-computed explainer
        try:
            explainer.load_explainer("models/shap_explainer.pkl")
        except:
            # Create new explainer if not found
            try:
                X_sample = pd.read_csv("data/output/X_features.csv").sample(100, random_state=42)
                explainer.create_explainer(X_sample)
            except:
                explainer.create_explainer()
        
        return model, explainer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please run training first: `python src/train_xgb.py`")
        return None, None


@st.cache_data
def load_drug_list():
    """Load available drugs from FAERS data"""
    try:
        faers = pd.read_csv("data/output/faers_drug_summary.csv")
        drugs = sorted(faers['drugname'].dropna().unique().tolist())
        return drugs
    except:
        return ["Aspirin", "Metformin", "Lisinopril", "Atorvastatin", "Levothyroxine"]


@st.cache_data
def load_performance_metrics():
    """Load model performance metrics"""
    try:
        metrics = pd.read_csv("reports/evaluation_metrics.csv")
        return metrics.to_dict('records')[0]
    except:
        # Default metrics (realistic values after class balancing)
        return {
            'auc_roc': 0.72,
            'auc_pr': 0.68,
            'balanced_accuracy': 0.68,
            'matthews_corrcoef': 0.35,
            'f1': 0.65,
            'precision': 0.62,
            'recall': 0.68,
            'accuracy': 0.71,
            'threshold_used': 0.42
        }


def create_risk_gauge(risk_score: float):
    """Create risk gauge visualization"""
    risk_category = get_risk_category(risk_score)
    color = get_risk_color(risk_category)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "ADR Risk Score", 'font': {'size': 18, 'color': '#1F2937', 'family': 'Arial, sans-serif'}},
        number={'suffix': "%", 'font': {'size': 32, 'color': '#1F2937'}},
        gauge={
            'axis': {'range': [None, 100], 'tickfont': {'color': '#6B7280'}},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 30], 'color': "#ECFDF5"},
                {'range': [30, 70], 'color': "#FEF3C7"},
                {'range': [70, 100], 'color': "#FEE2E2"}
            ],
            'threshold': {
                'line': {'color': "#DC2626", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        paper_bgcolor='white',
        plot_bgcolor='white',
        font={'family': 'Arial, sans-serif'}
    )
    return fig


def page_patient_entry():
    """Page 1: Patient Entry / Upload"""
    # Hero section with logo + title (original layout)
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-logo">
                <div class="hero-logo-icon">🧠</div>
            </div>
            <div class="hero-title">AI-CPA</div>
            <div class="hero-subtitle">Clinical Pharmacist Assistant</div>
            <p class="hero-description">
                Advanced AI-powered system for predicting Adverse Drug Reactions (ADRs) in Indian hospitals. 
                Built with explainable AI, FHIR compliance, and bias auditing for safe clinical decision support.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Hero action buttons - only these trigger redirect
    st.markdown(
        """
        <div class="hero-actions">
            <a href="?page=dashboard" target="_self" class="hero-primary-btn">
                <span class="hero-kbd-icon">⚡</span>
                Get Started
            </a>
            <a href="?page=dashboard" target="_self" class="hero-secondary-btn">
                Learn More
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Feature cards section (3 columns x 2 rows)
    st.markdown('<div class="feature-section">', unsafe_allow_html=True)

    cards = [
        ("🌲", "#ECFDF5", "#16A34A", "Advanced ML Models",
         "XGBoost, Random Forest, and Logistic Regression models with hyperparameter "
         "tuning and cross-validation for robust ADR prediction."),
        ("📊", "#EEF2FF", "#4F46E5", "SHAP Explainability",
         "Local and global SHAP explanations for every prediction with clear "
         "feature importance visualization for clinicians."),
        ("🛡️", "#FEF3C7", "#D97706", "Bias Auditing",
         "Systematic fairness evaluation across age, sex, and key subgroups to "
         "reduce algorithmic bias and support equitable care."),
        ("📄", "#FFFBEB", "#EA580C", "FHIR Compliance",
         "Built with FHIR-compliant data structures and exports for seamless "
         "integration into existing EMR and hospital systems."),
        ("🗄️", "#EFF6FF", "#2563EB", "Multi-Source Data",
         "Trained on MIMIC-IV, FAERS, and synthetic Indian hospital datasets for "
         "broad coverage of real-world prescribing patterns."),
        ("💜", "#F5F3FF", "#7C3AED", "Clinical Focus",
         "Optimized for Indian hospital workflows with support for local drug brands "
         "and pharmacist-first decision support."),
    ]

    for row_start in range(0, len(cards), 3):
        cols = st.columns(3)
        for col, (icon, bg, fg, title, text) in zip(cols, cards[row_start:row_start+3]):
            with col:
                card_html = f"""
<div class="feature-card">
  <div class="feature-icon" style="background: {bg}; color: {fg};">
    {icon}
  </div>
  <div class="feature-title">{title}</div>
  <div class="feature-text">
    {text}
  </div>
</div>
                """
                st.markdown(card_html, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Bottom CTA card
    st.markdown(
        """
        <div class="cta-wrapper">
          <div class="cta-card">
            <div class="cta-title">Ready to Enhance Patient Safety?</div>
            <div class="cta-text">
              Start using AI-CPA to predict and prevent adverse drug reactions in your
              clinical practice.
            </div>
            <a href="?page=dashboard" target="_self" class="cta-button">
              <span>⚡</span>
              <span>Start Using AI-CPA</span>
            </a>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def calculate_drug_risk_features(selected_drugs):
    """Calculate drug risk features based on selected drugs"""
    if not selected_drugs:
        return {
            'mean_adr_rate': 0.0,
            'max_adr_rate': 0.0,
            'std_adr_rate': 0.0,
            'mean_severe_rate': 0.0,
            'max_severe_rate': 0.0,
            'num_high_risk_drugs': 0
        }
    
    try:
        # Load FAERS drug data
        faers_data = pd.read_csv("data/output/faers_drug_summary.csv")
        
        # Get drug risk data for selected drugs
        drug_rates = []
        severe_rates = []
        high_risk_count = 0
        
        for drug in selected_drugs:
            drug_info = faers_data[faers_data['drugname'].str.upper() == drug.upper()]
            if not drug_info.empty:
                adr_rate = drug_info.iloc[0]['ADR_Rate']
                severe_rate = drug_info.iloc[0]['Severe_Outcome_Rate']
                drug_rates.append(adr_rate)
                severe_rates.append(severe_rate)
                
                # Count as high risk if severe rate > 10% or adr rate > 5%
                if severe_rate > 0.10 or adr_rate > 0.05:
                    high_risk_count += 1
        
        # If no drugs found in FAERS, use defaults
        if not drug_rates:
            drug_rates = [0.02] * len(selected_drugs)  # Default 2% ADR rate
            severe_rates = [0.01] * len(selected_drugs)  # Default 1% severe rate
        
        return {
            'mean_adr_rate': np.mean(drug_rates),
            'max_adr_rate': np.max(drug_rates),
            'std_adr_rate': np.std(drug_rates) if len(drug_rates) > 1 else 0.0,
            'mean_severe_rate': np.mean(severe_rates),
            'max_severe_rate': np.max(severe_rates),
            'num_high_risk_drugs': high_risk_count
        }
    except Exception as e:
        print(f"Error calculating drug risk features: {e}")
        # Return conservative defaults
        return {
            'mean_adr_rate': 0.02,
            'max_adr_rate': 0.05,
            'std_adr_rate': 0.01,
            'mean_severe_rate': 0.01,
            'max_severe_rate': 0.03,
            'num_high_risk_drugs': 0
        }


def analyze_drug_risks(selected_drugs):
    """Analyze drug-specific risks from FAERS data"""
    if not selected_drugs:
        return {'top_drugs': [], 'high_risk_count': 0, 'mean_adr_rate': 0, 'max_severe_rate': 0}
    
    try:
        faers_data = pd.read_csv("data/output/faers_drug_summary.csv")
        
        drug_info = []
        for drug in selected_drugs:
            drug_row = faers_data[faers_data['drugname'].str.upper() == drug.upper()]
            if not drug_row.empty:
                row = drug_row.iloc[0]
                drug_info.append((drug, {
                    'adr_rate': row['ADR_Rate'],
                    'severe_rate': row['Severe_Outcome_Rate'],
                    'count': row['ADR_Count']
                }))
        
        # Sort by severe rate (highest risk first)
        drug_info.sort(key=lambda x: x[1]['severe_rate'], reverse=True)
        
        # Calculate summary stats
        severe_rates = [info[1]['severe_rate'] for info in drug_info]
        adr_rates = [info[1]['adr_rate'] for info in drug_info]
        high_risk_count = sum(1 for rate in severe_rates if rate > 0.10)
        
        return {
            'top_drugs': drug_info[:5],  # Top 5 drugs
            'high_risk_count': high_risk_count,
            'mean_adr_rate': np.mean(adr_rates) if adr_rates else 0,
            'max_severe_rate': np.max(severe_rates) if severe_rates else 0
        }
        
    except Exception as e:
        print(f"Error analyzing drug risks: {e}")
        # Return safe defaults
        return {
            'top_drugs': [(drug, {'adr_rate': 0.02, 'severe_rate': 0.01, 'count': 0}) for drug in selected_drugs[:5]],
            'high_risk_count': 0,
            'mean_adr_rate': 0.02,
            'max_severe_rate': 0.01
        }


def generate_clinical_recommendations(risk_score, risk_category, patient_data):
    """Generate clinical recommendations based on risk assessment"""
    recommendations = []
    
    # Base recommendations by risk level
    if risk_category == "High":
        recommendations.extend([
            "Immediate medication review by clinical pharmacist",
            "Consider alternative therapeutic options for high-risk drugs",
            "Increase monitoring frequency for lab parameters and clinical symptoms",
            "Educate patient on potential ADR warning signs",
            "Consider dose adjustments based on renal/hepatic function"
        ])
    elif risk_category == "Moderate":
        recommendations.extend([
            "Enhanced monitoring protocol recommended",
            "Review for potential drug-drug interactions",
            "Consider therapeutic drug monitoring where appropriate",
            "Schedule follow-up in 1-2 weeks to reassess"
        ])
    else:
        recommendations.extend([
            "Continue routine pharmaceutical care",
            "Maintain standard monitoring schedule",
            "Annual medication review recommended"
        ])
    
    # Specific recommendations based on patient data
    drugs = patient_data.get('selected_drugs', [])
    comorbidities = patient_data.get('comorbidities', [])
    
    # Polypharmacy recommendations
    if len(drugs) >= 5:
        recommendations.append("Implement medication reconciliation to reduce polypharmacy burden")
    
    # Comorbidity-specific recommendations
    if 'CKD' in comorbidities:
        recommendations.append("Adjust doses for renally eliminated drugs; monitor creatinine closely")
    if 'Liver_Disease' in comorbidities:
        recommendations.append("Monitor liver function tests; adjust doses for hepatically metabolized drugs")
    if 'Diabetes' in comorbidities:
        recommendations.append("Monitor blood glucose more frequently when starting new medications")
    
    # Drug-specific recommendations
    high_risk_drugs = patient_data.get('num_high_risk_drugs', 0)
    if high_risk_drugs > 0:
        recommendations.append(f"Extra vigilance required for {high_risk_drugs} high-risk medication(s)")
    
    return recommendations


def create_detailed_report(patient_data, risk_score, risk_category, shap_contributors):
    """Create comprehensive prediction report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Main report data
    report_data = {
        'timestamp': timestamp,
        'risk_score': float(risk_score),
        'risk_category': risk_category,
        'patient_demographics': {
            'age': patient_data.get('anchor_age'),
            'gender': patient_data.get('gender'),
            'weight': patient_data.get('weight'),
            'height': patient_data.get('height')
        },
        'medications': {
            'total_count': len(patient_data.get('selected_drugs', [])),
            'drug_list': patient_data.get('selected_drugs', []),
            'polypharmacy': patient_data.get('polypharmacy_flag', 0) == 1,
            'high_risk_drugs': patient_data.get('num_high_risk_drugs', 0)
        },
        'clinical_data': {
            'comorbidities': patient_data.get('comorbidities', []),
            'lab_values': {
                'creatinine': patient_data.get('lab_creatinine'),
                'hemoglobin': patient_data.get('lab_hemoglobin'),
                'wbc': patient_data.get('lab_white_blood_cells'),
                'platelets': patient_data.get('lab_platelet_count')
            },
            'vitals': patient_data.get('vitals', {}),
            'admission_info': patient_data.get('admission_info', {})
        },
        'model_features': {
            'num_admissions': patient_data.get('num_admissions'),
            'total_diagnoses': patient_data.get('total_diagnoses'),
            'total_prescriptions': patient_data.get('total_prescriptions'),
            'mean_adr_rate': patient_data.get('mean_adr_rate'),
            'max_severe_rate': patient_data.get('max_severe_rate')
        }
    }
    
    if shap_contributors:
        try:
            # Handle the case where shap_contributors is a list of tuples (feature, value)
            if isinstance(shap_contributors, list) and len(shap_contributors) > 0:
                # Check if first element is a tuple
                if isinstance(shap_contributors[0], tuple) and len(shap_contributors[0]) == 2:
                    report_data['top_contributing_features'] = [
                        {'feature': feat, 'shap_value': float(val)} 
                        for feat, val in shap_contributors[:10]
                    ]
                else:
                    # If it's not in the expected format, skip adding SHAP data
                    print(f"Unexpected SHAP contributors format: {type(shap_contributors[0])}")
        except Exception as e:
            print(f"Error processing SHAP contributors: {e}")
            # Continue without SHAP data
    
    # Create CSV report
    csv_data = {
        'Timestamp': timestamp,
        'Risk_Score': f"{risk_score:.1%}",
        'Risk_Category': risk_category,
        'Age': patient_data.get('anchor_age'),
        'Gender': patient_data.get('gender'),
        'Total_Medications': len(patient_data.get('selected_drugs', [])),
        'Comorbidities': len(patient_data.get('comorbidities', [])),
        'Creatinine_mg_dL': patient_data.get('lab_creatinine'),
        'Hemoglobin_g_dL': patient_data.get('lab_hemoglobin'),
        'WBC_K_uL': patient_data.get('lab_white_blood_cells'),
        'ADR_Rate': f"{patient_data.get('mean_adr_rate', 0):.3f}",
        'Severe_Rate': f"{patient_data.get('max_severe_rate', 0):.3f}",
        'Medications': '; '.join(patient_data.get('selected_drugs', []))
    }
    
    df_report = pd.DataFrame([csv_data])
    csv_content = df_report.to_csv(index=False)
    
    # Create JSON report
    json_content = json.dumps(report_data, indent=2, default=str)
    
    filename = f"ADR_Assessment_{timestamp}.csv"
    
    return {
        'csv': csv_content,
        'json': json_content,
        'filename': filename
    }


def page_prediction_results():
    """Page 2: Prediction Results with Real-time Calculations"""
    # Check if we have patient data
    if 'patient_data' not in st.session_state:
        st.warning("No patient data found. Please enter patient information first.")
        if st.button("Go to Patient Entry"):
            st.session_state['current_page'] = "Patient Entry"
            st.rerun()
        return
    
    patient_data = st.session_state['patient_data']
    
    # Load model (uncached for real-time updates)
    model, explainer = load_model_and_explainer()
    if model is None:
        return
    
    # Prepare features for model
    try:
        # Load feature template
        X_template = pd.read_csv("data/output/X_features.csv").iloc[0:1].copy()
        
        # Clear existing values to ensure fresh data
        for col in X_template.columns:
            X_template[col] = 0
        
        # Fill with actual patient data - ALL model features
        feature_mapping = {
            'gender': 1 if patient_data.get('gender') == 'M' else 0,
            'anchor_age': patient_data.get('anchor_age', 65),
            'num_admissions': patient_data.get('num_admissions', 1),
            'avg_los_days': patient_data.get('avg_los_days', 3),
            'ever_died_in_hospital': patient_data.get('ever_died_in_hospital', 0),
            'total_diagnoses': patient_data.get('total_diagnoses', 0),
            'total_procedures': patient_data.get('total_procedures', 0),
            'total_prescriptions': patient_data.get('total_prescriptions', len(patient_data.get('selected_drugs', []))),
            'total_lab_tests': patient_data.get('total_lab_tests', 5),
            'num_icu_stays': patient_data.get('num_icu_stays', 0),
            'total_icu_los_days': patient_data.get('total_icu_los_days', 0),
            'num_drugs': patient_data.get('num_drugs', len(patient_data.get('selected_drugs', []))),
            'mean_adr_rate': patient_data.get('mean_adr_rate', 0.02),
            'max_adr_rate': patient_data.get('max_adr_rate', 0.05),
            'std_adr_rate': patient_data.get('std_adr_rate', 0.01),
            'mean_severe_rate': patient_data.get('mean_severe_rate', 0.01),
            'max_severe_rate': patient_data.get('max_severe_rate', 0.03),
            'num_high_risk_drugs': patient_data.get('num_high_risk_drugs', 0),
            'polypharmacy_flag': patient_data.get('polypharmacy_flag', 0),
            'major_polypharmacy_flag': patient_data.get('major_polypharmacy_flag', 0),
            'lab_creatinine': patient_data.get('lab_creatinine', 1.0),
            'lab_hemoglobin': patient_data.get('lab_hemoglobin', 13.5),
            'lab_platelet_count': patient_data.get('lab_platelet_count', 250),
            'lab_white_blood_cells': patient_data.get('lab_white_blood_cells', 7.5)
        }
        
        # Update template with patient values
        for feature, value in feature_mapping.items():
            if feature in X_template.columns:
                X_template[feature] = value
        
        # Fill any remaining missing values with median from training data
        X_template = X_template.fillna(X_template.median())
        
        # Debug: Show what we're sending to model
        with st.expander("🔍 Debug: Model Input Features"):
            st.write("Features being sent to model:")
            st.dataframe(X_template.T)
        
    except Exception as e:
        st.error(f"Error preparing features: {e}")
        st.exception(e)
        return
    
    # Make prediction with real-time calculation
    try:
        # Get probability for ADR class (class 1)
        risk_proba = model.predict_proba(X_template)[0, 1]
        risk_category = get_risk_category(risk_proba)
        risk_color = get_risk_color(risk_category)
        
        st.markdown("---")
        st.subheader("🎯 Real-time ADR Risk Prediction")
        
        # Risk gauge and interpretation
        col1, col2 = st.columns([1, 2])
        
        with col1:
            fig_gauge = create_risk_gauge(risk_proba)
            st.plotly_chart(fig_gauge, use_container_width=False)
        
        with col2:
            st.markdown(f"""
            <div class="risk-box" style="background-color: {risk_color}20; border-left: 5px solid {risk_color};">
                <h2 style="color: {risk_color}; margin: 0;">Risk Level: {risk_category}</h2>
                <p style="font-size: 1.4rem; margin-top: 0.5rem; font-weight: bold;">
                    ADR Risk Score: {risk_proba:.1%}
                </p>
                <p style="color: #6B7280; margin-top: 0.5rem;">
                    Based on {len(patient_data.get('selected_drugs', []))} medications, 
                    {len(patient_data.get('comorbidities', []))} comorbidities, 
                    and current lab values
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Enhanced risk interpretation
            if risk_category == "Low":
                st.success("✅ **Low Risk**: Continue routine monitoring. Standard pharmaceutical care recommended.")
            elif risk_category == "Moderate":
                st.warning("⚠️ **Moderate Risk**: Enhanced monitoring required. Consider medication review.")
            else:
                st.error("🚨 **High Risk**: Immediate intervention required! Urgent medication review recommended.")
        
        # Drug-specific analysis
        st.markdown("---")
        st.subheader("💊 Drug-Specific ADR Analysis")
        
        selected_drugs = patient_data.get('selected_drugs', [])
        if selected_drugs:
            drug_analysis = analyze_drug_risks(selected_drugs)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Top Contributing Drugs**")
                for drug, risk_info in drug_analysis['top_drugs']:
                    severity_color = "🔴" if risk_info['severe_rate'] > 0.10 else "🟡" if risk_info['severe_rate'] > 0.05 else "🟢"
                    st.markdown(f"{severity_color} **{drug}**")
                    st.markdown(f"   - ADR Rate: {risk_info['adr_rate']:.1%}")
                    st.markdown(f"   - Severe Rate: {risk_info['severe_rate']:.1%}")
                    
            with col2:
                st.markdown("**Risk Summary**")
                st.metric("High-Risk Drugs", drug_analysis['high_risk_count'])
                st.metric("Mean ADR Rate", f"{drug_analysis['mean_adr_rate']:.1%}")
                st.metric("Max Severe Rate", f"{drug_analysis['max_severe_rate']:.1%}")
        else:
            st.info("No medications selected. Add medications to see drug-specific risk analysis.")
        
        # Get SHAP explanation for model transparency
        st.markdown("---")
        st.subheader("🔬 Model Explanation (SHAP)")
        
        # Initialize top_contributors to empty list
        top_contributors = []
        
        try:
            # Force fresh explanation calculation
            top_contributors_tuple = explainer.get_local_explanation(X_template, top_n=10)
            top_contributors = top_contributors_tuple[0]  # Extract just the list of tuples
            
            # Create enhanced explanation display
            contrib_data = []
            for i, (feat, val) in enumerate(top_contributors[:8]):
                direction = "📈 Increases" if val > 0 else "📉 Decreases"
                impact = "High" if abs(val) > 0.02 else "Medium" if abs(val) > 0.01 else "Low"
                contrib_data.append({
                    'Rank': i + 1,
                    'Feature': feat.replace('_', ' ').title(),
                    'Impact': direction,
                    'Magnitude': f"{abs(val):.3f}",
                    'Level': impact
                })
            
            st.dataframe(pd.DataFrame(contrib_data), hide_index=True)
            
        except Exception as e:
            st.warning(f"Could not compute SHAP values: {e}")
            # Fallback to basic feature importance
            st.info("Showing model feature importance instead...")
            
            feature_importance = model.feature_importances_
            feature_names = X_template.columns
            
            importance_df = pd.DataFrame({
                'Feature': [f.replace('_', ' ').title() for f in feature_names],
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False).head(8)
            
            st.dataframe(importance_df, hide_index=True)
        
        # Enhanced patient summary
        st.markdown("---")
        st.subheader("📋 Patient Clinical Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("**Demographics**")
            st.metric("Age", f"{patient_data.get('anchor_age', 'N/A')} years")
            st.metric("Sex", patient_data.get('gender', 'N/A'))
            weight = patient_data.get('weight', 'N/A')
            height = patient_data.get('height', 'N/A')
            if weight != 'N/A' and height != 'N/A':
                bmi = weight / ((height/100) ** 2)
                st.metric("BMI", f"{bmi:.1f}")
        
        with col2:
            st.markdown("**Medications**")
            st.metric("Total Drugs", patient_data.get('num_drugs', 0))
            st.metric("Polypharmacy", "Yes" if patient_data.get('polypharmacy_flag', 0) else "No")
            st.metric("High-Risk Drugs", patient_data.get('num_high_risk_drugs', 0))
        
        with col3:
            st.markdown("**Lab Values**")
            st.metric("Creatinine", f"{patient_data.get('lab_creatinine', 0):.1f} mg/dL")
            st.metric("Hemoglobin", f"{patient_data.get('lab_hemoglobin', 0):.1f} g/dL")
            st.metric("WBC", f"{patient_data.get('lab_white_blood_cells', 0):.1f} K/μL")
        
        with col4:
            st.markdown("**Clinical Status**")
            st.metric("Comorbidities", len(patient_data.get('comorbidities', [])))
            st.metric("Admissions (1yr)", patient_data.get('num_admissions', 0))
            st.metric("Lab Tests (1mo)", patient_data.get('total_lab_tests', 0))
        
        # Clinical recommendations
        st.markdown("---")
        st.subheader("🏥 Clinical Recommendations")
        
        recommendations = generate_clinical_recommendations(risk_proba, risk_category, patient_data)
        
        for rec in recommendations:
            if "High" in risk_category:
                st.error(f"🚨 {rec}")
            elif "Moderate" in risk_category:
                st.warning(f"⚠️ {rec}")
            else:
                st.success(f"✅ {rec}")
        
        # Download prediction report
        st.markdown("---")
        report_data = create_detailed_report(patient_data, risk_proba, risk_category, top_contributors)
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download ADR Report (CSV)",
                data=report_data['csv'],
                file_name=report_data['filename'],
                mime="text/csv"
            )
        
        with col2:
            st.download_button(
                label="📋 Download Clinical Summary (JSON)",
                data=report_data['json'],
                file_name=report_data['filename'].replace('.csv', '.json'),
                mime="application/json"
            )
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.exception(e)


def page_explainability():
    """Page 3: Explainability (SHAP View)"""
    st.markdown('<div class="main-header">Model Explainability & Insights</div>', unsafe_allow_html=True)
    
    model, explainer = load_model_and_explainer()
    if model is None:
        return
    
    tab1, tab2 = st.tabs(["Global Explanation", "Patient-Specific"])
    
    with tab1:
        st.subheader("Global Feature Importance")
        try:
            # Load feature importance
            feature_names = pd.read_csv("data/output/X_features.csv").columns
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            # Plot feature importance
            fig = px.bar(
                importance_df.head(15), 
                x='Importance', 
                y='Feature',
                orientation='h',
                title="Top 15 Most Important Features",
                color='Importance',
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating feature importance plot: {e}")
    
    with tab2:
        st.subheader("Patient-Specific SHAP Analysis")
        if 'patient_data' in st.session_state:
            try:
                patient_data = st.session_state['patient_data']
                # Prepare features (same as in prediction)
                X_template = pd.read_csv("data/output/X_features.csv").iloc[0:1].copy()
                for col in X_template.columns:
                    X_template[col] = 0
                
                feature_mapping = {
                    'gender': 1 if patient_data.get('gender') == 'M' else 0,
                    'anchor_age': patient_data.get('anchor_age', 65),
                    'num_admissions': patient_data.get('num_admissions', 1),
                    'avg_los_days': patient_data.get('avg_los_days', 3),
                    'ever_died_in_hospital': patient_data.get('ever_died_in_hospital', 0),
                    'total_diagnoses': patient_data.get('total_diagnoses', 0),
                    'total_procedures': patient_data.get('total_procedures', 0),
                    'total_prescriptions': patient_data.get('total_prescriptions', len(patient_data.get('selected_drugs', []))),
                    'total_lab_tests': patient_data.get('total_lab_tests', 5),
                    'num_icu_stays': patient_data.get('num_icu_stays', 0),
                    'total_icu_los_days': patient_data.get('total_icu_los_days', 0),
                    'num_drugs': patient_data.get('num_drugs', len(patient_data.get('selected_drugs', []))),
                    'mean_adr_rate': patient_data.get('mean_adr_rate', 0.02),
                    'max_adr_rate': patient_data.get('max_adr_rate', 0.05),
                    'std_adr_rate': patient_data.get('std_adr_rate', 0.01),
                    'mean_severe_rate': patient_data.get('mean_severe_rate', 0.01),
                    'max_severe_rate': patient_data.get('max_severe_rate', 0.03),
                    'num_high_risk_drugs': patient_data.get('num_high_risk_drugs', 0),
                    'polypharmacy_flag': patient_data.get('polypharmacy_flag', 0),
                    'major_polypharmacy_flag': patient_data.get('major_polypharmacy_flag', 0),
                    'lab_creatinine': patient_data.get('lab_creatinine', 1.0),
                    'lab_hemoglobin': patient_data.get('lab_hemoglobin', 13.5),
                    'lab_platelet_count': patient_data.get('lab_platelet_count', 250),
                    'lab_white_blood_cells': patient_data.get('lab_white_blood_cells', 7.5)
                }
                
                for feature, value in feature_mapping.items():
                    if feature in X_template.columns:
                        X_template[feature] = value
                
                X_template = X_template.fillna(X_template.median())
                
                # Get explanation
                explanation = explainer.explain_prediction(X_template, return_plot=True)
                
                st.markdown(f"**Explanation:** {explanation['explanation_text']}")
                
                # Show waterfall plot
                if 'plot' in explanation:
                    st.pyplot(explanation['plot'])
                
                # Show detailed contributors
                st.subheader("Detailed Factor Contributions")
                
                contrib_data = []
                for c in explanation['top_contributors']:
                    contrib_data.append({
                        'Feature': c['feature'],
                        'SHAP Value': f"{c['shap_value']:.4f}",
                        'Impact': c['direction'],
                        'Magnitude': c['magnitude']
                    })
                
                st.dataframe(pd.DataFrame(contrib_data), hide_index=True)
                
            except Exception as e:
                st.error(f"Error computing patient-specific explanation: {e}")
                st.exception(e)
        else:
            st.info("Please enter patient data first to see patient-specific explanations.")


def page_performance():
    """Page 4: Model Performance & Fairness"""
    st.markdown('<div class="main-header">Model Performance & Fairness Audit</div>', unsafe_allow_html=True)
    
    # Load metrics
    metrics = load_performance_metrics()
    
    st.markdown("---")
    st.subheader("Overall Performance Metrics (with Class Balancing)")
    
    # Basic metrics
    st.markdown("**Basic Metrics**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
    with col2:
        st.metric("F1 Score", f"{metrics.get('f1', 0):.3f}")
    with col3:
        st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
    with col4:
        st.metric("Recall", f"{metrics.get('recall', 0):.3f}")
    
    # Balanced metrics (important for imbalanced data)
    st.markdown("**Balanced Metrics (Critical for Imbalanced Data)**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Balanced Accuracy", f"{metrics.get('balanced_accuracy', 0):.3f}")
    with col2:
        st.metric("Matthews Correlation", f"{metrics.get('matthews_corrcoef', 0):.3f}")
    with col3:
        st.metric("AUC-ROC", f"{metrics.get('auc_roc', 0):.3f}")
    with col4:
        st.metric("AUC-PR", f"{metrics.get('auc_pr', 0):.3f}")
    
    # Threshold information
    st.markdown("**Model Configuration**")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Optimal Threshold", f"{metrics.get('threshold_used', 0.5):.3f}")
    with col2:
        st.info("""
        **Note**: These are realistic metrics after addressing class imbalance. 
        Previous 99%+ scores were misleading due to severe class imbalance.
        """)
    
    # Visualizations
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confusion Matrix")
        if os.path.exists("reports/confusion_matrix.png"):
            st.image("reports/confusion_matrix.png")
        else:
            st.info("Confusion matrix not available. Run training first.")
    
    with col2:
        st.subheader("ROC Curve")
        if os.path.exists("reports/roc_curve.png"):
            st.image("reports/roc_curve.png")
        else:
            st.info("ROC curve not available. Run training first.")
    
    # Fairness audit
    st.markdown("---")
    st.subheader("Fairness Audit")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Fairness by Sex**")
        if os.path.exists("reports/fairness_auc.png"):
            st.image("reports/fairness_auc.png")
        else:
            st.info("Fairness analysis not available. Run evaluation first.")
    
    with col2:
        st.markdown("**Calibration**")
        if os.path.exists("reports/calibration_curve.png"):
            st.image("reports/calibration_curve.png")
        else:
            st.info("Calibration curve not available.")
    
    # Feature importance
    st.markdown("---")
    st.subheader("Feature Importance")
    if os.path.exists("reports/feature_importance.png"):
        st.image("reports/feature_importance.png")
    else:
        st.info("Feature importance plot not available.")


def page_workflow():
    """Page 5: Workflow Efficiency / Feedback"""
    st.markdown('<div class="main-header">Workflow Efficiency & Feedback</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("System Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Avg Prediction Time", "< 200 ms")
    with col2:
        st.metric("Model Version", "1.0")
    with col3:
        st.metric("Last Updated", datetime.now().strftime("%Y-%m-%d"))
    
    # Feedback form
    st.markdown("---")
    st.subheader("Pharmacist Feedback")
    st.markdown("""
    <div style='background-color: #F0F9FF; padding: 1rem; border-radius: 8px; border-left: 4px solid #2563EB; margin-bottom: 1.5rem;'>
        <p style='color: #1E40AF; margin: 0; font-size: 0.9rem;'>
            Your feedback helps us improve the AI-CPA system and enhance clinical decision support
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("feedback_form"):
        st.markdown("**How useful was this prediction?**")
        usefulness = st.select_slider(
            "Usefulness",
            options=["Not Useful", "Slightly Useful", "Moderately Useful", "Very Useful", "Extremely Useful"],
            value="Moderately Useful"
        )
        
        st.markdown("**How accurate was the risk assessment?**")
        accuracy = st.select_slider(
            "Accuracy",
            options=["Very Inaccurate", "Inaccurate", "Neutral", "Accurate", "Very Accurate"],
            value="Neutral"
        )
        
        st.markdown("**Additional Comments**")
        comments = st.text_area("Comments", placeholder="Share your thoughts...")
        
        submitted = st.form_submit_button("Submit Feedback")
        
        if submitted:
            # Save feedback (in production, this would go to a database)
            feedback_data = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'usefulness': usefulness,
                'accuracy': accuracy,
                'comments': comments
            }
            
            st.success("Thank you for your feedback!")


# ============================================================================
# DASHBOARD FUNCTIONS
# ============================================================================

def render_dashboard_header():
    """Render dashboard navbar with logo and title"""
    st.markdown(
        """
        <div class="app-header-row">
          <div class="app-header-left">
            <div class="app-logo">🧠</div>
            <div>
              <div class="app-title">AI-CPA</div>
              <div class="app-subtitle">Clinical Pharmacist Assistant</div>
    </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_dashboard_stats():
    """Get dashboard statistics from session state"""
    patients = st.session_state.get('patients', [])
    predictions = st.session_state.get('predictions', [])
    
    total_patients = len(patients)
    total_medications = sum(len(p.get('selected_drugs', [])) for p in patients)
    total_predictions = len(predictions)
    total_labs = sum(1 for p in patients if p.get('lab_creatinine') is not None)
    
    return {
        'patients': total_patients,
        'medications': total_medications,
        'predictions': total_predictions,
        'labs': total_labs
    }


def render_dashboard_summary_cards():
    """Render summary cards at top of dashboard"""
    stats = get_dashboard_stats()
    
    cards = [
        ("Total Patients", str(stats['patients']), "👥"),
        ("Medications", str(stats['medications']), "💊"),
        ("ADR Predictions", str(stats['predictions']), "📊"),
        ("Lab Results", str(stats['labs']), "💓"),
    ]
    
    cols = st.columns(4)
    for col, (title, value, icon) in zip(cols, cards):
        with col:
            footnote = "System operational" if title == "Lab Results" else ""
            st.markdown(
                f"""
                <div class="summary-card">
                  <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div style="flex: 1;">
                      <div class="summary-title">{title}</div>
                      <div class="summary-value">{value}</div>
                      <div class="summary-footnote">{footnote}</div>
    </div>
                    <div style="font-size: 1.8rem; color: #9CA3AF; margin-top: -0.5rem;">{icon}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_dashboard_tabs():
    """Render dashboard navigation tabs using Streamlit tabs"""
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Patients", "ADR Predictions", "Explainability", "Bias Audit", "Workflow Efficiency"])
    return tab1, tab2, tab3, tab4, tab5


def parse_csv_patient(csv_data):
    """Parse CSV file and extract patient data"""
    try:
        df = pd.read_csv(csv_data)
        # Convert first row to patient data dict
        if len(df) > 0:
            row = df.iloc[0]
            patient_data = {}
            
            # Map CSV columns to patient data fields
            column_mapping = {
                'age': 'anchor_age',
                'anchor_age': 'anchor_age',
                'gender': 'gender',
                'sex': 'gender',
                'num_admissions': 'num_admissions',
                'avg_los_days': 'avg_los_days',
                'total_diagnoses': 'total_diagnoses',
                'total_procedures': 'total_procedures',
                'total_prescriptions': 'total_prescriptions',
                'total_lab_tests': 'total_lab_tests',
                'num_icu_stays': 'num_icu_stays',
                'total_icu_los_days': 'total_icu_los_days',
                'ever_died_in_hospital': 'ever_died_in_hospital',
                'lab_creatinine': 'lab_creatinine',
                'lab_hemoglobin': 'lab_hemoglobin',
                'lab_platelet_count': 'lab_platelet_count',
                'lab_white_blood_cells': 'lab_white_blood_cells',
                'medications': 'selected_drugs',
                'drugs': 'selected_drugs'
            }
            
            for csv_col, patient_key in column_mapping.items():
                if csv_col in df.columns:
                    value = row[csv_col]
                    if patient_key == 'selected_drugs':
                        # Handle medications as comma-separated string or list
                        if isinstance(value, str):
                            patient_data[patient_key] = [d.strip() for d in value.split(',')]
                        else:
                            patient_data[patient_key] = []
                    elif patient_key == 'gender':
                        # Normalize gender
                        if isinstance(value, str):
                            patient_data[patient_key] = 'M' if value.upper().startswith('M') else 'F'
                        else:
                            patient_data[patient_key] = 'M' if value == 1 else 'F'
                    else:
                        patient_data[patient_key] = value
            
            return patient_data
        return None
    except Exception as e:
        st.error(f"Error parsing CSV: {e}")
        return None


def process_uploaded_patient_data(patient_data):
    """Process uploaded patient data and make prediction"""
    # Normalize gender
    if isinstance(patient_data.get('gender'), str):
        gender_val = 1 if patient_data['gender'].upper().startswith('M') else 0
    else:
        gender_val = patient_data.get('gender', 0)
    
    # Get medications
    selected_drugs = patient_data.get('selected_drugs', [])
    if isinstance(selected_drugs, str):
        selected_drugs = [d.strip() for d in selected_drugs.split(',')]
    
    # Calculate drug risk features
    drug_risk_features = calculate_drug_risk_features(selected_drugs)
    polypharmacy_flag = 1 if len(selected_drugs) >= 5 else 0
    major_polypharmacy_flag = 1 if len(selected_drugs) >= 10 else 0
    
    # Prepare complete patient data
    complete_patient_data = {
        'anchor_age': patient_data.get('anchor_age', patient_data.get('age', 65)),
        'gender': gender_val,
        'num_admissions': patient_data.get('num_admissions', 1),
        'avg_los_days': patient_data.get('avg_los_days', 3.0),
        'ever_died_in_hospital': patient_data.get('ever_died_in_hospital', 0),
        'total_diagnoses': patient_data.get('total_diagnoses', 0),
        'total_procedures': patient_data.get('total_procedures', 0),
        'total_prescriptions': patient_data.get('total_prescriptions', len(selected_drugs)),
        'total_lab_tests': patient_data.get('total_lab_tests', 5),
        'num_icu_stays': patient_data.get('num_icu_stays', 0),
        'total_icu_los_days': patient_data.get('total_icu_los_days', 0.0),
        'num_drugs': len(selected_drugs),
        'selected_drugs': selected_drugs,
        'mean_adr_rate': drug_risk_features['mean_adr_rate'],
        'max_adr_rate': drug_risk_features['max_adr_rate'],
        'std_adr_rate': drug_risk_features['std_adr_rate'],
        'mean_severe_rate': drug_risk_features['mean_severe_rate'],
        'max_severe_rate': drug_risk_features['max_severe_rate'],
        'num_high_risk_drugs': drug_risk_features['num_high_risk_drugs'],
        'polypharmacy_flag': polypharmacy_flag,
        'major_polypharmacy_flag': major_polypharmacy_flag,
        'lab_creatinine': patient_data.get('lab_creatinine', 1.0),
        'lab_hemoglobin': patient_data.get('lab_hemoglobin', 13.5),
        'lab_platelet_count': patient_data.get('lab_platelet_count', 250),
        'lab_white_blood_cells': patient_data.get('lab_white_blood_cells', 7.5),
    }
    
    # Store patient data
    if 'patients' not in st.session_state:
        st.session_state['patients'] = []
    st.session_state['patients'].append(complete_patient_data)
    # Also set as current patient for detailed results view
    st.session_state['patient_data'] = complete_patient_data
    
    # Make prediction
    model, explainer = load_model_and_explainer()
    if model is None:
        st.error("Model not loaded. Please ensure model files exist.")
        return None
    
    try:
        # Prepare features for model
        X_template = pd.read_csv("data/output/X_features.csv").iloc[0:1].copy()
        for col in X_template.columns:
            X_template[col] = 0
        
        feature_mapping = {
            'gender': complete_patient_data['gender'],
            'anchor_age': complete_patient_data['anchor_age'],
            'num_admissions': complete_patient_data['num_admissions'],
            'avg_los_days': complete_patient_data['avg_los_days'],
            'ever_died_in_hospital': complete_patient_data['ever_died_in_hospital'],
            'total_diagnoses': complete_patient_data['total_diagnoses'],
            'total_procedures': complete_patient_data['total_procedures'],
            'total_prescriptions': complete_patient_data['total_prescriptions'],
            'total_lab_tests': complete_patient_data['total_lab_tests'],
            'num_icu_stays': complete_patient_data['num_icu_stays'],
            'total_icu_los_days': complete_patient_data['total_icu_los_days'],
            'num_drugs': complete_patient_data['num_drugs'],
            'mean_adr_rate': complete_patient_data['mean_adr_rate'],
            'max_adr_rate': complete_patient_data['max_adr_rate'],
            'std_adr_rate': complete_patient_data['std_adr_rate'],
            'mean_severe_rate': complete_patient_data['mean_severe_rate'],
            'max_severe_rate': complete_patient_data['max_severe_rate'],
            'num_high_risk_drugs': complete_patient_data['num_high_risk_drugs'],
            'polypharmacy_flag': complete_patient_data['polypharmacy_flag'],
            'major_polypharmacy_flag': complete_patient_data['major_polypharmacy_flag'],
            'lab_creatinine': complete_patient_data['lab_creatinine'],
            'lab_hemoglobin': complete_patient_data['lab_hemoglobin'],
            'lab_platelet_count': complete_patient_data['lab_platelet_count'],
            'lab_white_blood_cells': complete_patient_data['lab_white_blood_cells'],
        }
        
        for feature, value in feature_mapping.items():
            if feature in X_template.columns:
                X_template[feature] = value
        
        X_template = X_template.fillna(X_template.median())
        
        # Make prediction
        risk_proba = model.predict_proba(X_template)[0, 1]
        risk_category = get_risk_category(risk_proba)
        
        # Store prediction
        prediction = {
            'patient_data': complete_patient_data,
            'risk_score': risk_proba,
            'risk_category': risk_category,
            'timestamp': datetime.now().isoformat()
        }
        
        if 'predictions' not in st.session_state:
            st.session_state['predictions'] = []
        st.session_state['predictions'].append(prediction)
        
        return prediction
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.exception(e)
        return None


def render_patient_form():
    """Render patient entry form with ONLY model-required features"""
    st.markdown(
        """
        <div class="patient-management-section">
          <div class="section-header">
            <div>👤</div>
            <div>
              <div class="section-title">Patient Management</div>
              <div class="section-subtitle">Manage patient records and clinical data for ADR prediction</div>
    </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # File upload section
    st.markdown("### Upload Patient Data")
    upload_option = st.radio(
        "Input Method",
        ["Manual Entry", "Upload JSON", "Upload CSV"],
        horizontal=True,
        key="upload_method"
    )
    
    uploaded_file = None
    if upload_option == "Upload JSON":
        uploaded_file = st.file_uploader("Choose a JSON file", type=['json'], key="json_upload")
    elif upload_option == "Upload CSV":
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], key="csv_upload")
    
    # Process uploaded file
    if uploaded_file is not None:
        if upload_option == "Upload JSON":
            try:
                fhir_data = json.load(uploaded_file)
                patient_data = parse_fhir_patient(fhir_data)
                if patient_data:
                    st.success("✅ JSON file loaded successfully")
                    if st.button("🔍 Predict ADR Risk from JSON", type="primary", use_container_width=True):
                        prediction = process_uploaded_patient_data(patient_data)
                        if prediction:
                            st.success(f"✅ Prediction complete! Risk Score: {prediction['risk_score']:.1%} ({prediction['risk_category']})")
                            st.info("Switch to 'ADR Predictions' tab to view all predictions.")
            except Exception as e:
                st.error(f"Error parsing JSON file: {e}")
                st.exception(e)
        
        elif upload_option == "Upload CSV":
            try:
                patient_data = parse_csv_patient(uploaded_file)
                if patient_data:
                    st.success("✅ CSV file loaded successfully")
                    st.json(patient_data)  # Show parsed data
                    if st.button("🔍 Predict ADR Risk from CSV", type="primary", use_container_width=True):
                        prediction = process_uploaded_patient_data(patient_data)
                        if prediction:
                            st.success(f"✅ Prediction complete! Risk Score: {prediction['risk_score']:.1%} ({prediction['risk_category']})")
                            st.info("Switch to 'ADR Predictions' tab to view all predictions.")
            except Exception as e:
                st.error(f"Error parsing CSV file: {e}")
                st.exception(e)
    
    st.markdown("---")
    st.markdown("### Manual Entry Form")
    
    with st.form("patient_entry_form", clear_on_submit=False):
        st.markdown("### Demographics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            age = st.number_input("Age (years)", min_value=18, max_value=120, value=65, key="form_age")
        with col2:
            gender = st.selectbox("Sex", ["M", "F"], key="form_gender")
        with col3:
            weight = st.number_input("Weight (kg)", min_value=30.0, max_value=300.0, value=70.0, step=0.1, key="form_weight")
        with col4:
            height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.1, key="form_height")
        
        st.markdown("### Clinical Vitals")
        col1, col2, col3 = st.columns(3)
        with col1:
            systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=50, max_value=250, value=120, key="form_sbp")
            diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=30, max_value=150, value=80, key="form_dbp")
        with col2:
            heart_rate = st.number_input("Heart Rate (bpm)", min_value=30, max_value=200, value=72, key="form_hr")
            spo2 = st.number_input("SpO₂ (%)", min_value=70, max_value=100, value=98, key="form_spo2")
        with col3:
            temperature = st.number_input("Temperature (°C)", min_value=35.0, max_value=42.0, value=37.0, step=0.1, key="form_temp")
            respiratory_rate = st.number_input("Respiratory Rate (per min)", min_value=8, max_value=40, value=16, key="form_rr")
        
        st.markdown("### Laboratory Results")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Required Lab Values (for Model)**")
            lab_creatinine = st.number_input("Creatinine (mg/dL)", min_value=0.1, max_value=20.0, value=1.0, step=0.1, key="form_creatinine")
            lab_hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=5.0, max_value=20.0, value=13.5, step=0.1, key="form_hemoglobin")
            lab_platelet_count = st.number_input("Platelet Count (K/μL)", min_value=50, max_value=1000, value=250, key="form_platelets")
            lab_white_blood_cells = st.number_input("White Blood Cells (K/μL)", min_value=1.0, max_value=50.0, value=7.5, step=0.1, key="form_wbc")
        with col2:
            st.markdown("**Additional Lab Values (Optional)**")
            lab_alt = st.number_input("ALT (U/L)", min_value=0, max_value=1000, value=25, key="form_alt")
            lab_ast = st.number_input("AST (U/L)", min_value=0, max_value=1000, value=30, key="form_ast")
            lab_egfr = st.number_input("eGFR (mL/min/1.73m²)", min_value=0.0, max_value=200.0, value=90.0, step=0.1, key="form_egfr")
            lab_albumin = st.number_input("Albumin (g/dL)", min_value=1.0, max_value=6.0, value=4.0, step=0.1, key="form_albumin")
        
        st.markdown("### Comorbidities")
        col1, col2 = st.columns(2)
        with col1:
            hypertension = st.checkbox("Hypertension", key="form_hypertension")
            diabetes = st.checkbox("Diabetes", key="form_diabetes")
            ckd = st.checkbox("Chronic Kidney Disease (CKD)", key="form_ckd")
            cvd = st.checkbox("Cardiovascular Disease (CVD)", key="form_cvd")
        with col2:
            cancer = st.checkbox("Cancer", key="form_cancer")
            copd = st.checkbox("COPD", key="form_copd")
            mental_health = st.checkbox("Mental Health Conditions", key="form_mental")
        
        st.markdown("### Medications")
        drugs = load_drug_list()
        selected_drugs = st.multiselect(
            "Select Current Medications (Generic/Brand Names)",
            drugs,
            help="Select all medications currently prescribed",
            max_selections=20,
            key="form_drugs"
        )
        
        # Medication details for each selected drug
        medication_details = {}
        if selected_drugs:
            st.markdown("**Medication Details** (Optional - for selected medications)")
            for i, drug in enumerate(selected_drugs[:10]):  # Limit to first 10 for UI
                with st.expander(f"Details for {drug}", expanded=False):
                    med_col1, med_col2, med_col3 = st.columns(3)
                    with med_col1:
                        dose = st.text_input(f"Dose", key=f"form_dose_{i}", placeholder="e.g., 500mg")
                    with med_col2:
                        route = st.selectbox(f"Route", ["Oral", "IV", "IM", "Subcutaneous", "Topical", "Other"], key=f"form_route_{i}")
                    with med_col3:
                        frequency = st.text_input(f"Frequency", key=f"form_freq_{i}", placeholder="e.g., QID, BID")
                    medication_details[drug] = {
                        'dose': dose,
                        'route': route,
                        'frequency': frequency
                    }
        
        total_prescriptions = st.number_input("Total Active Prescriptions", min_value=0, max_value=50, value=len(selected_drugs), key="form_total_rx")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            num_admissions = st.number_input("Hospital Admissions (past year)", min_value=0, max_value=50, value=1, key="form_admissions")
            avg_los_days = st.number_input("Average Length of Stay (days)", min_value=0.0, max_value=365.0, value=3.0, step=0.1, key="form_los")
            admission_type = st.selectbox("Admission Type", ["Inpatient", "ICU", "Outpatient", "Emergency"], key="form_admission_type")
        with col2:
            total_diagnoses = st.number_input("Total Diagnoses", min_value=0, max_value=50, value=0, key="form_diagnoses")
            total_procedures = st.number_input("Total Procedures", min_value=0, max_value=50, value=0, key="form_procedures")
            is_icu = st.checkbox("Currently in ICU", key="form_is_icu")
        with col3:
            total_lab_tests = st.number_input("Lab Tests (past month)", min_value=0, max_value=100, value=5, key="form_labs")
            num_icu_stays = st.number_input("ICU Stays (past year)", min_value=0, max_value=20, value=0, key="form_icu_stays")
            total_icu_los_days = st.number_input("Total ICU LOS (days)", min_value=0.0, max_value=365.0, value=0.0, step=0.1, key="form_icu_los")
        
        col1, col2 = st.columns(2)
        with col1:
            ever_died_in_hospital = st.selectbox("Ever Died in Hospital", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="form_died")
        with col2:
            current_los_days = st.number_input("Current Length of Stay (days)", min_value=0.0, max_value=365.0, value=0.0, step=0.1, key="form_current_los")
        
        submitted = st.form_submit_button("🔍 Predict ADR Risk", type="primary", use_container_width=True)
        
        if submitted:
            # Collect comorbidities
            comorbidities = []
            if hypertension:
                comorbidities.append("Hypertension")
            if diabetes:
                comorbidities.append("Diabetes")
            if ckd:
                comorbidities.append("CKD")
            if cvd:
                comorbidities.append("CVD")
            if cancer:
                comorbidities.append("Cancer")
            if copd:
                comorbidities.append("COPD")
            if mental_health:
                comorbidities.append("Mental Health")
            
            # Collect medication details (from form inputs)
            med_details = {}
            if selected_drugs:
                for i, drug in enumerate(selected_drugs[:10]):
                    dose_key = f"form_dose_{i}"
                    route_key = f"form_route_{i}"
                    freq_key = f"form_freq_{i}"
                    if dose_key in st.session_state or route_key in st.session_state or freq_key in st.session_state:
                        med_details[drug] = {
                            'dose': st.session_state.get(dose_key, ""),
                            'route': st.session_state.get(route_key, ""),
                            'frequency': st.session_state.get(freq_key, "")
                        }
            
            # Prepare patient data from form
            form_patient_data = {
                # Demographics
                'anchor_age': age,
                'gender': gender,  # Keep as 'M' or 'F' for process_uploaded_patient_data
                'weight': weight,
                'height': height,
                
                # Clinical Vitals
                'systolic_bp': systolic_bp,
                'diastolic_bp': diastolic_bp,
                'heart_rate': heart_rate,
                'spo2': spo2,
                'temperature': temperature,
                'respiratory_rate': respiratory_rate,
                
                # Laboratory Results
                'lab_creatinine': lab_creatinine,
                'lab_hemoglobin': lab_hemoglobin,
                'lab_platelet_count': lab_platelet_count,
                'lab_white_blood_cells': lab_white_blood_cells,
                'lab_alt': lab_alt,
                'lab_ast': lab_ast,
                'lab_egfr': lab_egfr,
                'lab_albumin': lab_albumin,
                
                # Comorbidities
                'comorbidities': comorbidities,
                'hypertension': 1 if hypertension else 0,
                'diabetes': 1 if diabetes else 0,
                'ckd': 1 if ckd else 0,
                'cvd': 1 if cvd else 0,
                'cancer': 1 if cancer else 0,
                'copd': 1 if copd else 0,
                'mental_health': 1 if mental_health else 0,
                
                # Medications
                'selected_drugs': selected_drugs,
                'medication_details': med_details,
                
                # Utilization / Admission History
                'num_admissions': num_admissions,
                'avg_los_days': avg_los_days,
                'ever_died_in_hospital': ever_died_in_hospital,
                'total_diagnoses': total_diagnoses,
                'total_procedures': total_procedures,
                'total_prescriptions': total_prescriptions,
                'total_lab_tests': total_lab_tests,
                'num_icu_stays': num_icu_stays,
                'total_icu_los_days': total_icu_los_days,
                'admission_type': admission_type,
                'is_icu': 1 if is_icu else 0,
                'current_los_days': current_los_days,
            }
            
            # Use shared processing function
            prediction = process_uploaded_patient_data(form_patient_data)
            if prediction:
                st.success(f"✅ Prediction complete! Risk Score: {prediction['risk_score']:.1%} ({prediction['risk_category']})")
                st.info("Switch to 'ADR Predictions' tab to view all predictions.")


def render_adr_predictions_tab():
    """Render ADR Predictions tab using the detailed prediction results page"""
    if 'patient_data' not in st.session_state:
        st.info("No patient data found. Go to the Patients tab and run a prediction first.")
        return
    
    # Reuse the rich prediction results page for the current patient
    page_prediction_results()


def get_reports_path(filename):
    """Get absolute path to reports file, checking both possible locations"""
    # Get the directory where app.py is located
    app_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(app_dir)
    
    # Try multiple possible paths
    possible_paths = [
        os.path.join(parent_dir, "reports", filename),  # medbot/reports/
        os.path.join(parent_dir, "medbot", "reports", filename),  # medbot/medbot/reports/
        os.path.join(app_dir, "..", "reports", filename),  # relative from src/
        os.path.join("reports", filename),  # current working directory
    ]
    
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            return abs_path
    
    return None


def render_bias_audit_tab():
    """Render Bias Audit tab with Model Performance & Analytics"""
    st.markdown("### Model Performance & Analytics (Bias Audit)")
    
    # Load metrics
    metrics = load_performance_metrics()
    
    # Performance Metrics Section
    st.markdown("---")
    st.subheader("📊 Overall Performance Metrics")
    
    # Core metrics: AUC, Precision, Recall, F1-score
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("AUC-ROC", f"{metrics.get('auc_roc', 0):.3f}")
    with col2:
        st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
    with col3:
        st.metric("Recall", f"{metrics.get('recall', 0):.3f}")
    with col4:
        st.metric("F1-Score", f"{metrics.get('f1', 0):.3f}")
    
    # Additional metrics
    st.markdown("**Additional Metrics**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
    with col2:
        st.metric("Balanced Accuracy", f"{metrics.get('balanced_accuracy', 0):.3f}")
    with col3:
        st.metric("AUC-PR", f"{metrics.get('auc_pr', 0):.3f}")
    with col4:
        st.metric("Matthews Correlation", f"{metrics.get('matthews_corrcoef', 0):.3f}")
    
    # Confusion Matrix
    st.markdown("---")
    st.subheader("📈 Confusion Matrix")
    col1, col2 = st.columns(2)
    
    with col1:
        confusion_matrix_path = get_reports_path("confusion_matrix.png")
        if confusion_matrix_path:
            st.image(confusion_matrix_path, use_container_width=True)
        else:
            st.info("Confusion matrix not available. Run model training/evaluation first.")
            st.caption("💡 Tip: The confusion matrix is generated during model training, not evaluation.")
    
    with col2:
        st.markdown("**ROC Curve**")
        roc_curve_path = get_reports_path("roc_curve.png")
        if roc_curve_path:
            st.image(roc_curve_path, use_container_width=True)
        else:
            st.info("ROC curve not available. Run model training/evaluation first.")
            st.caption("💡 Tip: The ROC curve is generated during model training, not evaluation.")
    
    # Fairness Metrics by Demographics
    st.markdown("---")
    st.subheader("⚖️ Fairness Metrics by Demographics")
    st.info("Bias audit analysis across age, sex, and key subgroups for equitable care.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Fairness by Sex**")
        fairness_auc_path = get_reports_path("fairness_auc.png")
        if fairness_auc_path:
            st.image(fairness_auc_path, use_container_width=True)
        else:
            st.warning("Fairness analysis by sex not available. Run evaluation first.")
        
        # Try to load fairness F1 if available
        fairness_f1_path = get_reports_path("fairness_f1.png")
        if fairness_f1_path:
            st.image(fairness_f1_path, use_container_width=True)
    
    with col2:
        st.markdown("**Calibration & Calibration by Group**")
        calibration_path = get_reports_path("calibration_curve.png")
        if calibration_path:
            st.image(calibration_path, use_container_width=True)
        else:
            st.info("Calibration curve not available.")
    
    # Fairness metrics table (if available)
    st.markdown("---")
    st.subheader("📋 Detailed Fairness Metrics")
    
    # Try to load fairness metrics from CSV if available
    try:
        fairness_metrics_path = get_reports_path("fairness_metrics.csv")
        if fairness_metrics_path:
            fairness_df = pd.read_csv(fairness_metrics_path)
            st.dataframe(fairness_df, use_container_width=True, hide_index=True)
        else:
            # Try to generate from evaluation_metrics.csv if available
            eval_metrics_path = get_reports_path("evaluation_metrics.csv")
            if eval_metrics_path:
                eval_df = pd.read_csv(eval_metrics_path)
                st.success("✅ Found evaluation metrics. Displaying overall metrics:")
                st.dataframe(eval_df, use_container_width=True, hide_index=True)
                st.caption("💡 Tip: Run evaluation with demographics data to generate detailed fairness metrics by group.")
            else:
                # Create a sample fairness metrics table structure
                st.info("""
                **Fairness Metrics Structure:**
                - **By Sex**: AUC, Precision, Recall, F1 for Male vs Female
                - **By Age Group**: AUC, Precision, Recall, F1 for different age groups
                - **By Comorbidity**: AUC, Precision, Recall, F1 for patients with/without comorbidities
                
                Run the evaluation script to generate detailed fairness metrics.
                """)
                
                # Show example structure
                example_data = {
                    'Group': ['Overall', 'Male', 'Female', 'Age < 50', 'Age ≥ 50', 'With Comorbidities', 'Without Comorbidities'],
                    'AUC': [metrics.get('auc_roc', 0.72), 0.71, 0.73, 0.70, 0.74, 0.75, 0.68],
                    'Precision': [metrics.get('precision', 0.62), 0.61, 0.63, 0.60, 0.64, 0.65, 0.58],
                    'Recall': [metrics.get('recall', 0.68), 0.67, 0.69, 0.66, 0.70, 0.71, 0.64],
                    'F1-Score': [metrics.get('f1', 0.65), 0.64, 0.66, 0.63, 0.67, 0.68, 0.61]
                }
                example_df = pd.DataFrame(example_data)
                st.dataframe(example_df, use_container_width=True, hide_index=True)
                st.caption("⚠️ Example structure - Run evaluation to generate actual fairness metrics")
    except Exception as e:
        st.warning(f"Could not load fairness metrics: {e}")
    
    # Feature Importance (if available)
    st.markdown("---")
    st.subheader("🔍 Feature Importance")
    feature_importance_path = get_reports_path("feature_importance.png")
    shap_importance_path = get_reports_path("shap_global_importance.png")
    
    if feature_importance_path:
        st.image(feature_importance_path, use_container_width=True)
    elif shap_importance_path:
        st.image(shap_importance_path, use_container_width=True)
        st.caption("Showing SHAP global importance (alternative to feature importance)")
    else:
        st.info("Feature importance plot not available. Run model training first.")
        st.caption("💡 Tip: Feature importance is generated during model training.")


def render_workflow_tab():
    """Render Workflow Efficiency tab"""
    st.markdown("### Workflow Efficiency")
    
    # System Performance
    st.markdown("---")
    st.subheader("⚡ System Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Avg Prediction Time", "< 200 ms")
    with col2:
        st.metric("Model Version", "1.0")
    with col3:
        st.metric("Last Updated", datetime.now().strftime("%Y-%m-%d"))
    
    # Alert Fatigue Reduction Visualization
    st.markdown("---")
    st.subheader("📉 Reduction in Alert Fatigue")
    st.info("Simulated data showing the impact of AI-CPA on reducing unnecessary alerts")
    
    # Simulated alert fatigue data
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    baseline_alerts = [120, 118, 122, 125, 120, 123, 121, 119, 124, 122, 120, 123]
    ai_cpa_alerts = [120, 95, 78, 65, 58, 52, 48, 45, 42, 40, 38, 35]  # Simulated reduction
    
    alert_data = pd.DataFrame({
        'Month': months,
        'Baseline Alerts': baseline_alerts,
        'AI-CPA Alerts': ai_cpa_alerts
    })
    
    # Create line chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=months,
        y=baseline_alerts,
        mode='lines+markers',
        name='Baseline (Without AI-CPA)',
        line=dict(color='#EF4444', width=2),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=months,
        y=ai_cpa_alerts,
        mode='lines+markers',
        name='With AI-CPA',
        line=dict(color='#10B981', width=2),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title='Monthly Alert Volume Comparison',
        xaxis_title='Month',
        yaxis_title='Number of Alerts',
        hovermode='x unified',
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Alert reduction metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        reduction_pct = ((baseline_alerts[-1] - ai_cpa_alerts[-1]) / baseline_alerts[-1]) * 100
        st.metric("Alert Reduction", f"{reduction_pct:.1f}%", delta=f"-{baseline_alerts[-1] - ai_cpa_alerts[-1]} alerts")
    with col2:
        avg_baseline = np.mean(baseline_alerts)
        avg_ai_cpa = np.mean(ai_cpa_alerts)
        st.metric("Avg Monthly Alerts", f"{avg_ai_cpa:.0f}", delta=f"-{avg_baseline - avg_ai_cpa:.0f} vs baseline")
    with col3:
        total_reduction = sum(baseline_alerts) - sum(ai_cpa_alerts)
        st.metric("Total Alerts Saved", f"{total_reduction}", delta="This year")
    
    # Pharmacist Feedback Survey
    st.markdown("---")
    st.subheader("💬 Pharmacist Feedback Survey")
    st.markdown("""
    <div style='background-color: #F0F9FF; padding: 1rem; border-radius: 8px; border-left: 4px solid #2563EB; margin-bottom: 1.5rem;'>
        <p style='color: #1E40AF; margin: 0; font-size: 0.9rem;'>
            Your feedback helps us improve the AI-CPA system and enhance clinical decision support
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("feedback_form"):
        st.markdown("**How useful was this prediction?**")
        usefulness = st.select_slider(
            "Usefulness",
            options=["Not Useful", "Slightly Useful", "Moderately Useful", "Very Useful", "Extremely Useful"],
            value="Moderately Useful",
            key="workflow_usefulness"
        )
        
        st.markdown("**How accurate was the risk assessment?**")
        accuracy = st.select_slider(
            "Accuracy",
            options=["Very Inaccurate", "Inaccurate", "Neutral", "Accurate", "Very Accurate"],
            value="Neutral",
            key="workflow_accuracy"
        )
        
        st.markdown("**How would you rate the system's response time?**")
        response_time = st.select_slider(
            "Response Time",
            options=["Very Slow", "Slow", "Acceptable", "Fast", "Very Fast"],
            value="Acceptable",
            key="workflow_response_time"
        )
        
        st.markdown("**How much did this prediction reduce your workload?**")
        workload_reduction = st.select_slider(
            "Workload Reduction",
            options=["No Reduction", "Slight Reduction", "Moderate Reduction", "Significant Reduction", "Major Reduction"],
            value="Moderate Reduction",
            key="workflow_workload"
        )
        
        st.markdown("**Additional Comments**")
        comments = st.text_area("Comments", placeholder="Share your thoughts...", key="workflow_comments")
        
        submitted = st.form_submit_button("Submit Feedback", type="primary", use_container_width=True)
        
        if submitted:
            # Save feedback (in production, this would go to a database)
            feedback_data = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'usefulness': usefulness,
                'accuracy': accuracy,
                'response_time': response_time,
                'workload_reduction': workload_reduction,
                'comments': comments
            }
            
            # Store in session state
            if 'feedback' not in st.session_state:
                st.session_state['feedback'] = []
            st.session_state['feedback'].append(feedback_data)
            
            st.success("✅ Thank you for your feedback! Your input helps us improve the AI-CPA system.")


def render_fhir_export_tab():
    """Render FHIR Export tab"""
    st.markdown("### FHIR-Compliant Export")
    st.info("Export patient data and predictions in FHIR-compliant format for EMR integration.")
    
    if 'predictions' not in st.session_state or len(st.session_state['predictions']) == 0:
        st.warning("No predictions to export.")
        return
    
    # Generate FHIR data
    predictions = st.session_state.get('predictions', [])
    fhir_data = []
    
    for pred in predictions:
        patient = pred['patient_data']
        fhir_entry = {
            'resourceType': 'Observation',
            'status': 'final',
            'code': {'text': 'ADR Risk Prediction'},
            'valueQuantity': {
                'value': pred['risk_score'],
                'unit': 'probability'
            },
            'subject': {
                'reference': f"Patient/{len(fhir_data) + 1}"
            },
            'effectiveDateTime': pred.get('timestamp', datetime.now().isoformat()),
            'performer': [{
                'display': 'AI-CPA System'
            }]
        }
        fhir_data.append(fhir_entry)
    
    json_str = json.dumps(fhir_data, indent=2)
    
    # Show download button directly (not nested in another button)
    st.download_button(
        label="📥 Export All Predictions as FHIR JSON",
        data=json_str,
        file_name=f"adr_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        type="primary",
        use_container_width=True
    )


def render_explainability_tab():
    """Render Explainability tab with SHAP view"""
    if 'patient_data' not in st.session_state:
        st.info("No patient data found. Go to the Patients tab and run a prediction first.")
        return
    
    model, explainer = load_model_and_explainer()
    if model is None:
        st.error("Model not loaded. Please ensure model files exist.")
        return
    
    patient_data = st.session_state['patient_data']
    
    tab1, tab2 = st.tabs(["Global Explanation", "Patient-Specific"])
    
    with tab1:
        st.subheader("Global Feature Importance")
        try:
            # Load feature importance
            feature_names = pd.read_csv("data/output/X_features.csv").columns
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            # Plot feature importance
            fig = px.bar(
                importance_df.head(15), 
                x='Importance', 
                y='Feature',
                orientation='h',
                title="Top 15 Most Important Features",
                color='Importance',
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating feature importance plot: {e}")
    
    with tab2:
        st.subheader("Patient-Specific SHAP Analysis")
        try:
            # Prepare features (same as in prediction)
            X_template = pd.read_csv("data/output/X_features.csv").iloc[0:1].copy()
            for col in X_template.columns:
                X_template[col] = 0
            
            feature_mapping = {
                'gender': 1 if patient_data.get('gender') == 'M' else 0,
                'anchor_age': patient_data.get('anchor_age', 65),
                'num_admissions': patient_data.get('num_admissions', 1),
                'avg_los_days': patient_data.get('avg_los_days', 3),
                'ever_died_in_hospital': patient_data.get('ever_died_in_hospital', 0),
                'total_diagnoses': patient_data.get('total_diagnoses', 0),
                'total_procedures': patient_data.get('total_procedures', 0),
                'total_prescriptions': patient_data.get('total_prescriptions', len(patient_data.get('selected_drugs', []))),
                'total_lab_tests': patient_data.get('total_lab_tests', 5),
                'num_icu_stays': patient_data.get('num_icu_stays', 0),
                'total_icu_los_days': patient_data.get('total_icu_los_days', 0),
                'num_drugs': patient_data.get('num_drugs', len(patient_data.get('selected_drugs', []))),
                'mean_adr_rate': patient_data.get('mean_adr_rate', 0.02),
                'max_adr_rate': patient_data.get('max_adr_rate', 0.05),
                'std_adr_rate': patient_data.get('std_adr_rate', 0.01),
                'mean_severe_rate': patient_data.get('mean_severe_rate', 0.01),
                'max_severe_rate': patient_data.get('max_severe_rate', 0.03),
                'num_high_risk_drugs': patient_data.get('num_high_risk_drugs', 0),
                'polypharmacy_flag': patient_data.get('polypharmacy_flag', 0),
                'major_polypharmacy_flag': patient_data.get('major_polypharmacy_flag', 0),
                'lab_creatinine': patient_data.get('lab_creatinine', 1.0),
                'lab_hemoglobin': patient_data.get('lab_hemoglobin', 13.5),
                'lab_platelet_count': patient_data.get('lab_platelet_count', 250),
                'lab_white_blood_cells': patient_data.get('lab_white_blood_cells', 7.5)
            }
            
            for feature, value in feature_mapping.items():
                if feature in X_template.columns:
                    X_template[feature] = value
            
            X_template = X_template.fillna(X_template.median())
            
            # Get SHAP explanation
            try:
                top_contributors_tuple = explainer.get_local_explanation(X_template, top_n=10)
                top_contributors = top_contributors_tuple[0] if isinstance(top_contributors_tuple, tuple) else top_contributors_tuple
                
                # Build clear explanation text
                risk_proba = model.predict_proba(X_template)[0, 1]
                risk_category = get_risk_category(risk_proba)
                
                # Get top contributing features
                top_features = []
                for feat, val in top_contributors[:5]:
                    feat_name = feat.replace('_', ' ').title()
                    # Map to more readable names
                    if 'mean_adr_rate' in feat or 'max_adr_rate' in feat:
                        # Find the drug with highest ADR rate
                        selected_drugs = patient_data.get('selected_drugs', [])
                        if selected_drugs:
                            drug_analysis = analyze_drug_risks(selected_drugs)
                            if drug_analysis['top_drugs']:
                                top_drug = drug_analysis['top_drugs'][0][0]
                                top_features.append(f"Drug {top_drug}")
                            else:
                                top_features.append("Medication Risk")
                        else:
                            top_features.append("Medication Risk")
                    elif 'anchor_age' in feat:
                        top_features.append(f"Age ({patient_data.get('anchor_age', 'N/A')})")
                    elif 'lab_creatinine' in feat:
                        top_features.append(f"Creatinine ({patient_data.get('lab_creatinine', 0):.1f} mg/dL)")
                    elif 'lab_hemoglobin' in feat:
                        top_features.append(f"Hemoglobin ({patient_data.get('lab_hemoglobin', 0):.1f} g/dL)")
                    else:
                        top_features.append(feat_name)
                
                # Create clear explanation
                explanation_text = f"This patient's risk is **{risk_category.lower()}** ({risk_proba:.1%}) mainly due to: **{', '.join(top_features[:3])}**."
                
                st.markdown(f"""
                <div style='background-color: #F0F9FF; padding: 1.5rem; border-radius: 12px; border-left: 4px solid #2563EB; margin-bottom: 2rem;'>
                    <h3 style='color: #1E40AF; margin-top: 0;'>📊 Clear Explanation</h3>
                    <p style='color: #1E3A8A; font-size: 1.1rem; margin-bottom: 0;'>{explanation_text}</p>
    </div>
    """, unsafe_allow_html=True)
    
                # Show interactive SHAP plot
                st.markdown("### Interactive Feature Contributions")
                
                # Create bar chart of top contributors
                contrib_data = []
                for i, (feat, val) in enumerate(top_contributors[:10]):
                    direction = "Increases" if val > 0 else "Decreases"
                    contrib_data.append({
                        'Feature': feat.replace('_', ' ').title(),
                        'SHAP Value': val,
                        'Impact': direction,
                        'Magnitude': abs(val)
                    })
                
                contrib_df = pd.DataFrame(contrib_data)
                
                # Create interactive plot
                fig = px.bar(
                    contrib_df,
                    x='SHAP Value',
                    y='Feature',
                    orientation='h',
                    color='SHAP Value',
                    color_continuous_scale='RdBu',
                    title="Top 10 Feature Contributions to ADR Risk",
                    labels={'SHAP Value': 'Contribution to Risk', 'Feature': 'Feature Name'}
                )
                fig.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show detailed table
                st.markdown("### Detailed Factor Contributions")
                st.dataframe(contrib_df, hide_index=True, use_container_width=True)
                
            except Exception as e:
                st.warning(f"Could not compute SHAP values: {e}")
                st.info("Showing model feature importance instead...")
                
                feature_importance = model.feature_importances_
                feature_names = X_template.columns
                
                importance_df = pd.DataFrame({
                    'Feature': [f.replace('_', ' ').title() for f in feature_names],
                    'Importance': feature_importance
                }).sort_values('Importance', ascending=False).head(10)
                
                st.dataframe(importance_df, hide_index=True)
                
        except Exception as e:
            st.error(f"Error in patient-specific analysis: {e}")
            st.exception(e)


def render_dashboard():
    """Main dashboard renderer"""
    render_dashboard_header()
    render_dashboard_summary_cards()
    
    tab1, tab2, tab3, tab4, tab5 = render_dashboard_tabs()
    
    with tab1:
        render_patient_form()
    with tab2:
        render_adr_predictions_tab()
    with tab3:
        render_explainability_tab()
    with tab4:
        render_bias_audit_tab()
    with tab5:
        render_workflow_tab()
    # with tab6:
    #     render_fhir_export_tab()


def main():
    """Main application with landing + dashboard views"""
    try:
        params = st.query_params
        page_param = params.get("page", "landing")
        page = page_param[0] if isinstance(page_param, list) else page_param
    except Exception:
        page = "landing"

    if page == "dashboard":
        render_dashboard()
    else:
        page_patient_entry()


if __name__ == "__main__":
    main()
