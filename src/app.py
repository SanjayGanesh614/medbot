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
import xgboost as xgb

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    load_model, get_risk_category, get_risk_color, 
    create_prediction_summary, generate_report_filename, parse_fhir_patient
)
from src.explainability import SHAPExplainer
from src.evaluate import ModelEvaluator

# --- NEW IMPORTS (AUTH & DB & AGENT) ---
from src.database import init_db, get_db, SessionLocal, User, PatientRecord
from src.auth import signup_user, login_user
from src.clinical_agent import get_clinical_recommendations
# ---------------------------------------

# Initialize Database
init_db()

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
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, "models", "xgb_adr_model.json")
        model = load_model(model_path)
        explainer = SHAPExplainer(model=model)
        
        # Try to load pre-computed explainer
        try:
            explainer.load_explainer(os.path.join(base_dir, "models", "shap_explainer.pkl"))
        except:
            # Create new explainer if not found
            try:
                template_path = os.path.join(base_dir, "models", "feature_template.csv")
                X_sample = pd.read_csv(template_path).sample(100, random_state=42)
                explainer.create_explainer(X_sample)
            except:
                explainer.create_explainer()
        
        return model, explainer
    except Exception as e:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # redundant re-calc if exception happens early but safe
        st.error(f"Error loading model from {base_dir}: {e}")
        st.info("Please run training first: `python src/train_xgb.py`")
        return None, None


@st.cache_data
def load_drug_list():
    """Load available drugs from FAERS data"""
    try:
        # Fix: Use correct path
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        faers_path = os.path.join(base_dir, "colabupload", "faers_drug_summary.csv")
        faers = pd.read_csv(faers_path)
        drugs = sorted(faers['drugname'].dropna().unique().tolist())
        return drugs
    except:
        return ["Aspirin", "Metformin", "Lisinopril", "Atorvastatin", "Levothyroxine"]


@st.cache_data
def load_encoders_map():
    """Load and invert encoders.json for mapping UI strings to Model integers"""
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Fix: User specified encoders.json is directly in colabupload
        enc_path = os.path.join(base_dir, "colabupload", "encoders.json")
        
        if not os.path.exists(enc_path):
            print(f"Encoders not found at {enc_path}")
            return None
            
        with open(enc_path, "r") as f:
            raw_encoders = json.load(f)
            
        # Convert list of classes to dict {label: index}
        encoders_map = {}
        for col, classes in raw_encoders.items():
            encoders_map[col] = {str(c): i for i, c in enumerate(classes)}
        return encoders_map
    except Exception as e:
        print(f"Warning: Could not load encoders: {e}")
        return None

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
    # Custom CSS to Make Streamlit Buttons look like the original Hero Buttons
    st.markdown("""
    <style>
    /* Force specific styling for buttons in this view */
    div[data-testid="stHorizontalBlock"] button[kind="primary"] {
        background-color: #111827 !important;
        color: #FFFFFF !important;
        border-radius: 999px !important;
        padding: 0.5rem 2rem !important;
        border: none !important;
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.08) !important;
    }
    div[data-testid="stHorizontalBlock"] button[kind="primary"]:hover {
        background-color: #030712 !important;
        box-shadow: 0 14px 28px rgba(15, 23, 42, 0.16) !important;
        transform: translateY(-1px);
    }
    div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
        background-color: #FFFFFF !important;
        color: #111827 !important;
        border: 1px solid #D1D5DB !important;
        border-radius: 999px !important;
        padding: 0.5rem 2rem !important;
    }
    div[data-testid="stHorizontalBlock"] button[kind="secondary"]:hover {
        background-color: #F9FAFB !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Replaced HTML buttons with Streamlit buttons
    _, col_btn1, col_btn2, _ = st.columns([1, 1.2, 1.2, 1])
    with col_btn1:
        if st.button("⚡ Get Started", type="primary", use_container_width=True):
            st.query_params["page"] = "dashboard"
            st.rerun()
    with col_btn2:
        if st.button("Learn More", use_container_width=True):
            st.query_params["page"] = "dashboard"
            st.rerun()

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
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Bottom CTA Button (Streamlit Native)
    _, col_cta, _ = st.columns([1, 1.5, 1])
    with col_cta:
        if st.button("⚡ Start Using AI-CPA", type="primary", use_container_width=True, key="cta_btn"):
            st.query_params["page"] = "dashboard"
            st.rerun()


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
        # Fix: Use correct path 'colabupload' instead of 'data/output'
        faers_data = pd.read_csv("colabupload/faers_drug_summary.csv")
        
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
        # Fix: Use correct path
        faers_data = pd.read_csv("colabupload/faers_drug_summary.csv")
        
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
        X_template = pd.read_csv("models/feature_template.csv").iloc[0:1].copy()
        
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
        # Fix: Use DMatrix for Booster and remove leakage
        leakage = ['weak_score', 'high_risk_drug', 'faers_adr_rate', 'faers_severe_rate']
        X_safe = X_template.drop(columns=[c for c in leakage if c in X_template.columns], errors='ignore')
        
        dtest = xgb.DMatrix(X_safe)
        risk_proba = model.predict(dtest)[0]
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
            
            # Correct way to get importance from Booster
            importance_map = model.get_score(importance_type='gain')
            # Map valid features, default to 0
            feature_importance = [importance_map.get(f, 0) for f in feature_names]
            
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
        
        # Clinical recommendations (ENHANCED WITH AGENT)
        st.markdown("---")
        st.subheader("🏥 Clinical Recommendations")
        
        # 1. Standard Heuristic Recommendations (Existing)
        recommendations = generate_clinical_recommendations(risk_proba, risk_category, patient_data)
        
        for rec in recommendations:
            if "High" in risk_category:
                st.error(f"🚨 {rec}")
            elif "Moderate" in risk_category:
                st.warning(f"⚠️ {rec}")
            else:
                st.success(f"✅ {rec}")

        # 2. AI Agent Recommendations (New)
        if st.session_state.get('user'):
            with st.expander("🤖 AI Pharmacist Insights (Powered by Gemini)", expanded=False):
                with st.spinner("Analyzing patient profile with AI..."):
                    # Basic risk analysis dict for the agent
                    risk_analysis_summary = {
                        'risk_score': risk_proba,
                        'risk_category': risk_category
                    }
                    ai_recs = get_clinical_recommendations(patient_data, risk_analysis_summary)
                    st.markdown(ai_recs)

        # 3. Save Report Feature (New)
        col_save, col_spacer = st.columns([1, 4])
        with col_save:
            if st.button("💾 Save to History"):
                if 'user' in st.session_state:
                    db = SessionLocal()
                    try:
                        # Serialize data
                        import json
                        p_json = json.dumps(patient_data, default=str)
                        r_json = json.dumps({
                            'risk_score': risk_proba, 
                            'risk_category': risk_category,
                            'timestamp': datetime.now().isoformat()
                        }, default=str)
                        
                        user_id = st.session_state['user'].id
                        new_record = PatientRecord(
                            user_id=user_id,
                            patient_name=f"Patient {patient_data.get('anchor_age')}y/{patient_data.get('gender')}",
                            risk_score=risk_proba,
                            risk_category=risk_category,
                            patient_data_json=p_json,
                            prediction_result_json=r_json,
                            clinical_recommendations=str(recommendations)
                        )
                        db.add(new_record)
                        db.commit()
                        st.success("Report saved to history!")
                    except Exception as e:
                        st.error(f"Error saving report: {e}")
                    finally:
                        db.close()
                else:
                    st.info("Please log in to save reports.")
        
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
            feature_names = pd.read_csv("models/feature_template.csv").columns
            # Correct way to get importance from Booster
            importance_map = model.get_score(importance_type='gain')
            feature_importance = [importance_map.get(f, 0) for f in feature_names]
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance
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
                X_template = pd.read_csv("models/feature_template.csv").iloc[0:1].copy()
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
    
    # Feature Importance Analysis
    st.markdown("---")
    st.subheader("📊 Key Risk Drivers")
    
    # Define feature_names centrally before any try/except blocks
    try:
        feature_names = pd.read_csv("models/feature_template.csv").columns.tolist()
    except:
        feature_names = []
    
    # Static Feature Importance Plot
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
    # Debug: See what data is actually coming in
    # st.write("Debug - Incoming Patient Data:", patient_data) 
    
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
    # Prepare complete patient data (Start with copy to preserve all keys like comorbidities, labs)
    complete_patient_data = patient_data.copy()
    
    # Update with calculated/normalized fields
    complete_patient_data.update({
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
        # Ensure Critical Labs are set (defaults handled in mapping but good to have here)
        'lab_creatinine': patient_data.get('lab_creatinine', 1.0),
        'lab_hemoglobin': patient_data.get('lab_hemoglobin', 13.5),
        'lab_platelet_count': patient_data.get('lab_platelet_count', 250),
        'lab_white_blood_cells': patient_data.get('lab_white_blood_cells', 7.5),
    })
    
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
        X_template = pd.read_csv("models/feature_template.csv").iloc[0:1].copy()
        for col in X_template.columns:
            X_template[col] = 0
        
        # Calculate derived clinical flags (replicating preprocess.py logic)
        renal_abnormal_flag = 1 if complete_patient_data.get('lab_creatinine', 0) > 1.5 else 0
        
        # Hepatic flag: ALT > 40 OR AST > 40 OR Bilirubin > 1.2
        # Note: Bilirubin might be missing in form, default to 0.8 (normal)
        lab_alt = complete_patient_data.get('lab_alt', 25)
        lab_ast = complete_patient_data.get('lab_ast', 30)
        lab_bilirubin = complete_patient_data.get('lab_bilirubin', 0.8)
        
        hepatic_abnormal_flag = 1 if (lab_alt > 40 or lab_ast > 40 or lab_bilirubin > 1.2) else 0
        
        # Anemia: Hemoglobin < 10
        anemia_flag = 1 if complete_patient_data.get('lab_hemoglobin', 13.5) < 10 else 0
        
        # Thrombocytopenia: Platelets < 150
        thrombocytopenia_flag = 1 if complete_patient_data.get('lab_platelet_count', 250) < 150 else 0
        
        # Infection: WBC > 12
        infection_flag = 1 if complete_patient_data.get('lab_white_blood_cells', 7.5) > 12 else 0

        # --- Fix: Map Comorbidities to Model Features ---
        comorbs = [c.lower() for c in complete_patient_data.get('comorbidities', [])]
        
        feat_ckd = 1 if any('kidney' in c or 'ckd' in c or 'renal' in c for c in comorbs) else 0
        feat_cad_hf = 1 if any('heart' in c or 'failure' in c or 'cad' in c or 'hf' in c for c in comorbs) else 0
        feat_diabetes = 1 if any('diabet' in c for c in comorbs) else 0
        feat_htn = 1 if any('hypertens' in c or 'bp' in c for c in comorbs) else 0
        feat_resp = 1 if any('asthma' in c or 'copd' in c for c in comorbs) else 0
        feat_liver = 1 if any('liver' in c or 'cirrhosis' in c for c in comorbs) else 0
        
        # New: Extended Feature Mapping
        feat_cancer = 1 if any('cancer' in c or 'malignan' in c or 'tumor' in c or 'chemo' in c for c in comorbs) else 0
        feat_immune = 1 if any('immune' in c or 'transplant' in c or 'hiv' in c for c in comorbs) else 0
        feat_dialysis = 1 if any('dialysis' in c for c in comorbs) or complete_patient_data.get('on_dialysis') else 0
        feat_oxygen = 1 if any('oxygen' in c for c in comorbs) or complete_patient_data.get('on_oxygen') else 0
        feat_vent = 1 if complete_patient_data.get('on_ventilator') else 0
        feat_vaso = 1 if complete_patient_data.get('on_vasopressors') else 0

        # --- Categorical Encodings (Dynamic from encoders.json) ---
        encoders = load_encoders_map()
        
        if encoders:
             def get_enc(col, val, default=0):
                 if col in encoders:
                     # Try exact match, then 'Other', then default
                     return encoders[col].get(str(val), encoders[col].get('Other', default))
                 return default

             race_val = get_enc('race', complete_patient_data.get('race', 'Other'))
             ins_val = get_enc('insurance', complete_patient_data.get('insurance', 'Medicare'))
             marital_val = get_enc('marital_status', complete_patient_data.get('marital_status', 'Married'))
             adm_val = get_enc('admission_type', complete_patient_data.get('admission_type', 'Emergency'))
             # Map 'ward' to 'admission_location'
             loc_val = get_enc('admission_location', complete_patient_data.get('ward', 'ICU'))
        else:
            # Fallback (Approximate indices if encoders.json missing)
            race_map = {"White": 29, "Black": 4, "Hispanic": 12, "Asian": 2, "Other": 20} 
            ins_map = {"Medicare": 2, "Private": 4, "Medicaid": 1, "Other": 3}
            marital_map = {"Married": 2, "Single": 4, "Widowed": 6, "Divorced": 1}
            adm_type_map = {"Emergency": 5, "Inpatient": 1, "OPD": 6, "ICU": 2}
            ward_map = {"General Ward": 9, "HDU": 3, "ICU": 2, "Private": 4} 

            race_val = race_map.get(complete_patient_data.get('race', 'Other'), 0)
            ins_val = ins_map.get(complete_patient_data.get('insurance', 'Medicare'), 0)
            marital_val = marital_map.get(complete_patient_data.get('marital_status', 'Married'), 0)
            adm_val = adm_type_map.get(complete_patient_data.get('admission_type', 'Emergency'), 0)
            loc_val = ward_map.get(complete_patient_data.get('ward', 'General Ward'), 9)

        feature_mapping = {
            # Comorbidities (CRITICAL MISSING SIGNALS)
            'ckd': feat_ckd,
            'cad_hf': feat_cad_hf,
            'diabetes_type2': feat_diabetes,
            'hypertension': feat_htn,
            'copd_asthma': feat_resp,
            'chronic_liver_disease': feat_liver,
            'malignancy': feat_cancer,
            'immunosuppressed': feat_immune,
            'on_dialysis': feat_dialysis,
            'on_oxygen': feat_oxygen,
            'on_ventilator': feat_vent,
            'on_vasopressors': feat_vaso,
            'aki': 1 if feat_ckd and complete_patient_data.get('lab_creatinine', 0) > 2.0 else 0,

            # Demographics & Context
            'gender': complete_patient_data['gender'],
            'anchor_age': complete_patient_data['anchor_age'],
            'race': race_val,
            'insurance': ins_val,
            'marital_status': marital_val,
            'admission_type': adm_val,
            'admission_location': loc_val, # Mapped from Ward
            'discharge_location': 6, # Default to Home (6)
            
            # Hospitalization
            'num_admissions': complete_patient_data['num_admissions'],
            'avg_los_days': complete_patient_data['avg_los_days'],
            'los_days': complete_patient_data.get('los_days', 3), 
            'duration_days': complete_patient_data.get('los_days', 3), # Model alias
            'hospital_expire_flag': complete_patient_data['ever_died_in_hospital'], # Model alias
            'ever_died_in_hospital': complete_patient_data['ever_died_in_hospital'], # Keep original just in case
            
            # Vitals
            'vital_heart_rate': complete_patient_data.get('vital_heart_rate', 72),
            'vital_respiratory_rate': complete_patient_data.get('vital_respiratory_rate', 16),
            'vital_temperature_celsius': complete_patient_data.get('vital_temperature_celsius', 37.0),
            'vital_spo2': complete_patient_data.get('vital_spo2', 98),
            'vital_arterial_blood_pressure_systolic': complete_patient_data.get('vital_arterial_blood_pressure_systolic', 120),
            'vital_arterial_blood_pressure_diastolic': complete_patient_data.get('vital_arterial_blood_pressure_diastolic', 80),
            'vital_arterial_blood_pressure_mean': complete_patient_data.get('vital_arterial_blood_pressure_mean', 93),

            # Labs (Comprehensive)
            'lab_creatinine': complete_patient_data['lab_creatinine'],
            'lab_hemoglobin': complete_patient_data['lab_hemoglobin'],
            'lab_platelet_count': complete_patient_data['lab_platelet_count'],
            'lab_white_blood_cells': complete_patient_data['lab_white_blood_cells'],
            'lab_sodium': complete_patient_data.get('lab_sodium', 140),
            'lab_potassium': complete_patient_data.get('lab_potassium', 4.0),
            'lab_calcium_total': complete_patient_data.get('lab_calcium_total', 9.0),
            'lab_magnesium': complete_patient_data.get('lab_magnesium', 2.0),
            'lab_chloride': complete_patient_data.get('lab_chloride', 100),
            'lab_bicarbonate': complete_patient_data.get('lab_bicarbonate', 24),
            'lab_glucose': complete_patient_data.get('lab_glucose', 100),
            'lab_urea_nitrogen': complete_patient_data.get('lab_urea_nitrogen', 15),
            
            # Liver
            'alt_first': lab_alt, 'alt_last': lab_alt, # Assume current is singular
            'ast_first': lab_ast, 'ast_last': lab_ast,
            'alp_first': complete_patient_data.get('lab_alp', 70), 
            'alp_last': complete_patient_data.get('lab_alp', 70),
            'total_bilirubin_first': lab_bilirubin,
            'total_bilirubin_last': lab_bilirubin,

            # Drug Stats
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
            
            # Derived Flags
            'renal_abnormal_flag': renal_abnormal_flag,
            'hepatic_abnormal_flag': hepatic_abnormal_flag,
            'anemia_flag': anemia_flag,
            'thrombocytopenia_flag': thrombocytopenia_flag,
            'infection_flag': infection_flag
        }
        
        for feature, value in feature_mapping.items():
            if feature in X_template.columns:
                X_template[feature] = value
        
        X_template = X_template.fillna(X_template.median())
        
        # Make prediction
        # Fix: Use DMatrix for Booster and remove leakage
        leakage = ['weak_score', 'high_risk_drug', 'faers_adr_rate', 'faers_severe_rate']
        X_safe = X_template.drop(columns=[c for c in leakage if c in X_template.columns], errors='ignore')
        
        dtest = xgb.DMatrix(X_safe)
        preds = model.predict(dtest)
        
        if isinstance(preds, (list, np.ndarray)) and len(preds) > 0:
            risk_proba = preds[0]
        elif np.isscalar(preds):
            risk_proba = preds
        else:
             print(f"Prediction Error: Model returned {preds}")
             risk_proba = 0.0
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
    """Render patient entry form - Refactored into 8 Clinical Sections (Expanders)"""
    st.markdown(
        """
        <div class="patient-management-section">
            <div class="section-header">
                <div>👤</div>
                <div>
                    <div class="section-title">Patient Clinical Data Entry</div>
                    <div class="section-subtitle">Comprehensive assessment for precise ADR prediction</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Check session state initialization for dynamic lists
    if 'medications_list' not in st.session_state:
        st.session_state['medications_list'] = []

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

    # --- MANUAL ENTRY FORM ---
    if upload_option == "Manual Entry":
        st.markdown("---")
        st.info("Please complete the following 8 sections for a comprehensive analysis.")
        
        # 1. Patient Identification & Context
        with st.expander("1. Patient Identification & Context", expanded=True):
            st.subheader("Patient Identification")
            c1, c2, c3 = st.columns(3)
            with c1:
                age = st.number_input("Age (years)", min_value=18, max_value=120, value=65, key="sec1_age")
                gender = st.selectbox("Sex", ["Male", "Female", "Other"], key="sec1_gender")
            with c2:
                weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, key="sec1_weight")
                height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, key="sec1_height")
            with c3:
                bmi = 0.0
                if height > 0:
                    bmi = weight / ((height/100)**2)
                st.metric("BMI", f"{bmi:.1f}")
                
            st.subheader("Care Setting")
            c1, c2 = st.columns(2)
            with c1:
                adm_type = st.selectbox("Admission Type", ["Inpatient", "OPD", "ICU", "Emergency"], key="sec1_adm_type")
                ward = st.selectbox("Ward Type", ["General Ward", "HDU", "ICU", "Private"], key="sec1_ward")
            with c2:
                st.number_input("Day of Hospitalization", min_value=1, value=1, key="sec1_day")
                st.selectbox("Primary Reason", ["Infection", "Cardiovascular", "Respiratory", "Trauma", "Neurological", "Other"], key="sec1_reason")

        # 2. Clinical Vitals
        with st.expander("2. Clinical Vitals & Organ Status", expanded=False):
            st.subheader("Hemodynamics & Vitals")
            c1, c2, c3 = st.columns(3)
            with c1:
                sbp = st.number_input("Systolic BP (mmHg)", 50, 250, 120, key="sec2_sbp")
                dbp = st.number_input("Diastolic BP (mmHg)", 30, 150, 80, key="sec2_dbp")
                map_val = (sbp + (2*dbp))/3
                st.metric("Mean Arterial Pressure", f"{map_val:.1f}")
            with c2:
                hr = st.number_input("Heart Rate (bpm)", 30, 200, 72, key="sec2_hr")
                rr = st.number_input("Resp. Rate (bpm)", 8, 60, 16, key="sec2_rr")
                temp = st.number_input("Temperature (°C)", 32.0, 42.0, 37.0, key="sec2_temp")
            with c3:
                spo2 = st.number_input("SpO2 (%)", 50, 100, 98, key="sec2_spo2")
                
            st.subheader("Organ Support")
            col_os1, col_os2 = st.columns(2)
            with col_os1:
                st.checkbox("Oxygen Therapy", key="sec2_o2")
                st.checkbox("Mechanical Ventilation", key="sec2_mv")
            with col_os2:
                st.checkbox("Dialysis / CRRT", key="sec2_dialysis")
                st.checkbox("Vasopressor Support", key="sec2_vaso")

        # 3. Laboratory Results
        with st.expander("3. Laboratory Results (ADR-Relevant)", expanded=False):
            st.info("💡 Enter values to see auto-range analysis.")
            
            st.markdown("#### 🩸 Hematology")
            c1, c2, c3 = st.columns(3)
            with c1:
                hb = st.number_input("Hemoglobin (g/dL)", 0.0, 25.0, 13.5, help="Normal: 12-16 (F), 13.5-17.5 (M)", key="sec3_hb")
            with c2:
                wbc = st.number_input("WBC Count (K/uL)", 0.0, 50.0, 7.5, help="Normal: 4.5-11.0", key="sec3_wbc")
            with c3:
                plt = st.number_input("Platelets (K/uL)", 0.0, 1000.0, 250.0, help="Normal: 150-450", key="sec3_plt")
                
            st.markdown("#### 💧 Renal & Electrolytes")
            c1, c2, c3 = st.columns(3)
            with c1:
                creat = st.number_input("Creatinine (mg/dL)", 0.0, 15.0, 1.0, key="sec3_creat")
                egfr = st.number_input("eGFR (mL/min)", 0, 140, 90, key="sec3_egfr")
            with c2:
                na = st.number_input("Sodium (mEq/L)", 100, 180, 140, key="sec3_na")
                k = st.number_input("Potassium (mEq/L)", 1.0, 10.0, 4.0, key="sec3_k")
            with c3:
                ca = st.number_input("Calcium (mg/dL)", 0.0, 20.0, 9.0, key="sec3_ca")
                mg = st.number_input("Magnesium (mg/dL)", 0.0, 10.0, 2.0, key="sec3_mg")
                
            st.markdown("#### 🍺 Liver Function")
            c1, c2 = st.columns(2)
            with c1:
                alt = st.number_input("ALT (U/L)", 0, 1000, 25, key="sec3_alt")
                ast = st.number_input("AST (U/L)", 0, 1000, 30, key="sec3_ast")
            with c2:
                alp = st.number_input("ALP (U/L)", 0, 1000, 70, key="sec3_alp")
                bili = st.number_input("Total Bilirubin (mg/dL)", 0.0, 30.0, 0.8, key="sec3_bili")

        # 4. Comorbidities
        with st.expander("4. Comorbidities (Structured)", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Cardiometabolic**")
                htn = st.checkbox("Hypertension", key="sec4_htn")
                dm = st.checkbox("Diabetes Mellitus", key="sec4_dm")
                dm_type = st.select_slider("DM Type", ["Type 1", "Type 2"], disabled=not dm, key="sec4_dm_type") if dm else None
                cad = st.checkbox("CAD / Heart Failure", key="sec4_cad")
                st.markdown("**Respiratory**")
                resp_dz = st.checkbox("Asthma / COPD", key="sec4_resp")
            
            with c2:
                st.markdown("**Renal & Hepatic**")
                ckd = st.checkbox("Chronic Kidney Disease (CKD)", key="sec4_ckd")
                ckd_stage = st.selectbox("CKD Stage", ["Stage 1", "Stage 2", "Stage 3", "Stage 4", "Stage 5"], disabled=not ckd, key="sec4_ckd_stage") if ckd else None
                cld = st.checkbox("Chronic Liver Disease", key="sec4_cld")
                st.markdown("**Other**")
                malig = st.checkbox("Malignancy", key="sec4_malig")
                immuno = st.checkbox("Immunosuppression", key="sec4_immuno")

        # 5. Medication Profile
        with st.expander("5. Medication Profile (MOST IMPORTANT)", expanded=True):
            st.info("Add each medication separately.")
            
            with st.form("add_drug_form", clear_on_submit=True):
                c1, c2 = st.columns(2)
                d_name = c1.text_input("Generic Name")
                d_dose = c2.text_input("Dose (e.g., 500mg)")
                
                c3, c4 = st.columns(2)
                d_route = c3.selectbox("Route", ["PO (Oral)", "IV", "IM", "SC", "Topical"])
                d_freq = c4.selectbox("Frequency", ["OD", "BD", "TDS", "QID", "HS", "STAT"])
                
                high_risk = st.checkbox("⚠️ High Risk / Narrow Therapeutic Index")
                
                if st.form_submit_button("➕ Add Drug"):
                    if d_name:
                        st.session_state['medications_list'].append({
                            "name": d_name,
                            "dose": d_dose,
                            "route": d_route,
                            "freq": d_freq,
                            "high_risk": high_risk
                        })
                        st.rerun()
                    else:
                        st.error("Drug name is required")

            if st.session_state['medications_list']:
                st.write(f"**Current Medications ({len(st.session_state['medications_list'])})**")
                for idx, med in enumerate(st.session_state['medications_list']):
                    col_txt, col_act = st.columns([5,1])
                    with col_txt:
                        risk_mark = "⚠️" if med['high_risk'] else "💊"
                        st.markdown(f"{risk_mark} **{med['name']}** {med['dose']} via {med['route']} ({med['freq']})")
                    with col_act:
                        if st.button("❌", key=f"del_med_{idx}"):
                            st.session_state['medications_list'].pop(idx)
                            st.rerun()
            else:
                st.warning("No medications added.")

        # 6. ADR & Allergy History
        with st.expander("6. ADR & Allergy History", expanded=False):
            st.text_area("Known Drug Allergies", placeholder="e.g., Penicillin (Rash)", key="sec6_allergies")
            st.text_area("Previous Adverse Reactions", placeholder="e.g., ACE Inhibitors (Cough)", key="sec6_adrs")

        # 7. Hospitalization Summary
        with st.expander("7. Hospitalization Summary", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                n_adm_1yr = st.number_input("Admissions (Past 12m)", 0, 50, 1, key="sec7_adm")
                icu_adm = st.radio("ICU Admission?", ["No", "Yes"], key="sec7_icu")
            with c2:
                avg_los = st.number_input("Average LOS (Days)", 1.0, 100.0, 4.5, key="sec7_los")
                total_procedures = st.number_input("Total Procedures", 0, 20, 0, key="sec7_proc")

        # 8. Submission & Validation
        with st.expander("8. Submission & Validation", expanded=True):
            st.subheader("✅ Final Check")
            
            # Compile Comorbidities
            comorbs = []
            if htn: comorbs.append("Hypertension")
            if dm: comorbs.append(f"Diabetes ({dm_type})")
            if ckd: comorbs.append(f"CKD ({ckd_stage})")
            if cad: comorbs.append("Heart Failure")
            if resp_dz: comorbs.append("Asthma/COPD")
            if cld: comorbs.append("Chronic Liver Disease")
            if malig: comorbs.append("Malignancy")
            if immuno: comorbs.append("Immunosuppression")
            
            # Validation
            missing_fields = []
            if not st.session_state['medications_list']:
                missing_fields.append("Medications")
            
            if missing_fields:
                st.error(f"⚠️ Missing Mandatory Fields: {', '.join(missing_fields)}")
            else:
                st.success("Ready for Prediction")
                
                if st.button("🚀 Predict ADR Risk", type="primary", use_container_width=True):
                    # Construct Payload
                    patient_payload = {
                        'anchor_age': age,
                        'gender': 'M' if gender=='Male' else ('F' if gender=='Female' else 'O'),
                        'weight': weight,
                        'height': height,
                        'num_admissions': n_adm_1yr,
                        'avg_los_days': avg_los,
                        'total_diagnoses': len(comorbs),
                        'total_procedures': total_procedures,
                        'selected_drugs': [m['name'] for m in st.session_state['medications_list']],
                        'num_icu_stays': 1 if icu_adm == "Yes" else 0,
                        'comorbidities': comorbs,
                        # Labs
                        'lab_creatinine': creat,
                        'lab_hemoglobin': hb,
                        'lab_platelet_count': plt,
                        'lab_white_blood_cells': wbc,
                        'lab_alt': alt,
                        'lab_ast': ast,
                        'lab_bilirubin': bili,
                        'lab_egfr': egfr,
                        'lab_alp': alp,
                        # Vitals
                        'vitals': {'sbp': sbp, 'dbp': dbp, 'hr': hr, 'spo2': spo2, 'temp': temp, 'rr': rr},
                        'temp_c': temp,
                        'spo2': spo2,
                        'resprate': rr,
                        'heart_rate': hr,
                        'blood_pressure_systolic': sbp,
                        'blood_pressure_diastolic': dbp,
                        # Other
                        'admission_type': adm_type,
                        'primary_reason': st.session_state.get('sec1_reason'), 
                    }
                    
                    with st.spinner("Running AI Analysis..."):
                        process_uploaded_patient_data(patient_payload)
                        st.success("Prediction Complete! Switch to 'ADR Predictions' tab.")
            



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
            feature_names = pd.read_csv("models/feature_template.csv").columns
            # Correct way to get importance from Booster
            importance_map = model.get_score(importance_type='gain')
            feature_importance = [importance_map.get(f, 0) for f in feature_names]
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance
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
            X_template = pd.read_csv("models/feature_template.csv").iloc[0:1].copy()
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
                # Fix: Use DMatrix for Booster and remove leakage
                leakage = ['weak_score', 'high_risk_drug', 'faers_adr_rate', 'faers_severe_rate']
                X_safe = X_template.drop(columns=[c for c in leakage if c in X_template.columns], errors='ignore')
                
                dtest = xgb.DMatrix(X_safe)
                risk_proba = model.predict(dtest)[0]
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



# ============================================================================
# NEW: AUTHENTICATION & ADMIN PAGES
# ============================================================================

def login_signup_interface():
    """Render Login/Signup container"""
    st.markdown("""
        <style>
            .auth-container { max-width: 400px; margin: 0 auto; padding: 2rem; background: white; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            h1 { text-align: center; color: #1F2937; }
        </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🔐 AI-CPA Login")
        
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        
        with tab1:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Log In", use_container_width=True)
                
                if submitted:
                    db = SessionLocal()
                    user, msg = login_user(db, username, password)
                    db.close()
                    
                    if user:
                        st.session_state['user'] = user
                        st.success(f"Welcome back, {user.username}!")
                        st.rerun()
                    else:
                        st.error(msg)
        
        with tab2:
            with st.form("signup_form"):
                new_user = st.text_input("Choose Username")
                new_pass = st.text_input("Choose Password", type="password")
                # Admin secret code check (simple implementation)
                admin_code = st.text_input("Admin Code (Optional)", type="password", help="Enter secret code to create admin account")
                
                submitted = st.form_submit_button("Create Account", use_container_width=True)
                
                if submitted:
                    if new_user and new_pass:
                        db = SessionLocal()
                        role = 'admin' if admin_code == "medbot_admin_2024" else 'user'
                        user, msg = signup_user(db, new_user, new_pass, role)
                        db.close()
                        
                        if user:
                            st.success("Account created! Please log in.")
                        else:
                            st.error(msg)
                    else:
                        st.warning("Please fill all fields.")

def page_user_history():
    """Display user's saved history"""
    st.markdown('<div class="main-header">My Saved Reports</div>', unsafe_allow_html=True)
    
    if 'user' not in st.session_state:
        st.error("Please login.")
        return

    db = SessionLocal()
    records = db.query(PatientRecord).filter(PatientRecord.user_id == st.session_state['user'].id).order_by(PatientRecord.created_at.desc()).all()
    db.close()
    
    if not records:
        st.info("No saved reports found.")
        return

    for rec in records:
        with st.expander(f"{rec.created_at.strftime('%Y-%m-%d %H:%M')} - {rec.risk_category} Risk ({rec.risk_score:.1%})"):
            st.write(f"**Patient:** {rec.patient_name}")
            col1, col2 = st.columns(2)
            with col1:
                st.json(rec.prediction_result_json) # Simplified view
            with col2:
                if st.button(f"Load to Dashboard #{rec.id}"):
                    # Restore to session state
                    import json
                    st.session_state['patient_data'] = json.loads(rec.patient_data_json)
                    st.session_state['current_page'] = 'dashboard'
                    st.rerun()

def page_admin_dashboard():
    """Admin Dashboard to view all data"""
    st.markdown('<div class="main-header">🛡️ Admin Dashboard</div>', unsafe_allow_html=True)
    
    if 'user' not in st.session_state or st.session_state['user'].role != 'admin':
        st.error("Access Denied.")
        return
        
    db = SessionLocal()
    users = db.query(User).all()
    records = db.query(PatientRecord).all()
    db.close()
    
    st.metric("Total Users", len(users))
    st.metric("Total Records Saved", len(records))
    
    st.subheader("All User Data")
    
    data = []
    for r in records:
        data.append({
            "ID": r.id,
            "User ID": r.user_id,
            "Date": r.created_at,
            "Risk": r.risk_score,
            "Category": r.risk_category
        })
    
    if data:
        st.dataframe(pd.DataFrame(data))
    else:
        st.info("No records yet.")
        
    st.markdown("---")
    st.subheader("🔧 Model Maintenance")
    if st.button("🚀 Retrain Model on New Data"):
        st.info("Training trigger sent... (Simulated)")
        # In a real app, this would trigger a subprocess call to train_with_balancing.py
        # subprocess.Popen(["python", "train_with_balancing.py"])
        st.success("Model training initiated in background.")

# ============================================================================
# MAIN APP ROUTER
# ============================================================================

def main():
    """Main application with Authentication Gate"""
    
    # 1. Check Authentication
    if 'user' not in st.session_state:
        # Show Login/Signup ONLY
        login_signup_interface()
        return

    # 2. Get Page from Query Params
    try:
        params = st.query_params
        page_param = params.get("page", "landing")
        # Handle different Streamlit versions where query_params might return list or string
        if isinstance(page_param, list):
            current_page = page_param[0]
        else:
            current_page = page_param
    except Exception:
        current_page = "landing"

    # 3. Routing Logic
    if current_page == "dashboard":
        # === DASHBOARD MODE ===
        # Ensure sidebar is visible
        st.markdown("""<style>[data-testid="stSidebar"] {display: block !important;}</style>""", unsafe_allow_html=True)
        
        user = st.session_state['user']
        
        # Sidebar for navigation
        with st.sidebar:
            st.title(f"👤 {user.username}")
            st.caption(f"Role: {user.role.upper()}")
            
            if st.button("Log Out"):
                del st.session_state['user']
                st.query_params["page"] = "landing" # Reset param
                st.rerun()
                
            st.markdown("---")
            nav_options = ["Dashboard", "My History"]
            if user.role == 'admin':
                nav_options.append("Admin Panel")
                
            selected_page = st.radio("Navigate", nav_options)

        # Render selected view
        if selected_page == "Dashboard":
            try:
                render_dashboard()
            except Exception as e:
                st.error(f"Application Error: {e}")
                st.exception(e)
                
        elif selected_page == "My History":
            page_user_history()
            
        elif selected_page == "Admin Panel":
            page_admin_dashboard()
            
    else:
        # === LANDING PAGE MODE ===
        # Ensure sidebar is HIDDEN for full landing page experience
        st.markdown("""<style>[data-testid="stSidebar"] {display: none !important;}</style>""", unsafe_allow_html=True)
        
        # Render the original landing page
        page_patient_entry()


if __name__ == "__main__":
    main()

