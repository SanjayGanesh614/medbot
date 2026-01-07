import os
import google.generativeai as genai
import streamlit as st
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure the API key
API_KEY = os.environ.get("GOOGLE_API_KEY") 

# Fallback to secrets if not in env, but wrap in try-except to avoid crashing if empty
if not API_KEY:
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            API_KEY = st.secrets["GOOGLE_API_KEY"]
    except FileNotFoundError:
        pass # No secrets file found, ignore

if API_KEY:
    genai.configure(api_key=API_KEY)

def get_clinical_recommendations(patient_data, risk_analysis):
    """
    Generate clinical recommendations using Google GenAI based on patient data and risk analysis.
    
    Args:
        patient_data (dict): Dictionary containing patient demographics, vitals, drugs, etc.
        risk_analysis (dict): Dictionary containing risk score, category, and explanation.
        
    Returns:
        str: Markdown formatted clinical recommendations.
    """
    if not API_KEY:
        return "⚠️ Google API Key not found. Clinical recommendations are unavailable."

    try:
        model = genai.GenerativeModel('gemini-pro')
        
        # Construct a comprehensive prompt
        # Masking PII is important, though this is a demo/local app.
        
        medications = ", ".join(patient_data.get('selected_drugs', []))
        conditions = ", ".join(patient_data.get('comorbidities', []))
        if not conditions:
            conditions = "None documented"
        
        prompt = f"""
        You are an expert Clinical Pharmacist specialized in Adverse Drug Reaction (ADR) prevention.
        
        **Patient Profile:**
        - Age: {patient_data.get('anchor_age')}
        - Sex: {'Male' if patient_data.get('gender') == 1 else 'Female'}
        - Weight: {patient_data.get('weight')} kg
        - Conditions: {conditions}
        - Current Medications: {medications}
        - Creatinine: {patient_data.get('lab_creatinine')} mg/dL
        - AST/ALT: {patient_data.get('lab_ast')}/{patient_data.get('lab_alt')} U/L
        
        **Model Analysis:**
        - ADR High Risk Score: {risk_analysis.get('risk_score', 0):.2%}
        - Risk Category: {risk_analysis.get('risk_category')}
        
        **Task:**
        Provide 3-5 specific, actionable, and realistic clinical recommendations to manage or mitigate the ADR risk for this patient.
        Focus on:
        1. Drug monitoring (specific labs to check).
        2. Potential drug-drug interactions to review.
        3. Dosage adjustments if applicable (based on renal/hepatic function).
        4. Warning signs to educate the patient about.
        
        **Format:**
        Return the response in clear, bulleted Markdown. Do not include a preamble or generic disclaimer (I will add that).
        Keep it professional, concise, and direct for a clinician to read.
        """
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"⚠️ Error generating recommendations: {str(e)}"
