
import sys
import os
import json
import pandas as pd
import streamlit as st

# Mock st just enough to import app
if 'session_state' not in st.__dict__:
    st.session_state = {}
st.cache_data = lambda func: func
st.cache_resource = lambda func: func

sys.path.append(os.getcwd())

from src.utils import parse_fhir_patient, load_model
# Import app to access calculate_drug_risk_features
# We need to mock st globals for app import
import src.app as app

def verify_components():
    print("--- COMPONENT VERIFICATION ---")
    
    # 1. Verify JSON Parsing
    print("\n[1] Testing JSON Parsing...")
    try:
        with open("test_patient_full.json", "r") as f:
            raw_data = json.load(f)
        
        parsed = parse_fhir_patient(raw_data)
        
        if 'selected_drugs' in parsed and len(parsed['selected_drugs']) >= 5:
            print("SUCCESS: 'selected_drugs' preserved correctly.")
            print(f"Drugs found: {len(parsed['selected_drugs'])}")
        else:
            print(f"FAILURE: 'selected_drugs' missing or empty. Keys: {list(parsed.keys())}")
            
        if 'comorbidities' in parsed and len(parsed['comorbidities']) > 0:
             print("SUCCESS: 'comorbidities' preserved.")
        else:
             print("FAILURE: 'comorbidities' missing.")
             
    except Exception as e:
        print(f"FAILURE: JSON Parsing Error: {e}")

    # 2. Verify Drug Risk Calculation (FAERS Path)
    print("\n[2] Testing Drug Risk Calculation...")
    try:
        drugs = ["Vancomycin", "Furosemide"]
        # Call the function from app.py
        risk_stats = app.calculate_drug_risk_features(drugs)
        
        print(f"Input: {drugs}")
        print(f"Output: {risk_stats}")
        
        if risk_stats['num_high_risk_drugs'] > 0:
            print("SUCCESS: High risk drugs identified (FAERS path correct).")
        elif risk_stats['mean_adr_rate'] > 0.021: # Default is 0.02
             print("SUCCESS: FAERS data loaded (rates found).")
        else:
            print("FAILURE: Returned defaults/zeros. FAERS path likely wrong.")
            
    except Exception as e:
        print(f"FAILURE: Drug Calculation Error: {e}")

    # 3. Verify Model Loading
    print("\n[3] Testing Model Loading...")
    try:
        # Construct absolute path as app does
        base_dir = os.getcwd() 
        model_path = os.path.join(base_dir, "models", "xgb_adr_model.json")
        
        model = load_model(model_path)
        if model:
            print("SUCCESS: Model loaded successfully.")
        else:
            print("FAILURE: Model is None.")
            
    except Exception as e:
        print(f"FAILURE: Model Load Error: {e}")

    # 4. Verify Encoders Loading
    print("\n[4] Testing Encoders Loading...")
    try:
        encoders = app.load_encoders_map()
        if encoders and isinstance(encoders, dict):
             print(f"SUCCESS: Encoders loaded from {os.path.join(base_dir, 'colabupload', 'encoders.json')}")
             if 'race' in encoders:
                print(f" - Race keys sample: {list(encoders['race'].keys())[:3]}")
             if 'drug' in encoders:
                print(f" - Drug encoded count: {len(encoders['drug'])}")
        else:
             print("FAILURE: Encoders not loaded or empty.")
    except Exception as e:
        print(f"FAILURE: Encoders Load Error: {e}")

if __name__ == "__main__":
    verify_components()
