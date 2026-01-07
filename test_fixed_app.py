
import streamlit as st
import json
import os
import sys

# Mock Streamlit logic
class MockSessionState(dict):
    def __getattr__(self, key):
        return self.get(key)
    def __setattr__(self, key, value):
        self[key] = value

if 'session_state' not in st.__dict__:
    st.session_state = MockSessionState()

# Mock other st functions
st.error = lambda x: print(f"ST_ERROR: {x}")
st.warning = lambda x: print(f"ST_WARNING: {x}")
st.info = lambda x: print(f"ST_INFO: {x}")
st.write = lambda *x: print(f"ST_WRITE: {x}")

# Add src to path
sys.path.append(os.getcwd())

# Import app logic
from src.app import process_uploaded_patient_data
from src.utils import parse_fhir_patient

def test_prediction():
    print("--- Starting Prediction Test ---")
    
    # Load JSON
    json_path = "test_patient_full.json"
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found")
        return

    with open(json_path, 'r') as f:
        raw_json = json.load(f)
    print("Loaded JSON patient data.")

    # Parse (using updated utils logic)
    print("Parsing patient data...")
    # Simulate parse_fhir_patient behavior
    # Note: Logic in utils.py was updated to preserving keys, so this should work
    patient_data = parse_fhir_patient(raw_json)
    
    print(f"Parsed Data Keys: {list(patient_data.keys())}")
    print(f"Selected Drugs: {patient_data.get('selected_drugs')}")
    print(f"Comorbidities: {patient_data.get('comorbidities')}")

    # Process & Predict
    print("\nRunning process_uploaded_patient_data...")
    try:
        prediction = process_uploaded_patient_data(patient_data)
        
        if prediction:
            score = prediction['risk_score']
            cat = prediction['risk_category']
            p_data = prediction['patient_data']
            
            print(f"\n--- Result ---")
            print(f"Risk Score: {score:.4f}")
            print(f"Category: {cat}")
            print(f"Num Drugs: {p_data.get('num_drugs')}")
            print(f"Num High Risk Drugs: {p_data.get('num_high_risk_drugs')}")
            print(f"Mean ADR Rate: {p_data.get('mean_adr_rate', 0):.4f}")
            print(f"Max ADR Rate: {p_data.get('max_adr_rate', 0):.4f}")
            print(f"Polypharmacy: {p_data.get('polypharmacy_flag')}")
            
            if score > 0.7:
                print("\nSUCCESS: High Risk correctly predicted!")
            else:
                print("\nFAILURE: Risk score too low.")
        else:
            print("FAILURE: Prediction returned None")
            
    except Exception as e:
        print(f"FAILURE: Exception during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_prediction()
