"""
Quick Setup Script for AI-CPA
Runs the complete pipeline: preprocess -> train -> evaluate
"""
import os
import sys

def main():
    print("\n" + "="*70)
    print(" AI-POWERED CLINICAL PHARMACIST ASSISTANT (AI-CPA)")
    print(" Complete Setup & Training Pipeline")
    print("="*70 + "\n")
    
    # Change to project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    
    print("📁 Project directory:", project_root)
    print()
    
    dataset = os.environ.get("TRAIN_DATASET", "vigiflow").lower()
    os.environ["TRAIN_DATASET"] = dataset
    
    # Check data files
    print("🔍 Checking for required data files...")
    required_files = []
    if dataset != "vigiflow":
        required_files = [
            "output/mimic_patient_summary.csv",
            "output/mimic_prescriptions.csv",
            "output/mimic_key_labs.csv",
            "output/faers_drug_summary.csv"
        ]
    
    missing_files = []
    for file in required_files:
        filepath = os.path.join("data", file)
        if os.path.exists(filepath):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} NOT FOUND")
            missing_files.append(file)
    
    if missing_files and dataset != "vigiflow":
        print("\n⚠️  Missing data files. Please ensure preprocessed data is in data/output/")
        return
    
    if dataset != "vigiflow":
        print("\n✓ All required data files found!\n")
    
    # Step 1: Preprocessing
    if dataset != "vigiflow":
        print("="*70)
        print("STEP 1: Data Preprocessing")
        print("="*70)
        print("Merging MIMIC-IV and FAERS data...")
        print()
        
        from src.preprocess import main as preprocess_main
        try:
            preprocess_main()
            print("\n✓ Preprocessing complete!")
        except Exception as e:
            print(f"\n✗ Preprocessing failed: {e}")
            return
    
    # Step 2: Training
    print("\n" + "="*70)
    print("STEP 2: Model Training")
    print("="*70)
    if dataset == "vigiflow":
        print("Training XGBoost classifier on VigiFlow dataset...")
    else:
        print("Training XGBoost classifier...")
    print()
    
    from src.train_xgb import main as train_main
    try:
        train_main()
        print("\n✓ Training complete!")
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        return
    
    # Step 3: SHAP Explainer
    print("\n" + "="*70)
    print("STEP 3: Creating SHAP Explainer")
    print("="*70)
    print("Generating explainability artifacts...")
    print()
    
    from src.explainability import main as shap_main
    try:
        shap_main()
        print("\n✓ SHAP explainer created!")
    except Exception as e:
        print(f"\n⚠️  SHAP creation warning: {e}")
        # Continue even if SHAP fails
    
    # Step 4: Evaluation
    print("\n" + "="*70)
    print("STEP 4: Model Evaluation & Fairness Audit")
    print("="*70)
    print("Computing fairness metrics...")
    print()
    
    from src.evaluate import main as eval_main
    try:
        eval_main()
        print("\n✓ Evaluation complete!")
    except Exception as e:
        print(f"\n⚠️  Evaluation warning: {e}")
        # Continue even if evaluation has issues
    
    # Success
    print("\n" + "="*70)
    print("🎉 AI-CPA SETUP COMPLETE!")
    print("="*70)
    print("\n✓ Model trained and saved to: models/xgb_adr_model.pkl")
    print("✓ Reports generated in: reports/")
    print("\nTo run the Streamlit app:")
    print("  streamlit run src/app.py")
    print("\nOr run directly with Python:")
    print("  python -m streamlit run src/app.py")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()

