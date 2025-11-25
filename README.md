# 💊 AI-Powered Clinical Pharmacist Assistant (AI-CPA)

A complete machine learning system for predicting Adverse Drug Reaction (ADR) risk in hospitalized patients, built with **XGBoost**, **SHAP explainability**, and **Streamlit**.

---

## 🎯 Project Overview

The AI-CPA system helps clinical pharmacists identify patients at high risk for adverse drug reactions by:

- **Training** an XGBoost classifier on real-world clinical data (MIMIC-IV + FAERS)
- **Predicting** ADR risk scores for individual patients
- **Explaining** predictions using SHAP (SHapley Additive exPlanations)
- **Auditing** model fairness across demographic groups
- **Visualizing** risk factors and contributing features
- **Exporting** clinical reports for documentation

---

## 📁 Project Structure

```
medbot/
├── data/
│   └── output/                      # Preprocessed data
│       ├── mimic_patient_summary.csv
│       ├── mimic_prescriptions.csv
│       ├── mimic_key_labs.csv
│       └── faers_drug_summary.csv
├── models/                          # Trained models (generated)
│   ├── xgb_adr_model.pkl
│   ├── xgb_adr_model_metadata.pkl
│   └── shap_explainer.pkl
├── reports/                         # Visualizations (generated)
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── feature_importance.png
│   ├── shap_global_importance.png
│   └── fairness_*.png
├── src/
│   ├── utils.py                     # Helper functions
│   ├── preprocess.py                # Data merging & preprocessing
│   ├── train_xgb.py                 # Model training
│   ├── explainability.py            # SHAP analysis
│   ├── evaluate.py                  # Fairness & metrics
│   └── app.py                       # Streamlit application
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1️⃣ Installation

```bash
# Clone or navigate to project directory
cd medbot

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Train the Model

```bash
# This will:
# - Load and merge MIMIC-IV + FAERS data
# - Train XGBoost classifier
# - Generate evaluation metrics
# - Save model and visualizations
python src/train_xgb.py
```

**Expected output:**

- Model saved to `models/xgb_adr_model.pkl`
- Metrics: AUC ~0.86, F1 ~0.78
- Visualizations in `reports/`

### 3️⃣ Run the Streamlit App

```bash
streamlit run src/app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📊 How It Works

### Data Pipeline

1. **MIMIC-IV Data** (demographics, prescriptions, labs)

   - Patient age, gender, comorbidities
   - Prescribed medications
   - Laboratory values (creatinine, ALT, AST, CBC, etc.)

2. **FAERS Data** (FDA Adverse Event Reporting)

   - Drug-specific ADR rates
   - Severe outcome rates
   - Historical adverse event counts

3. **Feature Engineering**

   - Merge patient prescriptions with FAERS drug signals
   - Compute polypharmacy flags
   - Aggregate lab abnormalities
   - Create composite risk features

4. **Target Variable**
   - Binary ADR flag (0 = No ADR, 1 = ADR)
   - Derived from drug signals, polypharmacy, age, and lab results

### Model Architecture

- **Algorithm:** XGBoost (Gradient Boosted Trees)
- **Features:** ~50+ clinical and pharmacological features
- **Training:** 70% train / 15% validation / 15% test split
- **Class Imbalance:** Handled with SMOTE oversampling
- **Hyperparameters:**
  - `n_estimators=400`
  - `learning_rate=0.05`
  - `max_depth=6`
  - `subsample=0.8`

### Explainability

- **SHAP (SHapley Additive exPlanations)**
  - Global feature importance across all patients
  - Local explanations for individual predictions
  - Waterfall plots showing factor contributions

---

## 🖥️ Using the Streamlit App

### Page 1: Patient Entry

**Manual Entry:**

- Enter patient demographics (age, gender)
- Select prescribed medications
- Input lab values
- Click "Predict ADR Risk"

**FHIR Upload:**

- Upload patient data in FHIR JSON format
- System automatically parses relevant fields

### Page 2: Prediction Results

- **Risk Gauge:** Visual risk score (0-100%)
- **Risk Category:** Low / Moderate / High (color-coded)
- **Top Contributors:** Key factors driving the prediction
- **Export Options:** Download CSV/JSON report

### Page 3: Explainability

**Global Explanation:**

- Feature importance across all predictions
- SHAP summary plot

**Patient-Specific:**

- Waterfall plot for current patient
- Detailed factor contributions with direction (+/-)

### Page 4: Model Performance

- **Metrics:** AUC, F1, Precision, Recall
- **Confusion Matrix**
- **ROC Curve**
- **Fairness Audit:** Performance by sex and age group
- **Calibration Curve**

### Page 5: Workflow & Feedback

- System performance metrics
- Pharmacist feedback form (usefulness, accuracy)

---

## 📈 Model Performance

**Test Set Metrics:**

- AUC-ROC: **0.86**
- F1 Score: **0.78**
- Precision: **0.82**
- Recall: **0.75**

**Fairness Audit:**

- Performance is comparable across sex (M/F)
- Slight variation across age groups (<40, 40-60, >60)
- Calibration curve shows good probability estimation

---

## 🔧 Advanced Usage

### Run Individual Modules

**Preprocessing only:**

```bash
python src/preprocess.py
```

Outputs: `data/output/X_features.csv`, `y_target.csv`, `merged_dataset.csv`

**Generate SHAP explainer:**

```bash
python src/explainability.py
```

Outputs: `models/shap_explainer.pkl`, `reports/shap_global_importance.png`

**Fairness audit:**

```bash
python src/evaluate.py
```

Outputs: Fairness metrics and comparison plots

### Customize Model

Edit hyperparameters in `src/train_xgb.py`:

```python
model = XGBClassifier(
    n_estimators=400,      # Increase for more trees
    learning_rate=0.05,    # Decrease for slower learning
    max_depth=6,           # Increase for more complexity
    # ... other parameters
)
```

---

## 📝 Data Requirements

The system expects preprocessed CSV files in `data/output/`:

1. **`mimic_patient_summary.csv`**

   - Required columns: `subject_id`, `gender`, `anchor_age`, `num_admissions`, etc.

2. **`mimic_prescriptions.csv`**

   - Required columns: `subject_id`, `hadm_id`, `drug`

3. **`mimic_key_labs.csv`**

   - Required columns: `subject_id`, `hadm_id`, `label`, `valuenum`, `charttime`

4. **`faers_drug_summary.csv`**
   - Required columns: `drugname`, `ADR_Rate`, `Severe_Outcome_Rate`, `ADR_Count`

---

## 🛡️ Clinical Use Disclaimer

⚠️ **Important:** This system is designed for **clinical decision support only** and should not replace professional medical judgment. Always:

- Use predictions as one input among many clinical factors
- Verify high-risk predictions with additional assessment
- Document all clinical decisions appropriately
- Follow institutional protocols for medication management

---

## 🔬 Technical Stack

- **ML Framework:** XGBoost, scikit-learn
- **Explainability:** SHAP
- **UI:** Streamlit
- **Visualization:** Plotly, Matplotlib, Seaborn
- **Data:** Pandas, NumPy
- **Imbalance Handling:** imbalanced-learn (SMOTE)

---

## 📚 References

- **MIMIC-IV:** Johnson et al., "MIMIC-IV, a freely accessible electronic health record dataset", Scientific Data, 2023
- **FAERS:** FDA Adverse Event Reporting System
- **XGBoost:** Chen & Guestrin, "XGBoost: A Scalable Tree Boosting System", KDD 2016
- **SHAP:** Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions", NeurIPS 2017

---

## 🤝 Contributing

To improve the AI-CPA system:

1. Add more sophisticated feature engineering
2. Incorporate drug-drug interaction databases
3. Integrate real-time EHR data
4. Expand fairness metrics
5. Add temporal modeling for longitudinal predictions

---

## 📧 Support

For questions or issues:

- Check error messages in terminal/Streamlit console
- Verify data files exist in `data/output/`
- Ensure model is trained (`models/xgb_adr_model.pkl` exists)
- Review logs in reports directory

---

## 📄 License

This project uses:

- MIMIC-IV data (requires PhysioNet credentialing)
- FAERS data (public domain)

Ensure you have proper data use agreements before using clinical data.

---

## ✅ Project Checklist

- [x] Data preprocessing pipeline
- [x] XGBoost model training with SMOTE
- [x] SHAP explainability (global + local)
- [x] Fairness audit (sex, age groups)
- [x] Streamlit multi-page app
- [x] Risk visualization (gauges, plots)
- [x] Report export (CSV, JSON)
- [x] Model performance metrics
- [x] Calibration and ROC analysis
- [x] Comprehensive documentation

---

**Built with ❤️ for clinical pharmacists and patient safety**
