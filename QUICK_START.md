# AI-CPA Quick Reference

## 🚀 Getting Started (3 Steps)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train model (one command)
python setup_and_train.py

# 3. Launch app
streamlit run src/app.py
```

## 📋 Common Commands

### Train Model

```bash
python src/train_xgb.py
```

### Run Preprocessing Only

```bash
python src/preprocess.py
```

### Generate SHAP Explainer

```bash
python src/explainability.py
```

### Run Fairness Audit

```bash
python src/evaluate.py
```

### Launch Streamlit App

```bash
streamlit run src/app.py
```

## 📂 Key Files

| File                    | Purpose                            |
| ----------------------- | ---------------------------------- |
| `src/app.py`            | Main Streamlit application         |
| `src/train_xgb.py`      | Model training pipeline            |
| `src/preprocess.py`     | Data merging & feature engineering |
| `src/explainability.py` | SHAP analysis                      |
| `src/evaluate.py`       | Fairness metrics                   |
| `src/utils.py`          | Helper functions                   |
| `setup_and_train.py`    | One-command setup script           |

## 📊 Expected Data Files

Place in `data/output/`:

- `mimic_patient_summary.csv`
- `mimic_prescriptions.csv`
- `mimic_key_labs.csv`
- `faers_drug_summary.csv`

## 🎯 Model Outputs

After training, you'll find:

- `models/xgb_adr_model.pkl` - Trained model
- `models/xgb_adr_model_metadata.pkl` - Feature names, metrics
- `models/shap_explainer.pkl` - SHAP explainer
- `reports/*.png` - Visualizations

## 🔧 Troubleshooting

**Problem:** Model not found error

```bash
# Solution: Train the model first
python src/train_xgb.py
```

**Problem:** Data files not found

```bash
# Solution: Check data is in correct location
ls data/output/
```

**Problem:** SHAP errors

```bash
# Solution: Regenerate explainer
python src/explainability.py
```

## 📈 Model Performance Expectations

- **AUC-ROC:** 0.84-0.88
- **F1 Score:** 0.75-0.80
- **Precision:** 0.80-0.85
- **Recall:** 0.70-0.78

## 🎨 Streamlit Pages

1. **Patient Entry** - Input patient data
2. **Prediction Results** - View risk scores & explanations
3. **Explainability** - SHAP analysis (global/local)
4. **Model Performance** - Metrics & fairness audit
5. **Workflow & Feedback** - User feedback form

## 💡 Tips

- Use **Manual Entry** for quick testing
- Upload **FHIR JSON** for production use (see `sample_fhir_patient.json`)
- Export reports as **CSV** for documentation
- Check **reports/** directory for visualizations
- Model retraining recommended every 3-6 months with new data

## 🔒 Data Privacy

- All processing is local (no external API calls)
- Patient data not stored between sessions
- Model files contain no patient identifiers
- Follow institutional data governance policies

## 📞 Support

Check logs in terminal if errors occur. Most issues are due to:

1. Missing trained model → Run `python src/train_xgb.py`
2. Missing data files → Verify `data/output/` contents
3. Package versions → Update with `pip install -r requirements.txt --upgrade`
