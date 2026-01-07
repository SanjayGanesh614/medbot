When i got the codebase it had preprocessing layer and trained model but After carefully going throught the XGBoost model Architecture and the data preprocessing layer and the input which the model takes vs the input which we give in frontend i had to change the whole XGBoost model structure and change the parameters to train it properly. After creating the whole project these are my observations

1. The preprocessing layer didnt take in the following feature from the dataset and aalso thus the model doesnt take these features as input while doing its prediction and hence even if we take these features as input these are not that useful.
| Category                          | Input                      | Why It Is Ignored                                            |
| --------------------------------- | -------------------------- | ------------------------------------------------------------ |
| **Laboratory Tests**              | AST                        | Stored under `additional_labs`, not mapped to model features |
|                                   | ALT                        | Not present in `X_features.csv`, never seen during training  |
|                                   | eGFR                       | Collected but not used; model only knows creatinine          |
|                                   | Albumin                    | Informational only, not part of trained feature set          |
| **Vitals**                        | Blood Pressure (SBP / DBP) | No corresponding model feature                               |
|                                   | Heart Rate                 | Not trained, not mapped                                      |
|                                   | SpO₂                       | Not trained, not mapped                                      |
|                                   | Temperature                | Not trained, not mapped                                      |
|                                   | Respiratory Rate           | Not trained, not mapped                                      |
| **Comorbidities (as categories)** | Hypertension               | Model only uses `total_diagnoses` (count), not labels        |
|                                   | Diabetes                   | Same — individual disease flags ignored                      |
|                                   | CKD                        | Renal risk captured only via labs (creatinine)               |
|                                   | CVD                        | Ignored as a category                                        |
|                                   | Cancer                     | Ignored as a category                                        |
|                                   | COPD                       | Ignored as a category                                        |
|                                   | Mental Health              | Ignored as a category                                        |
| **Medication Details**            | Dose                       | FAERS has no reliable dosing → not modeled                   |
|                                   | Route                      | Not part of trained feature space                            |
|                                   | Frequency                  | Ignored; only drug identity matters                          |


2.Features that our current model actually needs
Demographics (Model Inputs)
| Feature name (exact) | Type              | Source           |
| -------------------- | ----------------- | ---------------- |
| `anchor_age`         | int               | Age at admission |
| `gender`             | binary (M=1, F=0) | Sex              |

Utilization / Admission History (VERY IMPORTANT)
| Feature                 | Type   | Meaning                         |
| ----------------------- | ------ | ------------------------------- |
| `num_admissions`        | int    | Hospital admissions (past year) |
| `avg_los_days`          | float  | Length of stay (days)           |
| `ever_died_in_hospital` | binary | Historical mortality flag       |
| `total_diagnoses`       | int    | Count of diagnoses              |
| `total_procedures`      | int    | Count of procedures             |
| `total_prescriptions`   | int    | Active prescriptions            |
| `total_lab_tests`       | int    | Labs in recent window           |
| `num_icu_stays`         | int    | ICU admissions                  |
| `total_icu_los_days`    | float  | ICU LOS                         |

Medication / Drug Risk Features (DOMINANT SIGNALS)
| Feature                   | Type   | Meaning                        |
| ------------------------- | ------ | ------------------------------ |
| `num_drugs`               | int    | Number of active drugs         |
| `mean_adr_rate`           | float  | Mean FAERS ADR rate            |
| `max_adr_rate`            | float  | Max FAERS ADR rate             |
| `std_adr_rate`            | float  | Variability in ADR rates       |
| `mean_severe_rate`        | float  | Mean severe ADR rate           |
| `max_severe_rate`         | float  | Max severe ADR rate            |
| `num_high_risk_drugs`     | int    | Drugs above severity threshold |
| `polypharmacy_flag`       | binary | ≥5 drugs                       |
| `major_polypharmacy_flag` | binary | ≥10 drugs                      |

Laboratory Features (ONLY THESE FOUR)
| Feature                 | Type  | Meaning            |
| ----------------------- | ----- | ------------------ |
| `lab_creatinine`        | float | Renal function     |
| `lab_hemoglobin`        | float | Anemia signal      |
| `lab_platelet_count`    | float | Hematologic risk   |
| `lab_white_blood_cells` | float | Infection / stress |

3. Further these are the features that are actually avaliable in the database mainly in MIMIC IV and none of them are present in FAERS which we are currently taking in as input.

LABS
| Feature | MIMIC-IV   | FAERS |
| ------- | ---------- | ----- |
| AST     | ✅ ✅ Yes      | ❌ No  |
| ALT     | ✅ ✅ Yes      | ❌ No  |
| eGFR    | ⚠️ Derived | ❌ No  |
| Albumin | ✅ ✅ Yes      | ❌ No  |

VITALS
| Feature | MIMIC-IV   | FAERS |
| ------- | ---------- | ----- |
| BP      | ✅ ✅ Yes      | ❌ No  |
| HR      | ✅ ✅ Yes      | ❌ No  |
| SpO₂    | ⚠️ Partial | ❌ No  |
| Temp    | ✅ ✅ Yes      | ❌ No  |
| RR      | ⚠️ Partial | ❌ No  |

Comorbidities (Hypertension, CKD, Cancer, etc.)
| Feature       | MIMIC-IV | FAERS       |
| ------------- | -------- | ----------- |
| Comorbidities | ✅ ✅ Yes    | ⚠️ Indirect |
| Severity      | ❌ No     | ❌ No        |

Medication Details (dose, route, frequency)
| Feature   | MIMIC-IV   | FAERS |
| --------- | ---------- | ----- |
| Dose      | ⚠️ Partial | ❌ No |
| Route     | ⚠️ Partial | ❌ No |
| Frequency | ⚠️ Partial | ❌ No |


4. Features that can be added to the model to be used for prediction (For this we have to change the preprocessing layer fully + change Arch of XGBoost Model + Frontend to take input)

| Feature Group            | Useful  | Dataset Support | Can be added |
| ------------------------ | ------- | --------------- | ------------ |
| AST / ALT                | ✅ Yes  | MIMIC only      |✅ Yes       |
| eGFR                     | Very    | MIMIC (derived) |✅ Yes       |
| Albumin                  | Medium  | MIMIC           | Optional     |
| Vitals                   | ❌ No   | MIMIC           | ❌ No       |
| Individual comorbidities | Weak    | Both            | ❌ No       |
| Coarse disease flags     | Medium  | MIMIC           | Optional    |
| Dose / route / freq      | ❌ No   | Weak            |❌  No       |


5. Now the model performance diving this into 3 parts 
5.1 🔴The dataset

ADR prevalence is unrealistically high (87%)

This breaks metric interpretation

Handling Class Imbalance and Dataset Scale

The initial model was trained on a small cohort (100 patients) with a highly skewed ADR prevalence (~87%), which led to misleading performance metrics such as inflated accuracy and unstable balanced accuracy. This imbalance does not reflect real-world clinical settings, where ADR incidence is typically much lower.

Increasing the dataset size can improve model stability and generalization only if it introduces a more representative proportion of non-ADR cases and improves label quality. Simply adding more data with the same skewed ADR distribution would not resolve the issue and may reinforce existing bias.

To address this, the model evaluation prioritizes imbalance-aware metrics (PR-AUC, balanced accuracy, MCC) and uses class-weighted learning rather than oversampling. During inference, percentile-based thresholds are applied instead of fixed probability cutoffs to ensure clinically meaningful risk stratification.

Overall, effective imbalance handling requires both dataset expansion and careful label definition, not dataset size alone.

5.2 🔴 The UI metrics panel

Conceptually incorrect for inference

Should NOT show accuracy / AUC per patient

FIX
✔️ Show these (recommended)

Performance on held-out test set

ROC-AUC

PR-AUC (very important for ADR)

Balanced Accuracy

Recall at fixed sensitivity (e.g. Recall @ 80%)

Calibration

Calibration curve

Brier score

Dataset context

Number of patients

ADR prevalence

Evaluation method (train/val/test split)

✔️ Example UI text (use this verbatim if you want)

Model Performance (Offline Evaluation)
These metrics were computed on a held-out test set of 100 patients from MIMIC-IV + FAERS.
They represent average performance across a population, not individual predictions.

This alone removes 90% of confusion.

❌ What NOT to show in Model Performance

Accuracy alone

Per-patient metrics

Threshold-dependent metrics without explanation

🧪 What the Live Prediction window should show

For a single patient, show only:

ADR Risk Probability (%)

Risk band (Low / Moderate / High)

Explanation (SHAP)

Drug-specific risk summary

No accuracy. No recall. No AUC.

5.3 🔴 Threshold strategy

Fixed 0.5 threshold is inappropriate

You need percentile-based or recall-constrained thresholding


These above are all the faults I have noticed and the fix are mentioned too thus to rectify this we might need more time to solve all these problems.