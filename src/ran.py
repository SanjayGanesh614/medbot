import pandas as pd
X = pd.read_csv("data/output/X_features.csv")

print([
    "lab_creatinine",
    "lab_alt",
    "lab_ast",
    "lab_bilirubin",
    "lab_egfr",
    "renal_abnormal_flag",
    "hepatic_abnormal_flag"
])

print(X[[
    "lab_creatinine",
    "lab_alt",
    "lab_ast",
    "lab_bilirubin",
    "renal_abnormal_flag",
    "hepatic_abnormal_flag"
]].head())
