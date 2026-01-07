import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

def _parse_age_to_years(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    import re
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return np.nan
    magnitude = float(match.group(1))
    if "month" in text:
        return magnitude / 12.0
    if "week" in text:
        return magnitude / 52.0
    if "day" in text:
        return magnitude / 365.0
    if "hour" in text:
        return magnitude / (365.0 * 24.0)
    return magnitude

def _load_vigiflow_excel():
    root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    default_path = root / "data" / "VigiFlow_Excel_15102025_104021.xlsx"
    excel_path = Path(os.environ.get("VIGIFLOW_EXCEL_PATH", str(default_path)))
    reports = pd.read_excel(excel_path, sheet_name="Reports")
    drugs = pd.read_excel(excel_path, sheet_name="Drugs")
    reactions = pd.read_excel(excel_path, sheet_name="Reactions")
    return reports, drugs, reactions

def _aggregate_drug_features(drugs: pd.DataFrame) -> pd.DataFrame:
    grouped = drugs.groupby("Safety report id").agg(
        drug_total_count=("Safety report id", "size"),
        drug_unique_whodrug=("Drug name (WHODrug)", pd.Series.nunique),
        suspect_drug_count=("Drug role", lambda s: s.fillna("").str.contains("suspect", case=False).sum()),
    )
    return grouped.reset_index()

def _aggregate_reaction_features(reactions: pd.DataFrame) -> pd.DataFrame:
    grouped = reactions.groupby("Safety report id").agg(
        reaction_total_count=("Safety report id", "size"),
        unique_pt_count=("PT", pd.Series.nunique),
        serious_reaction_count=("Serious", lambda s: (s == "Yes").sum()),
    )
    return grouped.reset_index()

def build_vigiflow_dataset():
    reports, drugs, reactions = _load_vigiflow_excel()
    reports = reports[reports["Serious"].isin(["Yes", "No"])].copy()
    reports["target_serious"] = reports["Serious"].map({"Yes": 1, "No": 0})
    reports["age_onset_years"] = reports["Age at onset of reaction"].apply(_parse_age_to_years)
    reports["body_weight"] = reports["Body weight (kg)"]
    drug_features = _aggregate_drug_features(drugs)
    reaction_features = _aggregate_reaction_features(reactions)
    merged = reports.merge(drug_features, on="Safety report id", how="left").merge(
        reaction_features, on="Safety report id", how="left"
    )
    numeric_fill_cols = [
        "drug_total_count",
        "drug_unique_whodrug",
        "suspect_drug_count",
        "reaction_total_count",
        "unique_pt_count",
        "serious_reaction_count",
    ]
    merged[numeric_fill_cols] = merged[numeric_fill_cols].fillna(0)
    feature_cols = [
        "Report type",
        "Age group",
        "Sex",
        "Pregnant",
        "Lactating",
        "Drug role",
        "Drug name (WHODrug)",
        "Reaction / event (MedDRA)",
        "Outcome",
        "Country of reporter",
        "age_onset_years",
        "body_weight",
        "drug_total_count",
        "drug_unique_whodrug",
        "suspect_drug_count",
        "reaction_total_count",
        "unique_pt_count",
        "serious_reaction_count",
    ]
    for col in feature_cols:
        if col not in merged.columns:
            merged[col] = np.nan
    X = merged[feature_cols].copy()
    y = merged["target_serious"].copy()
    return X, y

def load_vigiflow_data():
    X, y = build_vigiflow_dataset()
    numeric_features = [
        "age_onset_years",
        "body_weight",
        "drug_total_count",
        "drug_unique_whodrug",
        "suspect_drug_count",
        "reaction_total_count",
        "unique_pt_count",
        "serious_reaction_count",
    ]
    categorical_features = [
        "Report type",
        "Age group",
        "Sex",
        "Pregnant",
        "Lactating",
        "Drug role",
        "Drug name (WHODrug)",
        "Reaction / event (MedDRA)",
        "Outcome",
        "Country of reporter",
    ]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            ("cat", Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                                    ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
        ]
    )
    preprocessor.fit(X)
    Xt = preprocessor.transform(X)
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    num_names = numeric_features
    cat_names = list(cat_encoder.get_feature_names_out(categorical_features))
    feature_names = num_names + cat_names
    Xt_df = pd.DataFrame(Xt.toarray() if hasattr(Xt, "toarray") else Xt, columns=feature_names)
    return Xt_df, y, feature_names
