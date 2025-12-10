"""
Training pipeline for VigiFlow Excel data.

Reads the pharmacovigilance export, engineers features, splits 80/20,
fits a RandomForest model, evaluates on validation data, and stores the trained
pipeline plus stratified datasets.
"""
from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parent
EXCEL_PATH = PROJECT_ROOT / "VigiFlow_Excel_15102025_104021.xlsx"
MODEL_PATH = PROJECT_ROOT / "models" / "vigiflow_model.pkl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"


def _parse_age_to_years(value: str | float | int) -> float | None:
    """Convert textual ages such as '5 Year' or '18 Month' into numeric years."""
    if pd.isna(value):
        return np.nan

    text = str(value).strip().lower()
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


def _load_excel() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three relevant sheets from the VigiFlow export."""
    reports = pd.read_excel(EXCEL_PATH, sheet_name="Reports")
    drugs = pd.read_excel(EXCEL_PATH, sheet_name="Drugs")
    reactions = pd.read_excel(EXCEL_PATH, sheet_name="Reactions")
    return reports, drugs, reactions


def _aggregate_drug_features(drugs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-report drug statistics."""
    grouped = drugs.groupby("Safety report id").agg(
        drug_total_count=("Safety report id", "size"),
        drug_unique_whodrug=("Drug name (WHODrug)", pd.Series.nunique),
        suspect_drug_count=(
            "Drug role",
            lambda s: s.fillna("").str.contains("suspect", case=False).sum(),
        ),
    )
    return grouped.reset_index()


def _aggregate_reaction_features(reactions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-report reaction statistics."""
    grouped = reactions.groupby("Safety report id").agg(
        reaction_total_count=("Safety report id", "size"),
        unique_pt_count=("PT", pd.Series.nunique),
        serious_reaction_count=("Serious", lambda s: (s == "Yes").sum()),
    )
    return grouped.reset_index()


def build_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load Excel data and assemble the modelling dataset."""
    reports, drugs, reactions = _load_excel()

    # Filter reports with explicit serious classification
    reports = reports[reports["Serious"].isin(["Yes", "No"])].copy()
    reports["target_serious"] = reports["Serious"].map({"Yes": 1, "No": 0})
    reports["age_onset_years"] = reports["Age at onset of reaction"].apply(
        _parse_age_to_years
    )
    reports["body_weight"] = reports["Body weight (kg)"]

    drug_features = _aggregate_drug_features(drugs)
    reaction_features = _aggregate_reaction_features(reactions)

    merged = (
        reports.merge(drug_features, on="Safety report id", how="left")
        .merge(reaction_features, on="Safety report id", how="left")
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

    # Ensure expected columns exist even when empty in the source
    for col in feature_cols:
        if col not in merged.columns:
            merged[col] = np.nan

    X = merged[feature_cols].copy()
    y = merged["target_serious"].copy()
    return X, y


def train_model(X: pd.DataFrame, y: pd.Series) -> dict:
    """Train the model using an 80/20 stratified split."""
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

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
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    y_proba = clf.predict_proba(X_val)[:, 1]

    metrics = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
    auc = roc_auc_score(y_val, y_proba)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": clf,
            "metrics": metrics,
            "roc_auc": auc,
            "feature_columns": X.columns.tolist(),
        },
        MODEL_PATH,
    )

    # Persist split datasets for downstream analysis
    train_df = X_train.copy()
    train_df["target_serious"] = y_train.values
    val_df = X_val.copy()
    val_df["target_serious"] = y_val.values
    train_df.to_csv(OUTPUT_DIR / "vigiflow_train_split.csv", index=False)
    val_df.to_csv(OUTPUT_DIR / "vigiflow_val_split.csv", index=False)

    return {
        "classification_report": metrics,
        "roc_auc": auc,
        "train_size": len(X_train),
        "val_size": len(X_val),
    }


def main() -> None:
    X, y = build_dataset()
    results = train_model(X, y)

    print("=" * 70)
    print("VIGIFLOW SERIOUSNESS MODEL - TRAINING SUMMARY")
    print("=" * 70)
    print(f"Train samples: {results['train_size']}")
    print(f"Validation samples: {results['val_size']}")
    print(f"Validation ROC-AUC: {results['roc_auc']:.3f}")

    report = results["classification_report"]
    for label in ["0", "1", "weighted avg"]:
        if label in report:
            metrics = report[label]
            label_name = "Serious=No" if label == "0" else "Serious=Yes" if label == "1" else "Weighted Avg"
            print(
                f"{label_name:>15} | Precision: {metrics['precision']:.3f} | "
                f"Recall: {metrics['recall']:.3f} | F1: {metrics['f1-score']:.3f}"
            )

    print("\nArtifacts saved:")
    print(f"- Model pipeline: {MODEL_PATH}")
    print(f"- Train split CSV: {OUTPUT_DIR / 'vigiflow_train_split.csv'}")
    print(f"- Validation split CSV: {OUTPUT_DIR / 'vigiflow_val_split.csv'}")
    print("=" * 70)


if __name__ == "__main__":
    main()

