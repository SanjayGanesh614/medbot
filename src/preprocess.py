"""
Data preprocessing module for AI-CPA system (FINAL)
Merges MIMIC-IV and FAERS data for ADR model training
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import os
import sys
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import normalize_drug_name, handle_missing_values


class MIMICFAERSPreprocessor:
    """
    Preprocessor to merge MIMIC-IV and FAERS datasets
    (extended with organ status & advanced labs)
    """

    def __init__(self, data_dir: str = None):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = data_dir or os.path.join(project_root, "data", "output")
        self.label_encoders = {}
        self.feature_names = []

        # Clinical thresholds
        self.lab_thresholds = {
            "creatinine": 1.5,
            "alt": 40,
            "ast": 40,
            "bilirubin": 1.2,
            "alp": 120,
            "hemoglobin": 10,
            "platelets": 150,
            "wbc": 12
        }

    # --------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------
    def load_data(self):
        patients = pd.read_csv(f"{self.data_dir}/mimic_patient_summary.csv")
        prescriptions = pd.read_csv(f"{self.data_dir}/mimic_prescriptions.csv")
        labs = pd.read_csv(f"{self.data_dir}/mimic_key_labs.csv")
        faers = pd.read_csv(f"{self.data_dir}/faers_drug_summary.csv")

        print(
            f"Loaded {len(patients)} patients, "
            f"{len(prescriptions)} prescriptions, "
            f"{len(labs)} labs, "
            f"{len(faers)} FAERS drugs"
        )
        return patients, prescriptions, labs, faers

    # --------------------------------------------------
    # FAERS LOOKUP
    # --------------------------------------------------
    def prepare_faers_lookup(self, faers: pd.DataFrame) -> Dict[str, Dict]:
        lookup = {}
        for _, row in faers.iterrows():
            name = normalize_drug_name(row["drugname"])
            lookup[name] = {
                "adr_rate": row.get("ADR_Rate", 0),
                "severe_rate": row.get("Severe_Outcome_Rate", 0)
            }
        return lookup

    # --------------------------------------------------
    # PRESCRIPTION AGGREGATION
    # --------------------------------------------------
    def aggregate_prescriptions(self, rx: pd.DataFrame, faers_lookup: Dict) -> pd.DataFrame:
        rx["drug_norm"] = rx["drug"].apply(normalize_drug_name)
        rx["adr_rate"] = rx["drug_norm"].apply(
            lambda x: faers_lookup.get(x, {}).get("adr_rate", 0)
        )
        rx["severe_rate"] = rx["drug_norm"].apply(
            lambda x: faers_lookup.get(x, {}).get("severe_rate", 0)
        )
        rx["high_risk"] = (rx["severe_rate"] > 0.05).astype(int)

        grouped = rx.groupby("subject_id").agg(
            num_drugs=("drug_norm", "nunique"),
            mean_adr_rate=("adr_rate", "mean"),
            max_adr_rate=("adr_rate", "max"),
            std_adr_rate=("adr_rate", "std"),
            mean_severe_rate=("severe_rate", "mean"),
            max_severe_rate=("severe_rate", "max"),
            num_high_risk_drugs=("high_risk", "sum")
        ).reset_index()

        grouped["std_adr_rate"] = grouped["std_adr_rate"].fillna(0)
        grouped["polypharmacy_flag"] = (grouped["num_drugs"] >= 5).astype(int)
        grouped["major_polypharmacy_flag"] = (grouped["num_drugs"] >= 10).astype(int)

        return grouped

    # --------------------------------------------------
    # LAB AGGREGATION (SAFE + EXTENDED)
    # --------------------------------------------------
    def aggregate_labs(self, labs: pd.DataFrame) -> pd.DataFrame:
        labs["charttime"] = pd.to_datetime(labs["charttime"])

        latest = (
            labs.sort_values("charttime")
            .groupby(["subject_id", "lab_name"])
            .last()
            .reset_index()
        )

        pivot = latest.pivot(
            index="subject_id",
            columns="lab_name",
            values="valuenum"
        ).reset_index()

        pivot.columns = [
            f"lab_{c.lower().replace(' ', '_')}" if c != "subject_id" else c
            for c in pivot.columns
        ]

        # Helper to always return Series
        def safe_col(col):
            return pivot[col] if col in pivot.columns else pd.Series(0, index=pivot.index)

        # Core labs
        pivot["lab_creatinine"] = safe_col("lab_creatinine")
        pivot["lab_alt"] = safe_col("lab_alt")
        pivot["lab_ast"] = safe_col("lab_ast")
        pivot["lab_bilirubin"] = safe_col("lab_bilirubin")
        pivot["lab_alp"] = safe_col("lab_alp")
        pivot["lab_hemoglobin"] = safe_col("lab_hemoglobin")
        pivot["lab_platelet_count"] = safe_col("lab_platelet_count")
        pivot["lab_white_blood_cells"] = safe_col("lab_white_blood_cells")

        # Derived labs
        pivot["lab_egfr"] = 100 / (pivot["lab_creatinine"] + 0.1)

        # Organ flags
        pivot["renal_abnormal_flag"] = (
            pivot["lab_creatinine"] > self.lab_thresholds["creatinine"]
        ).astype(int)

        pivot["hepatic_abnormal_flag"] = (
            (pivot["lab_alt"] > self.lab_thresholds["alt"]) |
            (pivot["lab_ast"] > self.lab_thresholds["ast"]) |
            (pivot["lab_bilirubin"] > self.lab_thresholds["bilirubin"])
        ).astype(int)

        pivot["anemia_flag"] = (
            pivot["lab_hemoglobin"] < self.lab_thresholds["hemoglobin"]
        ).astype(int)

        pivot["thrombocytopenia_flag"] = (
            pivot["lab_platelet_count"] < self.lab_thresholds["platelets"]
        ).astype(int)

        pivot["infection_flag"] = (
            pivot["lab_white_blood_cells"] > self.lab_thresholds["wbc"]
        ).astype(int)

        return pivot.fillna(0)

    # --------------------------------------------------
    # MERGE ALL
    # --------------------------------------------------
    def merge_datasets(self, patients, rx, labs):
        merged = patients.merge(rx, on="subject_id", how="left")
        merged = merged.merge(labs, on="subject_id", how="left")
        return merged.fillna(0)

    # --------------------------------------------------
    # TARGET CREATION (SIMULATED ADR)
    # --------------------------------------------------
    def create_target_variable(self, df):
        score = (
            (df["max_severe_rate"] > 0.05).astype(int) * 3 +
            (df["num_drugs"] >= 5).astype(int) * 2 +
            (df["renal_abnormal_flag"] == 1).astype(int) * 2 +
            (df["hepatic_abnormal_flag"] == 1).astype(int)
        )

        df["ADR_flag"] = (score >= 4).astype(int)
        return df

    # --------------------------------------------------
    # FEATURE PREPARATION
    # --------------------------------------------------
    def prepare_features(self, merged):
        exclude = ["subject_id", "ADR_flag"]
        X = merged.drop(columns=exclude)
        y = merged["ADR_flag"]

        for col in X.select_dtypes(include="object").columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le

        X = handle_missing_values(X, strategy="median")
        self.feature_names = X.columns.tolist()

        print(f"Final feature count: {len(self.feature_names)}")
        return X, y, self.feature_names

    # --------------------------------------------------
    # FULL PIPELINE
    # --------------------------------------------------
    def process_full_pipeline(self):
        print("\nStarting preprocessing pipeline\n")

        patients, rx, labs, faers = self.load_data()
        lookup = self.prepare_faers_lookup(faers)
        rx_agg = self.aggregate_prescriptions(rx, lookup)
        labs_agg = self.aggregate_labs(labs)
        merged = self.merge_datasets(patients, rx_agg, labs_agg)
        merged = self.create_target_variable(merged)
        X, y, features = self.prepare_features(merged)

        X.to_csv(f"{self.data_dir}/X_features.csv", index=False)
        y.to_csv(f"{self.data_dir}/y_target.csv", index=False)
        merged.to_csv(f"{self.data_dir}/merged_dataset.csv", index=False)

        print("Preprocessing complete\n")
        return X, y, features, merged


def main():
    MIMICFAERSPreprocessor().process_full_pipeline()

if __name__ == "__main__":
    main()
