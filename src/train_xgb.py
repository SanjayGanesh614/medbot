"""
XGBoost model training module for ADR prediction
"""
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, accuracy_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
    balanced_accuracy_score, matthews_corrcoef,
    average_precision_score, log_loss
)
import joblib
import os
import sys
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import MIMICFAERSPreprocessor
from src.utils import save_model
from src.vigiflow_preprocess import load_vigiflow_data


class ADRModelTrainer:
    """
    Trainer for XGBoost ADR prediction model
    """

    def __init__(self, use_smote: bool = False, use_class_weights: bool = True):
        self.model = None
        self.feature_names = []
        self.metrics = {}
        self.splits = {}
        self.use_smote = False
        self.use_class_weights = use_class_weights
        self.class_weights = None
        self.optimal_threshold = 0.5

    # ------------------------------------------------------------------
    # SIGMOID (REQUIRED FOR LOGITRAW)
    # ------------------------------------------------------------------
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # ------------------------------------------------------------------
    # DATA LOADING
    # ------------------------------------------------------------------
    def load_or_preprocess_data(self) -> Tuple[pd.DataFrame, pd.Series, list]:
        dataset = os.environ.get("TRAIN_DATASET", "mimic_faers").lower()
        if dataset == "vigiflow":
            return load_vigiflow_data()

        feature_path = "data/output/X_features.csv"
        target_path = "data/output/y_target.csv"

        if os.path.exists(feature_path) and os.path.exists(target_path):
            print("Loading existing preprocessed data...")
            X = pd.read_csv(feature_path)
            y = pd.read_csv(target_path).squeeze()
            feature_names = X.columns.tolist()
        else:
            print("Preprocessed data not found. Running preprocessing...")
            preprocessor = MIMICFAERSPreprocessor()
            X, y, feature_names, _ = preprocessor.process_full_pipeline()

        return X, y, feature_names

    # ------------------------------------------------------------------
    # SPLITTING
    # ------------------------------------------------------------------
    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        train_size: float = 0.7,
        val_size: float = 0.15,
        random_state: int = 42
    ) -> Dict:

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, train_size=train_size,
            random_state=random_state, stratify=y
        )

        val_ratio = val_size / (1 - train_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, train_size=val_ratio,
            random_state=random_state, stratify=y_temp
        )

        self.splits = {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }

        return self.splits

    # ------------------------------------------------------------------
    # TRAINING (LOGITRAW + FIXED WEIGHTING)
    # ------------------------------------------------------------------
    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None
    ):
        print("\nTraining XGBoost Model")

        pos = (y_train == 1).sum()
        neg = (y_train == 0).sum()
        scale_pos_weight = neg / pos

        model_params = {
            'n_estimators': 700,
            'learning_rate': 0.035,
            'max_depth': 7,
            'min_child_weight': 1,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'random_state': 42,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'scale_pos_weight': scale_pos_weight,
            'use_label_encoder': False,
        }

        self.model = XGBClassifier(**model_params)

        if X_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)

        self.optimal_threshold = self.find_optimal_threshold(X_val, y_val)

    # ------------------------------------------------------------------
    # CLINICAL THRESHOLDING (SIGMOID APPLIED)
    # ------------------------------------------------------------------
    def find_optimal_threshold(self, X: pd.DataFrame, y: pd.Series) -> float:
        logits = self.model.predict(X)
        y_prob = self.sigmoid(logits)

        precision, recall, thresholds = precision_recall_curve(y, y_prob)

        min_recall = 0.85
        valid = np.where(recall[:-1] >= min_recall)[0]

        if len(valid) > 0:
            idx = valid[np.argmax(precision[valid])]
            threshold = thresholds[idx]
        else:
            threshold = 0.5

        print(f"Optimal clinical threshold: {threshold:.3f}")
        return threshold

    # ------------------------------------------------------------------
    # EVALUATION
    # ------------------------------------------------------------------
    def evaluate_model(self, X: pd.DataFrame, y: pd.Series, set_name: str):
        y_prob = self.model.predict_proba(X)[:, 1]
        y_pred = (y_prob >= self.optimal_threshold).astype(int)

        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(y, y_pred),
            'mcc': matthews_corrcoef(y, y_pred),
            'auc_roc': roc_auc_score(y, y_prob),
            'auc_pr': average_precision_score(y, y_prob),
            'log_loss': log_loss(y, y_prob),
            'threshold': self.optimal_threshold,
            'confusion_matrix': confusion_matrix(y, y_pred).tolist()
        }

        print(f"\n{set_name} Metrics")
        for k, v in metrics.items():
            if k != "confusion_matrix":
                print(f"{k}: {v}")

        return metrics

    # ------------------------------------------------------------------
    # PLOTS
    # ------------------------------------------------------------------
    def plot_confusion_matrix(self, cm, save_path="reports/confusion_matrix.png"):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()

    def plot_roc_curve(self, X, y):
        logits = self.model.predict(X)
        y_prob = self.sigmoid(logits)
        fpr, tpr, _ = roc_curve(y, y_prob)
        auc = roc_auc_score(y, y_prob)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        plt.legend()
        plt.savefig("reports/roc_curve.png")
        plt.close()

    def plot_feature_importance(self, top_n=20):
        imp = self.model.feature_importances_
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': imp
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=df.head(top_n))
        plt.savefig("reports/feature_importance.png")
        plt.close()

    # ------------------------------------------------------------------
    # RISK BANDING
    # ------------------------------------------------------------------
    @staticmethod
    def risk_band(prob: float) -> str:
        if prob < 0.25:
            return "LOW"
        elif prob < 0.6:
            return "MODERATE"
        else:
            return "HIGH"

    # ------------------------------------------------------------------
    # FULL PIPELINE
    # ------------------------------------------------------------------
    def full_training_pipeline(self):
        X, y, self.feature_names = self.load_or_preprocess_data()
        splits = self.split_data(X, y)

        self.train_model(
            splits['X_train'], splits['y_train'],
            splits['X_val'], splits['y_val']
        )

        self.metrics = {
            'train': self.evaluate_model(splits['X_train'], splits['y_train'], "Train"),
            'val': self.evaluate_model(splits['X_val'], splits['y_val'], "Validation"),
            'test': self.evaluate_model(splits['X_test'], splits['y_test'], "Test")
        }

        os.makedirs("models", exist_ok=True)
        save_model(
            self.model,
            {
                'feature_names': self.feature_names,
                'threshold': self.optimal_threshold
            },
            "models/xgb_adr_model.pkl"
        )

        return self.model, self.metrics


def main():
    trainer = ADRModelTrainer()
    trainer.full_training_pipeline()

if __name__ == "__main__":
    main()
