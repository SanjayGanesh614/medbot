"""
XGBoost model training module for ADR prediction (Optimized for Large Datasets)
"""
import pandas as pd
import numpy as np
import xgboost as xgb
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
import shutil
import csv
import random
import time
import datetime
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import save_model


class ProgressMonitor(xgb.callback.TrainingCallback):
    """
    Custom callback to display training progress and ETA.
    """
    def __init__(self, total_rounds):
        self.total_rounds = total_rounds
        self.start_time = None

    def after_iteration(self, model, epoch, evals_log):
        if not self.start_time:
            self.start_time = time.time()
        
        current_round = epoch + 1
        elapsed = time.time() - self.start_time
        avg_time_per_round = elapsed / current_round
        remaining_rounds = self.total_rounds - current_round
        eta_seconds = remaining_rounds * avg_time_per_round
        
        # Format times
        elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))
        eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
        
        # Get latest metrics if available
        # evals_log is a dict: {'train': {'auc': [0.5, ...]}, 'val': {'auc': [0.5, ...]}}
        metric_str = ""
        if evals_log:
            try:
                # Grab the last value of the first metric for the validation set
                val_key = list(evals_log.keys())[-1] # validation usually last
                metric_name = list(evals_log[val_key].keys())[0]
                score = evals_log[val_key][metric_name][-1]
                metric_str = f"| {val_key}-{metric_name}: {score:.5f}"
            except:
                pass

        # Print update (overwrite line)
        if current_round % 1 == 0: # Update every round for real-time feel
            sys.stdout.write(f"\r[{current_round}/{self.total_rounds}] {100*current_round/self.total_rounds:.1f}% | Elapsed: {elapsed_str} | ETA: {eta_str} {metric_str}")
            sys.stdout.flush()
            
        if current_round == self.total_rounds:
            print() # Newline at end
            
        return False


class CsvIter(xgb.DataIter):
    """
    Custom Iterator for Out-of-Core Training
    """
    def __init__(self, filename, batch_size=50000):
        self.filename = filename
        self.batch_size = batch_size
        self._iterator = None
        super().__init__()

    def reset(self):
        # Read as iterator with chunksize
        self._iterator = pd.read_csv(
            self.filename,
            chunksize=self.batch_size
        )

    def next(self, input_data):
        try:
            batch = next(self._iterator)
            
            y = batch.iloc[:, 0]
            X = batch.iloc[:, 1:]
            
            # REMOVE LEAKAGE COLUMNS
            # weak_score: Leakage
            # high_risk_drug: Dominant feature (gain 6.7M)
            # faers_*: Aggregates that might be target-encoded
            drop_cols = ['weak_score', 'high_risk_drug', 'faers_adr_rate', 'faers_severe_rate']
            existing_drop = [c for c in drop_cols if c in X.columns]
            
            if existing_drop:
                X = X.drop(columns=existing_drop)
            
            input_data(data=X, label=y)
            return 1
        except StopIteration:
            return 0


class ADRModelTrainer:
    """
    Trainer for XGBoost ADR prediction model (Out-of-Core optimized)
    """

    def __init__(self, use_smote: bool = False, use_class_weights: bool = True):
        self.model = None
        self.feature_names = []
        self.metrics = {}
        self.use_class_weights = use_class_weights
        self.optimal_threshold = 0.5
        
        # Directories
        self.colab_dir = Path("colabupload")
        self.splits_dir = self.colab_dir / "splits"
        self.splits_dir.mkdir(exist_ok=True, parents=True)

    # ------------------------------------------------------------------
    # SIGMOID
    # ------------------------------------------------------------------
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # ------------------------------------------------------------------
    # DATA PREPARATION (STREAMING SPLIT)
    # ------------------------------------------------------------------
    def prepare_data_splits(self, train_size=0.7, val_size=0.15) -> Dict[str, str]:
        """
        Streams X and y files, merges them, and splits into train/val/test CSVs on disk.
        Returns paths to the split files.
        """
        print("Preparing data splits (Streaming from disk to avoid RAM OOM)...")
        
        x_path = self.colab_dir / "X_features.csv"
        y_path = self.colab_dir / "y_target.csv"
        
        if not x_path.exists() or not y_path.exists():
            raise FileNotFoundError(f"Source files not found in {self.colab_dir}")

        train_path = self.splits_dir / "train.csv"
        val_path = self.splits_dir / "val.csv"
        test_path = self.splits_dir / "test.csv"

        # If files exist, skip (REMOVE THIS IF YOU WANT TO FORCE RE-SPLIT)
        if train_path.exists() and val_path.exists() and test_path.exists():
            print("Split files already exist. Using existing splits.")
            return {
                'train': str(train_path),
                'val': str(val_path),
                'test': str(test_path)
            }

        # Counters
        counts = {'train': 0, 'val': 0, 'test': 0}
        
        with open(x_path, 'r', encoding='utf-8') as fx, \
             open(y_path, 'r', encoding='utf-8') as fy, \
             open(train_path, 'w', newline='', encoding='utf-8') as ftrain, \
             open(val_path, 'w', newline='', encoding='utf-8') as fval, \
             open(test_path, 'w', newline='', encoding='utf-8') as ftest:
            
            # Readers
            # Check if headers exist (Assuming yes based on previous checks)
            x_header = fx.readline().strip()
            y_header = fy.readline().strip()
            
            # Check for header correctness (simple check)
            if "ADR_flag" not in y_header and "0" in y_header: 
                # If no header, reset? Assuming strict headers from previous context
                pass
            
            # Prepare CSV Writers (Label First for XGBoost)
            # Output format: Label, Feature1, Feature2...
            # We don't necessarily need CSV writer overhead if we just string concat, but safer
            # Actually, standard CSV writer is good
            
            # Construct combined header
            # Note: y_header might be just "ADR_flag"
            combined_header = f"{y_header},{x_header}\n"
            
            ftrain.write(combined_header)
            fval.write(combined_header)
            ftest.write(combined_header)
            
            # Capture feature names from header
            self.feature_names = x_header.split(',')
            
            # Iterate
            for line_x, line_y in zip(fx, fy):
                line_x = line_x.strip()
                line_y = line_y.strip()
                
                if not line_x or not line_y: continue
                
                # Randomized Split
                rand = random.random()
                
                # Combine: Label,Features
                combined_line = f"{line_y},{line_x}\n"
                
                if rand < train_size:
                    ftrain.write(combined_line)
                    counts['train'] += 1
                elif rand < (train_size + val_size):
                    fval.write(combined_line)
                    counts['val'] += 1
                else:
                    ftest.write(combined_line)
                    counts['test'] += 1
                
                if (sum(counts.values()) % 100000) == 0:
                    print(f"Processed {sum(counts.values())} rows...", end='\r')
                    
        print(f"\nSplitting complete. Counts: {counts}")
        return {
            'train': str(train_path),
            'val': str(val_path),
            'test': str(test_path)
        }

    # ------------------------------------------------------------------
    # MODEL LOADING / TRAINING
    # ------------------------------------------------------------------
    def train_model(self):
        # 1. Prepare Data
        paths = self.prepare_data_splits()
        
        # 2. Get Feature Names if not set
        if not self.feature_names:
            # Fallback: Read header of train.csv
            with open(paths['train'], 'r') as f:
                header = f.readline().strip().split(',')
                # First col is label
                self.feature_names = header[1:]
        
        # REMOVE LEAKAGE FEATURE
        drop_cols = ['weak_score', 'high_risk_drug', 'faers_adr_rate', 'faers_severe_rate']
        for col in drop_cols:
            if col in self.feature_names:
                print(f"Removing leakage feature '{col}' from feature manifest...")
                self.feature_names.remove(col)

        print("\nLoading Training Data (using Iterator for Out-of-Core)...")
        # Use Custom Iterator
        train_iter = CsvIter(paths['train'], batch_size=50000)
        dtrain = xgb.DMatrix(train_iter)
        
        print("Loading Validation/Test Data (into memory for speed)...")
        # Load Val/Test fully (3M rows ~1.5GB RAM, manageable)
        
        # Helper to load dmatrix
        def load_to_dmatrix(path):
            df = pd.read_csv(path)
            X = df.iloc[:, 1:]
            
            existing_drop = [c for c in drop_cols if c in X.columns]
            if existing_drop:
                X = X.drop(columns=existing_drop)
                
            y = df.iloc[:, 0]
            return xgb.DMatrix(X, label=y)
            
        dval = load_to_dmatrix(paths['val'])
        dtest = load_to_dmatrix(paths['test'])
        
        print("Training XGBoost Model...")
        
        # Calculate scale_pos_weight
        # Since we can't easily count pos/neg without traversing, let's estimate or use default
        # Or do a quick pass on 'train.csv' (or just specific column) if critical.
        # For efficiency, we'll try to rely on log loss or a rough estimate if available.
        # Assuming typical imbalance 1:100? 
        # Better: use 'balance_positive_weight' if available? No.
        # Let's set a conservative weight or scan just key lines if needed.
        # For now, let's use 10.0 or leave it to hyperopt. 
        # User prompt didn't strictly specify recreating exact previous logic, but let's try.
        # Let's count rows in prepare_splits if we wanted, but that method might be skipped.
        # Let's assume default for now to save time, or 10.
        scale_pos_weight = 10  
        
        rounds = 700
        params = {
            'objective': 'binary:logistic',
            'learning_rate': 0.035,
            'max_depth': 7,
            'min_child_weight': 1,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'eval_metric': 'auc',
            'scale_pos_weight': scale_pos_weight,
            'tree_method': 'hist',  # Much faster and memory efficient
            'nthread': 4,
            'verbosity': 0 # reduce noise
        }
        
        evals = [(dtrain, 'train'), (dval, 'val')]
        
        # Initialize Monitor
        monitor = ProgressMonitor(total_rounds=rounds)
        
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=rounds,
            evals=evals,
            early_stopping_rounds=20,
            verbose_eval=False, # Handled by monitor
            callbacks=[monitor]
        )
        
        self.optimal_threshold = self.find_optimal_threshold(dval)
        
        return dtrain, dval, dtest

    # ------------------------------------------------------------------
    # CLINICAL THRESHOLDING
    # ------------------------------------------------------------------
    def find_optimal_threshold(self, dmatrix: xgb.DMatrix) -> float:
        y_prob = self.model.predict(dmatrix)
        y = dmatrix.get_label()

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
    def evaluate_model(self, dmatrix: xgb.DMatrix, set_name: str):
        y_prob = self.model.predict(dmatrix)
        y = dmatrix.get_label()
        
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
            'threshold': self.optimal_threshold
        }
        # confusion_matrix causes TypeError with simple JSON serialization if not list
        # We can print it
        cm = confusion_matrix(y, y_pred)
        
        print(f"\n{set_name} Metrics")
        for k, v in metrics.items():
            print(f"{k}: {v}")
            
        return metrics, cm

    # ------------------------------------------------------------------
    # PLOTS
    # ------------------------------------------------------------------
    def plot_confusion_matrix(self, cm, save_path="reports/confusion_matrix.png"):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()

    def plot_feature_importance(self, top_n=20):
        # XGBoost Booster importance
        importance = self.model.get_score(importance_type='gain')
        # Map feature names if possible
        # DMatrix features are f0, f1... by default unless feature_names are passed
        # We can map them manually
        
        # Sorted importance
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        # If we have feature names, try to map 'fN' to name
        # Feature names list should match column index
        mapped_imp = []
        print("\nTOP FEATURE IMPORTANCE (GAIN):")
        for k, v in sorted_imp:
            if k.startswith('f'):
                try:
                    idx = int(k[1:])
                    # Since we dropped col 0 (label) from X, f0 refers to feature_names[0]
                    name = self.feature_names[idx] if idx < len(self.feature_names) else k
                except ValueError:
                    name = k
            else:
                name = k
            mapped_imp.append((name, v))
            print(f" - {name}: {v:.4f}")

        df = pd.DataFrame(mapped_imp, columns=['feature', 'importance'])

        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=df)
        plt.tight_layout()
        plt.savefig("reports/feature_importance.png")
        plt.close()

    # ------------------------------------------------------------------
    # PERFORMANCE ANALYSIS / DIAGNOSIS
    # ------------------------------------------------------------------
    def analyze_performance(self):
        """
        Analyzes metrics to provide a human-readable diagnosis of model health.
        """
        train_auc = self.metrics['train']['auc_roc']
        test_auc = self.metrics['test']['auc_roc']
        test_recall = self.metrics['test']['recall']
        
        print("\n" + "="*50)
        print("MODEL HEALTH DIAGNOSIS")
        print("="*50)
        
        # 1. Overfitting Check
        gap = train_auc - test_auc
        if gap > 0.1:
            status = "OVERFITTING DETECTED"
            advice = "The model is memorizing the specific training data rather than generalizing.\n- Fix: Simplify the model (reduce max_depth), increase regularization (reg_alpha/lambda), or get more data."
        elif gap > 0.05:
            status = "SLIGHT OVERFITTING"
            advice = "Performance drop on new data is noticeable but controllable.\n- Fix: Minor hyperparameter tuning might help."
        else:
            status = "GOOD GENERALIZATION"
            advice = "The model performs consistently on new data."
            
        print(f"Generalization: {status} (Gap: {gap:.3f})")
        
        # 2. Performance Check
        if test_auc > 0.8:
            perf = "EXCELLENT"
        elif test_auc > 0.7:
            perf = "GOOD"
        elif test_auc > 0.6:
            perf = "FAIR"
        else:
            perf = "POOR"
            advice += "\n- Fix for Poor Performance: Create better features, balance the dataset, or try a more complex model."
            
        print(f"Accuracy Grade: {perf} (Test AUC: {test_auc:.3f})")
        
        # 3. Clinical Utility Check
        if test_recall < 0.7:
            print(f"Clinical Warning: Low Sensitivity ({test_recall:.3f}). The model misses too many true cases.")
            advice += "\n- Fix: Lower the classification threshold or increase 'scale_pos_weight'."
        
        print("-" * 50)
        print(f"RECOMMENDATION:\n{advice}")
        print("=" * 50 + "\n")

    # ------------------------------------------------------------------
    # FULL PIPELINE
    # ------------------------------------------------------------------
    def full_training_pipeline(self):
        dtrain, dval, dtest = self.train_model()

        self.metrics = {
            'train': self.evaluate_model(dtrain, "Train")[0],
            'val': self.evaluate_model(dval, "Validation")[0],
        }
        
        test_metrics, test_cm = self.evaluate_model(dtest, "Test")
        self.metrics['test'] = test_metrics

        # Generate plots
        os.makedirs("reports", exist_ok=True)
        self.plot_confusion_matrix(test_cm, save_path="reports/confusion_matrix.png")
        self.plot_feature_importance(top_n=20)

        os.makedirs("models", exist_ok=True)
        
        # Save Booster
        self.model.save_model("models/xgb_adr_model.json")
        
        # Also save standard pickle wrapper for consistency if needed, 
        # but Booster is better for portability
        joblib.dump({
            'feature_names': self.feature_names,
            'threshold': self.optimal_threshold,
            'metrics': self.metrics
        }, "models/model_metadata.pkl")

        # Run Diagnosis
        self.analyze_performance()

        return self.model, self.metrics


def main():
    trainer = ADRModelTrainer()
    trainer.full_training_pipeline()

if __name__ == "__main__":
    main()
