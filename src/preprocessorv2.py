
# ==================================================================================
# ====================  CELL 1: SETUP & PROCESSING (Run First)  ====================
# ==================================================================================
"""
Run this cell EXACTLY ONCE to process the data.
It will generate the files in the local VM (fast), but won't save them to Drive yet.
"""

import json
import warnings
import gc
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

try:
    from google.colab import drive
except ImportError:
    pass # Local dev fallback

warnings.filterwarnings("ignore")

# ---- Google Colab Paths (Hardcoded) ----
DATA_DIR = Path("/content/colabupload") 
OUT_DIR = Path("/content/output/ml_ready")     # Local VM storage (Fast)
DRIVE_DIR = Path("/content/drive/MyDrive/MedBot_ML_Ready") # Drive (Persistent)

# Ensure local output directory exists
OUT_DIR.mkdir(parents=True, exist_ok=True)
print("INFO: Cell 1 Started - Setup & Processing")
print(f"INFO: Local Output Dir: {OUT_DIR}")

# ---- Configurable parameters ----
CHUNK_SIZE = 50000 
FAERS_HIGH_SEVERE_THRESHOLD = 0.05
WEAK_SUPERVISION_SCORE_THRESHOLD = 4

# ---- Utility helpers ----
def check_file(path: Path):
    if not path.exists():
        print(f"WARNING: missing file: {path}")
        return False
    return True

def normalize_drug(name):
    if pd.isna(name): return ""
    return str(name).lower().strip().replace("(", "").replace(")", "").replace("-", " ")

def safe_read_csv(path: Path, **kwargs):
    if not check_file(path): return None
    return pd.read_csv(path, **kwargs)

# ---- 1. Load & Compact Auxiliary Tables (In-Memory + INDEXING) ----
def load_and_compact_auxiliary():
    print("INFO: Loading and compacting auxiliary tables (Indexing for Speed)...")
    aux = {}
    
    # 1. FAERS (Small - No hadm_id)
    aux["faers"] = safe_read_csv(DATA_DIR / "faers_drug_summary.csv")
    
    # 2. Admissions (Index by hadm_id)
    adm_df = safe_read_csv(DATA_DIR / "mimic_admission_summary.csv")
    if adm_df is not None and "hadm_id" in adm_df.columns:
        # Drop subject_id from aux to avoid dup cols if merging on hadm_id only, 
        # or keep if merging on both. Pass 2 merges on subject+hadm.
        # Ideally, index by [subject_id, hadm_id] or just hadm_id. 
        # Let's Index by hadm_id for O(1) checks.
        aux["admissions"] = adm_df.set_index("hadm_id")
    else:
        aux["admissions"] = None

    # 3. ICU Stays (Aggregated)
    icustays = safe_read_csv(DATA_DIR / "mimic_icustays.csv")
    aux["icustays_map"] = icustays # Keep raw for vitals mapping
    
    # Pre-aggregate LOS
    if icustays is not None:
        icu_agg = icustays.groupby("hadm_id")["los"].sum().reset_index().rename(columns={"los":"icu_total_los"})
        aux["icustays_agg"] = icu_agg.set_index("hadm_id")
    else:
        aux["icustays_agg"] = None
    
    # 4. Labs (Pivot -> Index)
    lab_df = safe_read_csv(DATA_DIR / "mimic_lab_summary.csv")
    lab_block = build_lab_block(lab_df)
    if not lab_block.empty and "hadm_id" in lab_block.columns:
         aux["lab_block"] = lab_block.set_index("hadm_id")
    else:
         aux["lab_block"] = None
    del lab_df, lab_block
    gc.collect()
    
    # 5. Vitals (Pivot -> Index)
    vital_df = safe_read_csv(DATA_DIR / "mimic_vital_signs_summary.csv")
    vital_block = build_vital_block(vital_df, icustays)
    if not vital_block.empty and "hadm_id" in vital_block.columns:
        aux["vital_block"] = vital_block.set_index("hadm_id")
    else:
        aux["vital_block"] = None
    del vital_df, vital_block
    gc.collect()

    # 6. Flags (Index)
    d_flags = safe_read_csv(DATA_DIR / "dialysis_flags.csv")
    if d_flags is not None and "hadm_id" in d_flags.columns:
        aux["dialysis_flags"] = d_flags.set_index("hadm_id")
    else:
        aux["dialysis_flags"] = None
        
    p_flags = safe_read_csv(DATA_DIR / "pressor_flags.csv")
    if p_flags is not None and "hadm_id" in p_flags.columns:
        aux["pressor_flags"] = p_flags.set_index("hadm_id")
    else:
        aux["pressor_flags"] = None
    
    print("INFO: Auxiliary tables loaded, compacted, and indexed.")
    return aux

def scan_aux_vocab(aux):
    """Scan indexed aux tables for object columns."""
    print("INFO: Scanning Auxiliary Tables for generic vocab...")
    new_encoders = {}
    ignore_cols = ["subject_id", "hadm_id", "starttime", "drug", "drug_norm", "stay_id"]
    
    for key, item in aux.items():
        if isinstance(item, pd.DataFrame):
            df = item
            for col in df.select_dtypes(include=["object", "category"]).columns:
                 if col in ignore_cols: continue
                 if col in new_encoders: continue
                 vals = df[col].astype(str).fillna("<<NA>>").unique()
                 le = LabelEncoder()
                 le.fit(list(vals) + ["<<NA>>"])
                 new_encoders[col] = le
    return new_encoders

# ---- Logic reused from v1 (modified for standalone usage) ----

def build_faers_lookup(faers_df: pd.DataFrame):
    if faers_df is None or faers_df.empty: return {}
    faers_df = faers_df.copy()
    if "ADR_Rate" not in faers_df.columns: faers_df["ADR_Rate"] = 0.0
    if "Severe_Outcome_Rate" not in faers_df.columns: faers_df["Severe_Outcome_Rate"] = 0.0
    faers_df["drug_norm"] = faers_df["drugname"].apply(normalize_drug)
    lookup = {}
    for _, r in faers_df.iterrows():
        lookup[r["drug_norm"]] = {
            "ADR_Rate": float(r.get("ADR_Rate", 0.0) or 0.0),
            "Severe_Outcome_Rate": float(r.get("Severe_Outcome_Rate", 0.0) or 0.0),
        }
    return lookup

def build_drug_block_chunk(rx_chunk: pd.DataFrame, faers_lookup: dict):
    rx = rx_chunk.copy()
    for col in ["subject_id", "hadm_id"]:
        if col not in rx.columns: rx[col] = np.nan

    drug_col = "drug" if "drug" in rx.columns else ("drug_name" if "drug_name" in rx.columns else None)
    if drug_col is None:
        rx["drug"] = ""
        drug_col = "drug"

    rx["drug_norm"] = rx[drug_col].apply(normalize_drug)
    
    def get_faers_stats(d_norm):
        entry = faers_lookup.get(d_norm, {})
        return entry.get("ADR_Rate", 0.0), entry.get("Severe_Outcome_Rate", 0.0)
    
    stats = rx["drug_norm"].apply(get_faers_stats)
    rx["faers_adr_rate"] = [x[0] for x in stats]
    rx["faers_severe_rate"] = [x[1] for x in stats]
    rx["high_risk_drug"] = (rx["faers_severe_rate"] >= FAERS_HIGH_SEVERE_THRESHOLD).astype(int)

    if "duration_hours" in rx.columns: rx["duration_days"] = rx["duration_hours"] / 24.0
    elif "duration" in rx.columns: rx["duration_days"] = rx["duration"]
    else: rx["duration_days"] = np.nan

    dose_col = next((c for c in ["dose_val_rx", "dose", "dose_val"] if c in rx.columns), None)
    route_col = next((c for c in ["route", "admin_route"] if c in rx.columns), None)

    out = rx[["subject_id", "hadm_id", drug_col]].rename(columns={drug_col: "drug"})
    out["drug_norm"] = rx["drug_norm"]
    out["dose_val_rx"] = rx[dose_col] if dose_col is not None else np.nan
    out["route"] = rx[route_col] if route_col is not None else np.nan
    out["duration_days"] = rx["duration_days"]
    out["faers_adr_rate"] = rx["faers_adr_rate"]
    out["faers_severe_rate"] = rx["faers_severe_rate"]
    out["high_risk_drug"] = rx["high_risk_drug"]
    return out

def build_lab_block(lab_summary_df: pd.DataFrame):
    if lab_summary_df is None: return pd.DataFrame(columns=["hadm_id"])
    df = lab_summary_df.copy()
    if {"hadm_id", "lab_name", "last_value"}.issubset(df.columns):
        pivot = df.pivot_table(index="hadm_id", columns="lab_name", values="last_value", aggfunc="last").reset_index()
    elif {"hadm_id", "lab_name", "first_value", "last_value"}.issubset(df.columns):
        pivot = df.pivot_table(index="hadm_id", columns="lab_name", values="last_value", aggfunc="last").reset_index()
    elif "hadm_id" in df.columns and any(col.startswith("lab_") for col in df.columns):
        pivot = df.copy() 
    else:
        return pd.DataFrame(columns=["hadm_id"])
    cols = pivot.columns.tolist()
    pivot.columns = ["hadm_id"] + [f"lab_{str(c).lower().replace(' ', '_')}" for c in cols[1:]]
    return pivot

def build_vital_block(vital_summary_df: pd.DataFrame, icustays_df: pd.DataFrame):
    if vital_summary_df is None: return pd.DataFrame(columns=["hadm_id"])
    vs = vital_summary_df.copy()
    if {"stay_id", "vital_sign", "mean"}.issubset(vs.columns):
        vs = vs.rename(columns={"vital_sign": "vital", "mean": "mean_val"})
    elif {"stay_id", "vital", "mean"}.issubset(vs.columns):
        vs = vs.rename(columns={"mean": "mean_val"})

    if {"stay_id", "vital", "mean_val"}.issubset(vs.columns):
        pivot = vs.pivot_table(index="stay_id", columns="vital", values="mean_val", aggfunc="mean").reset_index()
        pivot.columns = ["stay_id"] + [f"vital_{str(c).lower().replace(' ', '_')}" for c in pivot.columns[1:]]
    else:
        pivot = vs.copy()

    if icustays_df is None or "stay_id" not in pivot.columns:
        return pd.DataFrame(columns=["hadm_id"])

    icu_map = icustays_df[["stay_id", "hadm_id"]].drop_duplicates()
    pivot = pivot.merge(icu_map, on="stay_id", how="left")
    agg_cols = [c for c in pivot.columns if c not in ("stay_id", "hadm_id")]
    if not agg_cols: return pd.DataFrame(columns=["hadm_id"])
    
    hadm_vitals = pivot.groupby("hadm_id")[agg_cols].mean().reset_index()
    hadm_vitals.columns = ["hadm_id"] + [f"{c}" for c in hadm_vitals.columns.tolist()[1:]]
    return hadm_vitals

def weak_supervision_labeling(df):
    score = np.zeros(len(df), dtype=float)
    score += (df.get("high_risk_drug", 0) == 1).astype(int) * 2
    score += (df.get("on_dialysis", False) == True).astype(int) * 2
    score += (df.get("aki", False) == True).astype(int) * 2
    score += (df.get("chronic_liver_disease", False) == True).astype(int) * 1
    df["weak_score"] = score
    df["ADR_flag"] = (df["weak_score"] >= WEAK_SUPERVISION_SCORE_THRESHOLD).astype(int)
    return df

# ---- Main Stream Processing & Saving ----

def run_pass_1_vocab(rx_path: Path):
    print("INFO: Pass 1 - Building Vocab (Encoders)...")
    if not rx_path.exists(): return None
    unique_vals = {"drug": set(), "route": set()}
    for chunk in tqdm(pd.read_csv(rx_path, chunksize=CHUNK_SIZE), desc="Scanning Vocab"):
        drug_col = "drug" if "drug" in chunk.columns else ("drug_name" if "drug_name" in chunk.columns else None)
        if drug_col:
            chunk["drug_temp"] = chunk[drug_col].apply(normalize_drug)
            unique_vals["drug"].update(chunk["drug_temp"].dropna().unique())
        route_col = next((c for c in ["route", "admin_route"] if c in chunk.columns), None)
        if route_col:
            unique_vals["route"].update(chunk[route_col].astype(str).fillna("<<NA>>").unique())
            
    encoders = {}
    for col, vals in unique_vals.items():
        le = LabelEncoder()
        le.fit(list(vals) + ["<<NA>>"])
        encoders[col] = le
    print("INFO: Pass 1 Complete.")
    return encoders

def run_pass_2_stream_fast(rx_path: Path, aux: dict, encoders: dict):
    print("INFO: Pass 2 - Stream Processing (Fast Index Merging)...")
    faers_lookup = build_faers_lookup(aux.get("faers"))
    X_out_path = OUT_DIR / "X_features.csv"
    y_out_path = OUT_DIR / "y_target.csv"
    if X_out_path.exists(): X_out_path.unlink()
    if y_out_path.exists(): y_out_path.unlink()
    
    first_chunk = True
    total_processed = 0
    
    for chunk in tqdm(pd.read_csv(rx_path, chunksize=CHUNK_SIZE), desc="Processing chunks"):
        # 1. Base Drug Block
        merged = build_drug_block_chunk(chunk, faers_lookup)
        
        # 2. Merge Admissions (Index Join)
        if aux.get("admissions") is not None:
            # We want to join on hadm_id. 
            # merged has hadm_id as column. aux['admissions'] has it as Index.
            # Using join (left) on index 'hadm_id'.
            if "subject_id" in aux["admissions"].columns: # drop duplicate
                aux["admissions"] = aux["admissions"].drop(columns=["subject_id"], errors='ignore')
            merged = merged.join(aux["admissions"], on="hadm_id", how="left", rsuffix="_adm")
        
        # 3. Labs (Index Join)
        if aux.get("lab_block") is not None:
            merged = merged.join(aux["lab_block"], on="hadm_id", how="left")
            
        # 4. Vitals (Index Join)
        if aux.get("vital_block") is not None:
             merged = merged.join(aux["vital_block"], on="hadm_id", how="left")
             
        # 5. ICU LOS (Index Join)
        if aux.get("icustays_agg") is not None:
             merged = merged.join(aux["icustays_agg"], on="hadm_id", how="left")
             
        # 6. Flags (Index Join)
        if aux.get("dialysis_flags") is not None:
             merged = merged.join(aux["dialysis_flags"], on="hadm_id", how="left")
        else: merged["on_dialysis"] = False
        
        if aux.get("pressor_flags") is not None:
             merged = merged.join(aux["pressor_flags"], on="hadm_id", how="left")
        else: merged["on_vasopressors"] = False

        # 7. Cleanup
        for col in ["aki","chronic_liver_disease","hypertension","diabetes_type1","diabetes_type2","ckd"]:
            if col not in merged.columns: merged[col] = False
        
        num_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
        merged[num_cols] = merged[num_cols].fillna(0)
        bool_cols = merged.select_dtypes(include=["bool"]).columns.tolist()
        merged[bool_cols] = merged[bool_cols].fillna(False)
        
        # 8. Target
        labeled_df = weak_supervision_labeling(merged)
        y_chunk = labeled_df["ADR_flag"]
        
        # 9. Encode
        drop_cols = ["ADR_flag", "drug", "drug_norm", "starttime", "subject_id", "hadm_id", "route"]
        X_chunk = labeled_df.copy()
        for col, le in encoders.items():
            if col == "drug":
                X_chunk["drug_encoded"] = le.transform(X_chunk["drug_norm"].fillna("<<NA>>"))
            elif col == "route":
                X_chunk["route_encoded"] = le.transform(X_chunk["route"].astype(str).fillna("<<NA>>"))
            elif col in X_chunk.columns:
                 X_chunk[col] = le.transform(X_chunk[col].astype(str).fillna("<<NA>>"))

        final_cols = [c for c in X_chunk.columns if c not in drop_cols and c != "drug_temp" and pd.api.types.is_numeric_dtype(X_chunk[c])]
        X_final = X_chunk[final_cols].fillna(0)

        # Write
        mode = "w" if first_chunk else "a"
        header = True if first_chunk else False
        X_final.to_csv(X_out_path, mode=mode, header=header, index=False)
        y_chunk.to_csv(y_out_path, mode=mode, header=header, index=False)
        
        if first_chunk:
             with open(OUT_DIR / "feature_manifest.json", "w") as f:
                json.dump({"features": X_final.columns.tolist()}, f, indent=2)
                
        first_chunk = False
        total_processed += len(X_final)
        del chunk, merged, labeled_df, X_chunk, X_final, y_chunk
        gc.collect()

    print(f"✓ Pass 2 Complete. Total Rows: {total_processed}")
    encoder_data = {col: le.classes_.tolist() for col, le in encoders.items()}
    with open(OUT_DIR / "encoders.json", "w") as f:
        json.dump(encoder_data, f, indent=2)

# ---- EXECUTION BLOCK (CELL 1) ---- #
# Only execute if script is run (or copied into cell)

if __name__ == "__main__":
    rx_path = DATA_DIR / "mimic_prescriptions.csv"
    if not rx_path.exists():
        print("CRITICAL: Input files not found in /content/colabupload")
    else:
        aux = load_and_compact_auxiliary()
        encoders = run_pass_1_vocab(rx_path)
        encoders.update(scan_aux_vocab(aux))
        run_pass_2_stream_fast(rx_path, aux, encoders)
        print("\nSUCCESS: Cell 1 Finished. Files stored in LOCAL VM.")


# ==================================================================================
# ====================  CELL 2: SAVE TO DRIVE (Run Second)  ========================
# ==================================================================================
# Copy this block to a second cell
"""
try:
    from google.colab import drive
    import shutil
    from pathlib import Path

    # ---- Configuration (Must match Cell 1) ----
    OUT_DIR = Path("/content/output/ml_ready")     
    DRIVE_DIR = Path("/content/drive/MyDrive/MedBot_ML_Ready")

    print("INFO: Cell 2 Started - Saving to Drive")
    print("INFO: Mounting Google Drive...")
    drive.mount('/content/drive')
    
    print(f"INFO: Copying files from {OUT_DIR} to {DRIVE_DIR}...")
    DRIVE_DIR.mkdir(parents=True, exist_ok=True)
    
    files_to_copy = ["X_features.csv", "y_target.csv", "feature_manifest.json", "encoders.json"]
    
    for filename in files_to_copy:
        src = OUT_DIR / filename
        dst = DRIVE_DIR / filename
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  ✓ Copied {filename}")
        else:
            print(f"  WARNING: {filename} not found (Did Cell 1 finish successfully?)")
            
    print("\nSUCCESS: All files saved to Google Drive.")

except Exception as e:
    print(f"ERROR: {e}")
    print("Make sure you are running this in Colab and Cell 1 completed.")
"""
