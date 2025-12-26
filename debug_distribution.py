
import pandas as pd
import os

def debug():
    path = "data/output/merged_dataset.csv"
    if not os.path.exists(path):
        print("Merged dataset not found.")
        return

    df = pd.read_csv(path)
    print("Total rows:", len(df))
    
    # Re-calculate score to see distribution
    # We need to ensure columns exist. 
    # Based on preprocess.py, they should be in merged_dataset.csv
    
    # Check if columns exist
    required = ["max_severe_rate", "num_drugs", "renal_abnormal_flag", "hepatic_abnormal_flag"]
    for col in required:
        if col not in df.columns:
            print(f"Missing column: {col}")
    
    if all(col in df.columns for col in required):
        score = (
            (df["max_severe_rate"] > 0.05).astype(int) * 3 +
            (df["num_drugs"] >= 5).astype(int) * 2 +
            (df["renal_abnormal_flag"] == 1).astype(int) * 2 +
            (df["hepatic_abnormal_flag"] == 1).astype(int)
        )
        print("\nScore Distribution:")
        print(score.value_counts().sort_index())
        
        print("\nCurrent Target (ADR_flag > 4) Distribution:")
        print((score >= 4).astype(int).value_counts())
        
        # Propose better threshold
        median_score = score.median()
        print(f"\nMedian Score: {median_score}")

if __name__ == "__main__":
    debug()
