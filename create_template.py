import pandas as pd
import os

def create_template():
    source = "colabupload/X_features.csv"
    dest = "models/feature_template.csv"
    
    print(f"Reading header from {source}...")
    # Only read 5 rows to get schema/dtypes
    df = pd.read_csv(source, nrows=5)
    
    print(f"Saving template to {dest}...")
    # Save just the header and one empty row of zeros/defaults
    # Actually, saving the first 5 rows is fine, we just need the structure.
    # The app code uses .iloc[0:1].copy() and zeros it out anyway.
    df.to_csv(dest, index=False)
    print("Done!")

if __name__ == "__main__":
    create_template()
