import pandas as pd
import os

# Folder containing your CSVs
folder = "/Users/raymondsun/Dai_new"

# Files to fix
files = [
    "0428_17_59_14.csv",
    "0428_21_33_38.csv",
    "0428_19_12_12.csv",
    "0428_20_48_07.csv",
    "0428_21_19_29.csv"
]

for file in files:
    path = os.path.join(folder, file)
    
    # Load
    df = pd.read_csv(path)
    
    # --- Fix track column ---
    df["track"] = (
        df["track"]
        .astype(str)                      # ensure string
        .str.replace("track_", "", regex=False)  # remove prefix if present
        .astype(int)                      # convert to numeric
    )
    
    # Optional safety check
    unique_tracks = df["track"].unique()
    print(f"{file}: tracks found -> {unique_tracks}")
    
    # Save (overwrite original OR change name)
    df.to_csv(path, index=False)
