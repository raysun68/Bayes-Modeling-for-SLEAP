import pandas as pd

df = pd.read_csv("kalman_predictions_SIMBA_TRACKWISE_with_scores.csv")

# Must exist
assert "frame_idx" in df.columns
assert "track" in df.columns

# Must be integer
df["frame_idx"] = df["frame_idx"].astype(int)

# Sort correctly
df = df.sort_values(["track", "frame_idx"])

# Check continuity per track
for t, g in df.groupby("track"):
    diffs = g["frame_idx"].diff().dropna()
    if not (diffs == 1).all():
        print(f"❌ Non-contiguous frames in {t}")
