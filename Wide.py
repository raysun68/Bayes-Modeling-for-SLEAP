import pandas as pd
import numpy as np

# Load CSV
df = pd.read_csv("kalman_predictions_18_55_10.csv")

# Rename for clarity
df = df.rename(columns={
    "frame": "frame_idx",
    "node": "bodypart"
})

# Make track names strings like track_0
df["track"] = df["track"].astype(int)

# If no score column exists, add default 1.0
if "score" not in df.columns:
    df["score"] = 1.0

# Pivot x, y, score separately
wide_x = df.pivot_table(
    index=["frame_idx", "track"],   # 🔑 frame FIRST
    columns="bodypart",
    values="x"
)

wide_y = df.pivot_table(
    index=["frame_idx", "track"],   # 🔑 frame FIRST
    columns="bodypart",
    values="y"
)

wide_s = df.pivot_table(
    index=["frame_idx", "track"],   # 🔑 frame FIRST
    columns="bodypart",
    values="score"
)

# Flatten column names
wide_x.columns = [f"{bp}.x" for bp in wide_x.columns]
wide_y.columns = [f"{bp}.y" for bp in wide_y.columns]
wide_s.columns = [f"{bp}.score" for bp in wide_s.columns]

# Combine into single DataFrame
wide = (
    pd.concat([wide_x, wide_y, wide_s], axis=1)
      .reset_index()
      .sort_values(["frame_idx", "track"])  # 🔑 explicit frame-major order
)

# Compute instance_score (average of all bodypart scores per row)
bp_score_cols = [c for c in wide.columns if c.endswith(".score")]
wide["instance.score"] = wide[bp_score_cols].mean(axis=1)

# Enforce column order
nodes = ["Nose", "Mid-center", "Tail-base"]
ordered_cols = ["track", "frame_idx", "instance.score"]
for n in nodes:
    ordered_cols += [f"{n}.x", f"{n}.y", f"{n}.score"]

wide = wide[ordered_cols]

# Final safety sort (SimBA-critical)
wide = wide.sort_values(["frame_idx", "track"]).reset_index(drop=True)

# Save CSV
wide.to_csv(
    "kalman_predictions_18_55_10_wide.csv",
    index=False
)

