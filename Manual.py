import sleap

# Load your labeled dataset
labels = sleap.load_file("labels.1221.slp")

rows = []
for lf in labels.labeled_frames:
    frame_idx = lf.frame_idx
    for inst in lf.instances:
        for node, pt in zip(inst.skeleton.nodes, inst.points):
            # Some manual labels may not have scores, but coordinates are always there
            x, y = pt.x, pt.y
            score = getattr(pt, "score", 1.0)  # fallback score = 1.0
            rows.append({
                "frame": frame_idx,
                "instance": inst.track.name if inst.track else None,
                "node": node.name,
                "x": x,
                "y": y,
                "score": score
            })

import pandas as pd
df = pd.DataFrame(rows)
print(df.head())
df.to_csv("predictions_manual_1221.csv", index=False)
