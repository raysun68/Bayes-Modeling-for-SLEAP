import sleap
import pandas as pd

labels = sleap.load_file("Manual_122634.slp")

rows = []

for f_idx, frame in enumerate(labels):

    # 🔴 KEY FIX: filter FIRST
    preds = [
        inst for inst in frame.instances
        if isinstance(inst, sleap.PredictedInstance)
    ]

    for inst_idx, inst in enumerate(preds):

        for node in inst.skeleton.nodes:
            pt = inst[node]

            if pt is None:
                continue

            rows.append({
                "frame": f_idx,
                "instance": inst_idx,
                "node": node.name,
                "x": pt.x,
                "y": pt.y,
                "score": getattr(pt, "score", None),
            })

df = pd.DataFrame(rows)

print(df.head())
print("Rows:", len(df))
print("Frames:", df["frame"].nunique())

df.to_csv("predictions_122634.csv", index=False)
