import sleap
import pandas as pd

labels = sleap.load_file("predictions_19_12_12_allframes.slp")

rows = []
for f_idx, frame in enumerate(labels):
    for inst_idx, inst in enumerate(frame.instances):
        for node in inst.skeleton.nodes:
            pt = inst[node]
            rows.append({
                "frame": f_idx,
                "instance": inst_idx,
                "node": node.name,
                "x": pt.x,
                "y": pt.y,
                "score": pt.score,
            })

df = pd.DataFrame(rows)
print(df.head())
df.to_csv("predictions_19_12_12.csv", index=False)
