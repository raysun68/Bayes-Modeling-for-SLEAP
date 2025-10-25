import sleap
import h5py
import numpy as np

df = pd.read_csv("kalman_predictions.csv")

# Create arrays in the shape [frames, instances, nodes, 2]
frames = int(df["frame"].max() + 1)
instances = df["instance"].nunique()
nodes = df["node"].nunique()
arr = np.full((frames, instances, nodes, 2), np.nan)

node_order = ["Nose", "Mid-center", "Tail-base"]
for _, row in df.iterrows():
    f, i, n = int(row["frame"]), int(row["instance"]), node_order.index(row["node"])
    arr[f, i, n, 0] = row["x"]
    arr[f, i, n, 1] = row["y"]

with h5py.File("kalman_predictions.h5", "w") as f:
    f.create_dataset("predictions", data=arr)

# Load predictions
with h5py.File("kalman_predictions.h5", "r") as f:
    arr = f["predictions"][:]  # [frames, instances, nodes, 2]

node_order = ["Nose", "Mid-center", "Tail-base"]

# Create skeleton
skeleton = sleap.Skeleton()
for name in node_order:
    skeleton.add_node(name)

# Load video
video_path = "/Users/raymondsun/Downloads/Adjusted/WIN_20250120_16_29_51_Pro_adjusted_Used!.mp4"
video = sleap.Video.from_filename(video_path)

# Create Labels object
labels = sleap.Labels()
labels.videos.append(video)

frames, instances, nodes, _ = arr.shape

for f in range(frames):
    labeled_frame = sleap.LabeledFrame(video=video, frame_idx=f)

    for i in range(instances):
        inst_coords = arr[f, i]
        # Create numpy array for points
        points = np.array(inst_coords)
        instance = sleap.Instance.from_numpy(points, skeleton)
        labeled_frame.instances.append(instance)

    labels.append(labeled_frame)

# Save to SLEAP file
labels.save("kalman_predictions.slp")


