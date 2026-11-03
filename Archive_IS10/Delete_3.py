import sleap

# Load labels file
labels = sleap.load_file("predictions_18_11_53_allframes.slp")

for lf in labels:
    # lf = labeled frame
    if len(lf.instances) >= 3:
        # Remove 3rd instance (Python index 2)
        lf.instances.pop(2)

# Save modified file
labels.save("predictions2_18_11_53_allframes.slp")
