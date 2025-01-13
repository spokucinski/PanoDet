# [TODO]
# Script used to:
# 1. Load 

import os
import fiftyone as fo
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "cvat_datasets"
DATASET_NAME = "no_items"
IMPORT_TYPE = fo.types.CVATImageDataset

# Function to compute mean object counts per room type
def compute_statistics(dataset):
    stats = {}

    # Iterate over samples and aggregate data
    for sample in dataset:
        # Extract room type from classifications
        room_type = sample.classifications.classifications[0].label

        # Extract object detections
        detections = sample.detections.detections

        # Count object labels in the detections
        object_counts = {}
        for detection in detections:
            label = detection.label
            object_counts[label] = object_counts.get(label, 0) + 1

        # Aggregate counts per room type
        if room_type not in stats:
            stats[room_type] = {}

        for label, count in object_counts.items():
            stats[room_type][label] = stats[room_type].get(label, 0) + count

    # Compute mean counts for each object type per room
    rows = []
    for room_type, counts in stats.items():
        total_samples = len([s for s in dataset if s.classifications.classifications[0].label == room_type])
        mean_counts = {label: count / total_samples for label, count in counts.items()}
        mean_counts["Room Type"] = room_type
        rows.append(mean_counts)

    # Convert to DataFrame
    df = pd.DataFrame(rows).fillna(0)

    # Sort columns by total count across all rows
    total_counts = df.drop(columns=["Room Type"]).sum(axis=0)
    sorted_columns = ["Room Type"] + total_counts.sort_values(ascending=False).index.tolist()
    df = df[sorted_columns]

    return df

def plot_statistics(df, output_path="statistics_plot_no_items.png"):
    """
    Saves the mean object counts per room type plot to disk.

    Args:
        df (pandas.DataFrame): The statistics table.
        output_path (str): Path to save the plot image.
    """
    df.set_index("Room Type").plot(kind="bar", figsize=(16, 8), stacked=True)
    plt.title("Mean Object Counts Per Room Type")
    plt.ylabel("Mean Count")
    plt.xlabel("Room Type")
    plt.legend(
        title="Object Type",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize="small",
    )
    plt.tight_layout(pad=3)  # Add padding to ensure everything fits
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Plot saved to {output_path}")

# Expected CVAT format has images in one folder and all annotations in an .xml file
dataset_images_path = os.path.join(DATA_PATH, DATASET_NAME, "images")
dataset_ann_path = os.path.join(DATA_PATH, DATASET_NAME, "annotations.xml") 

dataset = fo.Dataset.from_dir(name=DATASET_NAME, 
                              dataset_type=IMPORT_TYPE, 
                              data_path=dataset_images_path, 
                              labels_path=dataset_ann_path,
                              persistent=False)

print(f"Dataset loaded with: {len(dataset)} samples")

# Compute statistics
stats_df = compute_statistics(dataset)

# Display the table
print("Mean object counts per room type:")
print(stats_df)

# Plot the statistics
plot_statistics(stats_df)