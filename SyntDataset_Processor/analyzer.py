# Script used to:
# 1. Load 

# import os
# import fiftyone as fo
# import pandas as pd
# import matplotlib.pyplot as plt

# DATA_PATH = "cvat_datasets"
# DATASET_NAME = "no_items"
# IMPORT_TYPE = fo.types.CVATImageDataset

# # Function to compute mean object counts per room type
# def compute_statistics(dataset):
#     stats = {}

#     # Iterate over samples and aggregate data
#     for sample in dataset:
#         # Extract room type from classifications
#         room_type = sample.classifications.classifications[0].label

#         # Extract object detections
#         detections = sample.detections.detections

#         # Count object labels in the detections
#         object_counts = {}
#         for detection in detections:
#             label = detection.label
#             object_counts[label] = object_counts.get(label, 0) + 1

#         # Aggregate counts per room type
#         if room_type not in stats:
#             stats[room_type] = {}

#         for label, count in object_counts.items():
#             stats[room_type][label] = stats[room_type].get(label, 0) + count

#     # Compute mean counts for each object type per room
#     rows = []
#     for room_type, counts in stats.items():
#         total_samples = len([s for s in dataset if s.classifications.classifications[0].label == room_type])
#         mean_counts = {label: count / total_samples for label, count in counts.items()}
#         mean_counts["Room Type"] = room_type
#         rows.append(mean_counts)

#     # Convert to DataFrame
#     df = pd.DataFrame(rows).fillna(0)

#     # Sort columns by total count across all rows
#     total_counts = df.drop(columns=["Room Type"]).sum(axis=0)
#     sorted_columns = ["Room Type"] + total_counts.sort_values(ascending=False).index.tolist()
#     df = df[sorted_columns]

#     return df

# def plot_statistics(df, output_path="statistics_plot_no_items.png"):
#     """
#     Saves the mean object counts per room type plot to disk.

#     Args:
#         df (pandas.DataFrame): The statistics table.
#         output_path (str): Path to save the plot image.
#     """
#     df.set_index("Room Type").plot(kind="bar", figsize=(16, 8), stacked=True)
#     plt.title("Mean Object Counts Per Room Type")
#     plt.ylabel("Mean Count")
#     plt.xlabel("Room Type")
#     plt.legend(
#         title="Object Type",
#         bbox_to_anchor=(1.05, 1),
#         loc="upper left",
#         fontsize="small",
#     )
#     plt.tight_layout(pad=3)  # Add padding to ensure everything fits
#     plt.savefig(output_path, bbox_inches="tight")
#     print(f"Plot saved to {output_path}")

# # Expected CVAT format has images in one folder and all annotations in an .xml file
# dataset_images_path = os.path.join(DATA_PATH, DATASET_NAME, "images")
# dataset_ann_path = os.path.join(DATA_PATH, DATASET_NAME, "annotations.xml") 

# dataset = fo.Dataset.from_dir(name=DATASET_NAME, 
#                               dataset_type=IMPORT_TYPE, 
#                               data_path=dataset_images_path, 
#                               labels_path=dataset_ann_path,
#                               persistent=False)

# print(f"Dataset loaded with: {len(dataset)} samples")

# # Compute statistics
# stats_df = compute_statistics(dataset)

# # Display the table
# print("Mean object counts per room type:")
# print(stats_df)

# # Plot the statistics
# plot_statistics(stats_df)

import os
import fiftyone as fo
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

DATA_PATH = "cvat_datasets"
DATASETS_TO_PROCESS = ["ITEMS", "NO_ITEMS", "REAL_TOP30"]
IMPORT_TYPE = fo.types.CVATImageDataset

# Function to export statistics to CSV
def export_statistics_to_csv(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Statistics exported to {output_path}")

# Stat. 1. Function to compute general statistics:
#   - Total count of labels in a dataset
#   - Average count of labels per single image in the dataset
#   - Standard deviation of labels per single image in the dataset
def computeAggregatedStatistics(dataset):
    total_objects = 0
    total_images = len(dataset)
    object_counts = []

    # Count total objects with a progress bar
    for sample in tqdm(dataset, desc="Counting total objects", leave=False):
        num_objects = len(sample.detections.detections)
        total_objects += num_objects
        object_counts.append(num_objects)

    # Compute mean objects per image
    mean_objects_per_image = round(total_objects / total_images, 3) if total_images > 0 else 0

    # Compute standard deviation of objects per image
    std_objects_per_image = round(pd.Series(object_counts).std(), 3) if total_images > 1 else 0

    return total_objects, mean_objects_per_image, std_objects_per_image

# Stat. 2. Function to compute statistics per each of the classes:
#   - total count of specific object classes
#   - avg count of specific object classes
#   - std of specific object classes
#
# Room types are not important
def computePerObjectStatistics(dataset):
    label_counts = {}
    total_images = len(dataset)
    per_image_counts = {}

    for sample in tqdm(dataset, desc="Computing per object statistics", leave=False):
        image_counts = {}
        for detection in sample.detections.detections:
            label = detection.label
            label_counts[label] = label_counts.get(label, 0) + 1
            image_counts[label] = image_counts.get(label, 0) + 1
        
        # Store per-image counts
        for label, count in image_counts.items():
            if label not in per_image_counts:
                per_image_counts[label] = []
            per_image_counts[label].append(count)

    # Ensure all labels in label_counts are in per_image_counts
    for label in label_counts.keys():
        if label not in per_image_counts:
            per_image_counts[label] = []

    # Compute statistics for each label
    label_stats = []
    for label, total_count in label_counts.items():
        counts = per_image_counts.get(label, [])
        avg_count = round(total_count / total_images, 3) if total_images > 0 else 0
        std_count = round(pd.Series(counts).std(), 3) if counts else 0.0  # Ensure std is 0.0 for empty counts

        label_stats.append({
            "Label": label,
            "Count": total_count,
            "Avg Count": avg_count,
            "Std Count": std_count,
        })

    # Convert to DataFrame for easy manipulation
    df = pd.DataFrame(label_stats)
    df = df.sort_values(by="Count", ascending=False).reset_index(drop=True)  # Sort by total count

    return df

# Stat. 3. Function to compute mean and std object counts per room type:
# - Rows represent objects
# - Columns represent room types with "Mean" and "Std" for each room type
def computePerRoomStatistics(dataset):
    stats = {}

    # Collect counts for each room type and object
    for sample in tqdm(dataset, desc="Computing statistics per room type", leave=False):
        room_type = sample.classifications.classifications[0].label

        detections = sample.detections.detections
        object_counts = {}
        for detection in detections:
            label = detection.label
            object_counts[label] = object_counts.get(label, 0) + 1

        if room_type not in stats:
            stats[room_type] = {}

        for label, count in object_counts.items():
            if label not in stats[room_type]:
                stats[room_type][label] = []
            stats[room_type][label].append(count)

    # Prepare aggregated statistics
    all_labels = set()  # Collect all labels across all room types
    room_statistics = {}
    for room_type, object_counts in stats.items():
        total_samples = len([s for s in dataset if s.classifications.classifications[0].label == room_type])

        mean_counts = {label: round(sum(counts) / total_samples, 3) for label, counts in object_counts.items()}
        std_counts = {label: round(pd.Series(counts).std(), 3) if len(counts) > 1 else 0 for label, counts in object_counts.items()}

        for label in mean_counts:
            if label not in room_statistics:
                room_statistics[label] = {}
            room_statistics[label][f"{room_type} (Mean)"] = mean_counts[label]
            room_statistics[label][f"{room_type} (Std)"] = std_counts[label]

        # Track all unique labels
        all_labels.update(mean_counts.keys())

    # Create DataFrame
    rows = []
    for label in sorted(all_labels):
        row = {"Object": label}
        row.update(room_statistics.get(label, {}))  # Add statistics for each label
        rows.append(row)

    df = pd.DataFrame(rows).fillna(0)

    # Sort columns: "Object" first, then interleaved room types (Mean, Std)
    sorted_columns = ["Object"]
    room_types = sorted(stats.keys())  # Get room types in alphabetical order
    for room_type in room_types:
        sorted_columns.append(f"{room_type} (Mean)")
        sorted_columns.append(f"{room_type} (Std)")

    df = df[sorted_columns]  # Reorder columns
    return df

# Stat. 4. Function to compute average number of labels per room type:
#   - min 
#   - max 
#   - median 
#   - mean
#   - std
def computePerImageStatistics(dataset):
    room_type_stats = {}
    for sample in tqdm(dataset, desc="Computing average labels per room type", leave=False):
        room_type = sample.classifications.classifications[0].label
        if room_type not in room_type_stats:
            room_type_stats[room_type] = []

        room_type_stats[room_type].append(len(sample.detections.detections))

    rows = []
    for room_type, counts in room_type_stats.items():
        rows.append({
            "Room Type": room_type,
            "Min": min(counts),
            "Max": max(counts),
            "Median": pd.Series(counts).median(),
            "Mean": pd.Series(counts).mean(),
            "Std": pd.Series(counts).std()
        })

    df = pd.DataFrame(rows)
    df = df.round(3)
    return df

# Process all datasets in the directory
dataset_stats = {}
for dataset_name in DATASETS_TO_PROCESS:
    print(f"Processing dataset: {dataset_name}")
    dataset_images_path = os.path.join(DATA_PATH, dataset_name, "images")
    dataset_ann_path = os.path.join(DATA_PATH, dataset_name, "annotations.xml")

    # Load dataset
    dataset = fo.Dataset.from_dir(
        name=dataset_name,
        dataset_type=IMPORT_TYPE,
        data_path=dataset_images_path,
        labels_path=dataset_ann_path,
        persistent=False,
    )
    print(f"Dataset loaded with {len(dataset)} samples.")

    # Prepare the statistics directory
    stats_output_dir = os.path.join(DATA_PATH, dataset_name, "statistics")
    os.makedirs(stats_output_dir, exist_ok=True)

    # Stat. 1. Compute general statistics
    total_objects, mean_objects_per_image, std_objects_per_image = computeAggregatedStatistics(dataset)
    print(f"General statistics for {dataset_name}:")
    print(f"  Total objects: {total_objects}")
    print(f"  Mean objects per image: {mean_objects_per_image:.3f}")
    print(f"  Std of objects per image: {std_objects_per_image:.3f}")

    aggregated_path = os.path.join(stats_output_dir, "Stat_1_aggregated.txt")
    with open(aggregated_path, "w") as f:
        f.write(f"General statistics for {dataset_name}:\n")
        f.write(f"  Total objects: {total_objects}\n")
        f.write(f"  Mean objects per image: {mean_objects_per_image:.3f}\n")
        f.write(f"  Std of objects per image: {std_objects_per_image:.3f}\n")

    print(f"General statistics saved to {aggregated_path}")

    # Stat. 2. Compute label distribution
    label_distribution_df = computePerObjectStatistics(dataset)
    dataset_stats[dataset_name] = label_distribution_df
    output_path = os.path.join(stats_output_dir, "Stat_2_object_statistics.csv")
    label_distribution_df.to_csv(output_path, index=False)
    print(f"Label distribution statistics saved to {output_path}")

    # Stat. 3. Compute mean object counts per room type
    stats_df = computePerRoomStatistics(dataset)
    export_statistics_to_csv(stats_df, os.path.join(stats_output_dir, "Stat_3_room_type_statistics.csv"))

    # Stat. 4. Compute average labels per image per room type
    room_stats_df = computePerImageStatistics(dataset)
    export_statistics_to_csv(room_stats_df, os.path.join(stats_output_dir, "Stat_4_room_level_object_stats.csv"))

    print(f"Finished processing dataset: {dataset_name}\n")