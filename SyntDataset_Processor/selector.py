# Script used to select the topN images of a specific dataset by a specific label
# In this case:
# 1. Load a CVAT-exported dataset representing 5 types of rooms and 55 classes of objects in the set
# 2. Group the images by the room type label
# 3. Order the images in each group by the count of objects in image (descending)
# 4. Take top30 images with the highest number of annotated objects for each class.
# 5. Export the images as a YOLO-compatible dataset.

import os
import fiftyone as fo
from tqdm import tqdm

# Configuration
DATA_PATH = "cvat_datasets"
EXPORT_PATH = "cvat_datasets"
DATASET_NAME = "real"
EXPORT_DATASET_NAME = DATASET_NAME + "_TOP30"

IMPORT_TYPE = fo.types.CVATImageDataset
EXPORT_TYPE = fo.types.CVATImageDataset

# Custom list of room types to include
ROOM_TYPE_FILTER = ["Living Room", "Kitchen", "Bathroom", "Bedroom", "Office"]
TOP_N = 30

# Ensure paths exist
if not os.path.exists(DATA_PATH):
    print("Source data path does not exist! Quitting.")
    quit()

if not os.path.exists(EXPORT_PATH):
    os.makedirs(EXPORT_PATH)

# Expected CVAT format has images in one folder and all annotations in an .xml file
dataset_images_path = os.path.join(DATA_PATH, DATASET_NAME, "images")
dataset_ann_path = os.path.join(DATA_PATH, DATASET_NAME, "annotations.xml") 

# Load the dataset
dataset = fo.Dataset.from_dir(name=DATASET_NAME, 
                              dataset_type=IMPORT_TYPE, 
                              data_path=dataset_images_path, 
                              labels_path=dataset_ann_path,
                              persistent=False)
print(f"Dataset loaded with: {len(dataset)} samples")

# Group images by room type
room_groups = {}
for sample in tqdm(dataset, desc="Processing samples", unit="sample"):
    room_type = sample.classifications.classifications[0].label  # Room type label

    # Filter based on the custom room type list
    if room_type not in ROOM_TYPE_FILTER:
        continue

    if room_type not in room_groups:
        room_groups[room_type] = []

    room_groups[room_type].append(sample)

# Log summary of room types
print("\nRoom type summary (after filtering):")
for room_type, samples in room_groups.items():
    print(f"- {room_type}: {len(samples)} samples")

# Select top-N images for each room type
selected_samples = []
for room_type, samples in tqdm(room_groups.items(), desc="Processing room types", unit="room type"):
    # Order samples by the number of objects in descending order
    samples.sort(key=lambda s: len(s.detections.detections), reverse=True)

    # Take the top-N samples
    top_samples = samples[:TOP_N]
    selected_samples.extend(top_samples)

    # Detailed log for each room type
    print(f"\nSelected {len(top_samples)} images for room type: {room_type}")
    for sample in top_samples:
        image_filename = os.path.basename(sample.filepath)
        num_detections = len(sample.detections.detections)
        print(f"  - Image: {image_filename}, Detections: {num_detections}")

# Sort selected samples alphabetically by file name
selected_samples.sort(key=lambda s: os.path.basename(s.filepath))

# Create a new dataset for the selected samples
selected_dataset = fo.Dataset(name=EXPORT_DATASET_NAME)
selected_dataset.add_samples(selected_samples)

# Prepare the list of classes of objects available in the dataset
dataset_classes = []
for task_label in dataset.info['task_labels']:
    dataset_classes.append(task_label['name'])

# Export the selected dataset as a YOLO dataset
selected_dataset.export(dataset_type=EXPORT_TYPE,
                        export_dir=os.path.join(EXPORT_PATH, EXPORT_DATASET_NAME), 
                        classes=dataset_classes)
print(f"Top-{TOP_N} images for each room type exported to: {EXPORT_PATH}")
