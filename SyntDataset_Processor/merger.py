# Script used to:
# 1. Load a specific set of datasets (default: YOLOv5-formatted)
# 2. Merge their matching splits together (train + train; val + val; test + test)
# 3. Export the resulting combined set as a single, bigger dataset (default: YOLOv5-formatted)

import os
import fiftyone as fo
import yaml

# Configuration
SPLITTED_DATASETS_PATH = "yolo_datasets"
MERGED_DATASETS_PATH = "yolo_datasets"
MERGED_SETS = [
    {"name": "MIXED", "datasets": ["ITEMS", "NO_ITEMS"]},
    {"name": "ALL", "datasets": ["ITEMS", "NO_ITEMS", "REAL_TOP30"]},
]
IMPORT_TYPE = fo.types.YOLOv5Dataset
EXPORT_TYPE = fo.types.YOLOv5Dataset

SUBSETS_TO_MERGE = ["train", "val", "test"]  # Subsets to merge (e.g., train, val, test)

# Ensure output path exists
if not os.path.exists(MERGED_DATASETS_PATH):
    os.makedirs(MERGED_DATASETS_PATH)

def load_classes_from_yaml(dataset_path):
    yaml_path = os.path.join(dataset_path, "dataset.yaml")
    if not os.path.exists(yaml_path):
        print(f"Warning: 'dataset.yaml' not found in {dataset_path}.")
        return []
    
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    
    # Extract names from dictionary
    names = data.get("names", {})
    if isinstance(names, dict):
        return [names[key] for key in sorted(names.keys())]
    else:
        print(f"Warning: 'names' field in {yaml_path} is not a dictionary.")
        return []

# Function to load and merge datasets while preserving splits
def merge_datasets(dataset_names, subsets, merged_name):
    merged_dataset = fo.Dataset()

    for dataset_name in dataset_names:
        dataset_path = os.path.join(SPLITTED_DATASETS_PATH, dataset_name)
        
        # Load classes from dataset.yaml
        classes = load_classes_from_yaml(dataset_path)

        for subset in subsets:
            subset_path = os.path.join(dataset_path, "labels", subset)
            if not os.path.exists(subset_path):
                print(f"Subset {subset} not found in dataset {dataset_name}. Skipping...")
                continue

            print(f"Loading subset '{subset}' from dataset '{dataset_name}'...")
            subset_dataset = fo.Dataset.from_dir(
                dataset_type=IMPORT_TYPE,
                dataset_dir=dataset_path,
                split=subset,
                persistent=False,
            )

            # Tag each sample with its subset
            for sample in subset_dataset:
                sample.tags.append(subset)
                sample.save()

            # Add the loaded subset to the merged dataset
            merged_dataset.add_samples(subset_dataset)

    print(f"Merged dataset '{merged_name}' created with {len(merged_dataset)} samples.")
    return merged_dataset, classes

# Process each merged set
for merged_set in MERGED_SETS:
    merged_name = merged_set["name"]
    datasets_to_merge = merged_set["datasets"]

    print(f"\nMerging datasets: {datasets_to_merge} into '{merged_name}'...")
    merged_dataset, dataset_classes = merge_datasets(datasets_to_merge, SUBSETS_TO_MERGE, merged_name)

    if not dataset_classes:
        print(f"Warning: No classes found for merged dataset '{merged_name}'.")
        continue

    # Export subsets
    export_path = os.path.join(MERGED_DATASETS_PATH, merged_name)

    train_subset = merged_dataset.match_tags("train")
    train_subset.export(
        dataset_type=EXPORT_TYPE,
        export_dir=export_path,
        split="train",
        classes=dataset_classes,
    )

    val_subset = merged_dataset.match_tags("val")
    val_subset.export(
        dataset_type=EXPORT_TYPE,
        export_dir=export_path,
        split="val",
        classes=dataset_classes,
    )

    test_subset = merged_dataset.match_tags("test")
    test_subset.export(
        dataset_type=EXPORT_TYPE,
        export_dir=export_path,
        split="test",
        classes=dataset_classes,
    )

    print(f"Merged dataset '{merged_name}' exported to: {export_path}")

print("All merged datasets processed and exported!")