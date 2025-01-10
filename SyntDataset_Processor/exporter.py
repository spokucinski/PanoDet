# Script used to:
# 1. Load CVAT-exported image datasets
# 2. Split them into desired train/val/test subsets
# 3. Export the subsets as YOLO-ready datasets

import os
import fiftyone as fo
import fiftyone.utils.random as four

IMPORT_TYPE = fo.types.CVATImageDataset
EXPORT_TYPE = fo.types.YOLOv5Dataset

RUN_FO_SESSION = False

DATA_PATH = "cvat_datasets"
EXPORT_PATH = "yolo_datasets"

TRAIN = 0.7
VAL = 0.1
TEST = 0.2

if not os.path.exists(DATA_PATH):
    print("Source data path does not exist! Quiting.")
    quit()
else:
    print("Source data path exists, continuing...")

if TRAIN + VAL + TEST != 1.0:
    print("Expected split percentages do not sum up to 100%! Quiting.")
    quit()
else:
    print(f"Requested dataset splits: Train/Val/Test - {TRAIN}/{VAL}/{TEST}")

# Search for subfolders with datasets in the source dir
print(f"Searching for elements in: \"{DATA_PATH}\"")
dataset_paths = os.listdir(DATA_PATH)
dataset_paths = [dataset_path for dataset_path in dataset_paths if os.path.isdir(os.path.join(DATA_PATH, dataset_path))]
print(f"Found subfolders:", dataset_paths)

# Process dataset after dataset
for dataset_name in dataset_paths:
    
    print(f"Starting processing of: {dataset_name}")

    # Expected CVAT format has images in one folder and all annotations in an .xml file
    dataset_images_path = os.path.join(DATA_PATH, dataset_name, "images")
    dataset_ann_path = os.path.join(DATA_PATH, dataset_name, "annotations.xml") 

    dataset = fo.Dataset.from_dir(name=dataset_name, 
                                  dataset_type=IMPORT_TYPE, 
                                  data_path=dataset_images_path, 
                                  labels_path=dataset_ann_path,
                                  persistent=False)

    # For debugging purposes an interactive FiftyOne session may be started
    if RUN_FO_SESSION:
        session = fo.launch_app(dataset)
        session.wait()

    # Call the library for dataset split
    four.random_split(dataset, {"train": TRAIN, "val": VAL, "test": TEST})
    print("Dataset split counts:", dataset.count_sample_tags())

    # Prepare the list of classes of objects available in the dataset
    datasetClasses = []
    for taskLabel in dataset.info['task_labels']:
        datasetClasses.append(taskLabel['name'])

    # Export all dataset splits, one by one
    # FiftyOne merges the exported splits into a single dataset
    # Configuration file will be extended as new splits are loaded
    dataset_out_path = os.path.join(EXPORT_PATH, dataset_name)

    train_subset = dataset.match_tags("train")
    train_subset.export(dataset_type=EXPORT_TYPE,
                        export_dir=dataset_out_path,
                        split="train",
                        classes=datasetClasses)
    
    val_subset = dataset.match_tags("val")
    val_subset.export(dataset_type=EXPORT_TYPE,
                      export_dir=dataset_out_path,
                      split="val",
                      classes=datasetClasses)
    
    test_subset = dataset.match_tags("test")
    test_subset.export(dataset_type=EXPORT_TYPE,
                       export_dir=dataset_out_path,
                       split="test",
                       classes=datasetClasses)

    print(f"Processing of: {dataset_name} ended!")

print(f"No more datasets to be processed, quiting...")