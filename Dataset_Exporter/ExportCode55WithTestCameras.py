import os
import fiftyone as fo
import json
import matplotlib.pyplot as plt
import numpy as np
import cv2
import yaml
import fiftyone.utils.random as four

MAIN_DATASET = 'CODE55'
MAIN_DATASET_PATH = 'data/CODE55_CVAT11'

C1_DATASET = 'C1'
C1_DATASET_PATH = 'data/C1'

C2_DATASET = 'C2'
C2_DATASET_PATH = 'data/C2'

C3_DATASET = 'C3'
C3_DATASET_PATH = 'data/C3'

TEST_DATASET = 'TEST'

OUTPUT_PATH = 'result'
RUN_SESSION = True
EXPORT_TYPE = fo.types.YOLOv5Dataset

LABELS = {
    "Bathtub": 0,
    "Chair": 1,
    "Table": 2,
    "TV": 3,
    "Washing Machine": 4,
    "Cabinet": 5,
    "Gaming Console": 6,
    "Sofa": 7,
    "Speaker": 8,
    "Fireplace": 9,
    "Bed": 10,
    "Wardrobe": 11,
    "Pillow": 12,
    "Nightstand": 13,
    "Toilet": 14,
    "Shower": 15,
    "Laundry Rack": 16,
    "Hair Dryer": 17,
    "Fridge": 18,
    "Microwave": 19,
    "Dishwasher": 20,
    "Stove": 21,
    "Kettle": 22,
    "Coffe Machine": 23,
    "Toaster": 24,
    "Oven": 25,
    "Lamp": 26,
    "Air Conditioning": 27,
    "Computer": 28,
    "Plant": 29,
    "Window": 30,
    "Desk": 31,
    "Door": 32,
    "Mirror": 33,
    "Socket": 34,
    "Sink": 35,
    "Aquarium": 36,
    "Painting": 37,
    "Air Purifier": 38,
    "Switch": 39,
    "Boiler": 40,
    "Rug": 41,
    "Board": 42,
    "Vase": 43,
    "Faucet": 44,
    "Curtain": 45,
    "Roller Blind": 46,
    "Shelf": 47,
    "Fire Extinguisher": 48,
    "Fan": 49,
    "Heater": 50,
    "Car": 51,
    "Phone": 52,
    "Clock": 53,
    "Alarm Sensor": 54
}

print("Currently available in FiftyOne datasets:")
allDatasets = fo.list_datasets()
print(allDatasets)
print()

main_dataset : fo.Dataset
c1_dataset: fo.Dataset
c2_dataset: fo.Dataset
c3_dataset: fo.Dataset
test_dataset: fo.Dataset = fo.Dataset(name=TEST_DATASET)

if MAIN_DATASET not in allDatasets:
    print("CODE55 not found in the loaded datasets! Starting import...")
    main_dataset = fo.Dataset(name=MAIN_DATASET)
    subset_images_path = os.path.join(MAIN_DATASET_PATH, 'images')
    subset_ann_path = os.path.join(MAIN_DATASET_PATH, 'annotations.xml')
    main_dataset.merge_dir(data_path=subset_images_path, labels_path=subset_ann_path, dataset_type=fo.types.CVATImageDataset)
else:
    print("CODE55 already imported. Loading...")
    main_dataset = fo.load_dataset(MAIN_DATASET)
main_dataset.persistent = True

if C1_DATASET not in allDatasets:
    print("C1 not found in the loaded datasets! Starting import...")
    c1_dataset = fo.Dataset(name=C1_DATASET)
    subset_images_path = os.path.join(C1_DATASET_PATH, 'images')
    subset_ann_path = os.path.join(C1_DATASET_PATH, 'annotations.xml')
    c1_dataset.merge_dir(data_path=subset_images_path, labels_path=subset_ann_path, dataset_type=fo.types.CVATImageDataset)
else:
    print("C1 already imported. Loading...")
    c1_dataset = fo.load_dataset(C1_DATASET)
c1_dataset.persistent = True

if C2_DATASET not in allDatasets:
    print("C2 not found in the loaded datasets! Starting import...")
    c2_dataset = fo.Dataset(name=C2_DATASET)
    subset_images_path = os.path.join(C2_DATASET_PATH, 'images')
    subset_ann_path = os.path.join(C2_DATASET_PATH, 'annotations.xml')
    c2_dataset.merge_dir(data_path=subset_images_path, labels_path=subset_ann_path, dataset_type=fo.types.CVATImageDataset)
else:
    print("C2 already imported. Loading...")
    c2_dataset = fo.load_dataset(C2_DATASET)
c2_dataset.persistent = True
    
if C3_DATASET not in allDatasets:
    print("C3 not found in the loaded datasets! Starting import...")
    c3_dataset = fo.Dataset(name=C3_DATASET)
    subset_images_path = os.path.join(C3_DATASET_PATH, 'images')
    subset_ann_path = os.path.join(C3_DATASET_PATH, 'annotations.xml')
    c3_dataset.merge_dir(data_path=subset_images_path, labels_path=subset_ann_path, dataset_type=fo.types.CVATImageDataset)
else:
    print("C3 already imported. Loading...")
    c3_dataset = fo.load_dataset(C3_DATASET)
c3_dataset.persistent = True

# Tag C1, C2, C3 as test
c1_dataset.tag_samples("test")
c2_dataset.tag_samples("test")
c3_dataset.tag_samples("test")

test_dataset.merge_samples(c1_dataset)
test_dataset.merge_samples(c2_dataset)
test_dataset.merge_samples(c3_dataset)

# Tag Main as rest
main_dataset.shuffle()
four.random_split(main_dataset, {"train": 0.9, "val": 0.1})
train_dataset = main_dataset.match_tags("train")
val_dataset = main_dataset.match_tags("val")

# Export all dataset splits, one by one
# FiftyOne merges the exported splits into a single dataset
# Configuration file will be extended as new splits are loaded
dataset_out_path = os.path.join(OUTPUT_PATH, "FullCode")
datasetClasses = [label for label, _ in sorted(LABELS.items(), key=lambda x: x[1])]

train_dataset.export(dataset_type=EXPORT_TYPE,
                    export_dir=dataset_out_path,
                    split="train",
                    classes=datasetClasses)

val_dataset.export(dataset_type=EXPORT_TYPE,
                    export_dir=dataset_out_path,
                    split="val",
                    classes=datasetClasses)

test_dataset.export(dataset_type=EXPORT_TYPE,
                    export_dir=dataset_out_path,
                    split="test",
                    classes=datasetClasses)