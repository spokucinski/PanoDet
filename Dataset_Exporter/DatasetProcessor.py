from datetime import datetime
import os
import fiftyone as fo
import shutil
import fiftyone.utils.random as four
import cv2
import numpy as np
import tensorflow as tf
from scipy import ndimage as nd
from matplotlib import pyplot as plt
import albumentations as A
from pathlib import Path

from Exporter import CSVImageClassificationDatasetExporter
os.chdir("Dataset_Exporter")

custom_augmentations = {
    "LivingRoom": 1,
    "Bedroom": 1,
    "Bathroom": 2,
    "Hallway": 5,
    "Kitchen": 6
}

def export_dataset(dataset, name, export_path, use_aug):
    
    print(f"Processing {name} dataset. Count of elements: {len(dataset)}")
    
    for sample in dataset:
        try:
            label = sample["classifications"]["classifications"][0]["label"]
            image_path = sample.filepath
            export_folder = export_path + f"\\{name}\\" + label
            if not os.path.exists(export_folder):
                os.makedirs(export_folder)
            shutil.copy(image_path, export_folder)

            if use_aug:
                number_of_augmentations = custom_augmentations[label]
                for i in range(number_of_augmentations):
                    augmented_image = augment_image(image_path)
                    image_name = Path(image_path).stem
                    target_file_path = f"{export_folder}\\{image_name}_aug_{i}.jpg"
                    cv2.imwrite(target_file_path, augmented_image)
        except:
            continue

def augment_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = A.Compose([
        A.Affine(mode=cv2.BORDER_WRAP, translate_percent={'x':(0, 1),'y':0}, p=1),
        A.CoarseDropout(max_holes=5, min_holes=3, max_height=0.1, max_width=0.1, p=0.15),
        A.HorizontalFlip(p=0.15),
        A.PixelDropout(dropout_prob=0.01, p=0.15),
        A.Blur(blur_limit=15, p=0.15),
        A.CLAHE(p=0.15),
        A.ColorJitter(brightness=0, contrast=0, saturation=0.1, hue=0.1, p=0.15),
        A.Equalize(mode='cv', by_channels=False, p=0.15),
        A.GaussNoise(var_limit=(50, 250), p=0.2),
        A.Sharpen(alpha=(0.05, 0.1), p=0.15),
        A.ToGray(p=0.15),
        A.RandomBrightness(limit=0.25, p=0.25),
        A.RandomContrast(limit=0.25, p=0.25)
    ])

    transformed =  transform(image=image)
    result = transformed["image"]
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    
    return result
    
FO_DATASET_NAME = f"CODE_5C_Classifier"
IMPORT_TYPE = fo.types.CVATImageDataset
EXPORT_TYPE = fo.types.TFImageClassificationDataset
DATA_PATH = "data\DataSources"
EXPORT_PATH = "results"
RUN_FO_SESSION = False
USE_AUG = True
AUG_NUM = 10

data_path_exists = os.path.exists(DATA_PATH)
existing_datasets = fo.list_datasets()
dataset : fo.Dataset = fo.Dataset(name=FO_DATASET_NAME)

data_sources = os.listdir(DATA_PATH)
data_sources = [datasource for datasource in data_sources if os.path.isdir(os.path.join(DATA_PATH, datasource))]
for data_source in data_sources:
    subset_images_path = os.path.join(DATA_PATH, data_source, "images")
    subset_ann_path = os.path.join(DATA_PATH, data_source, "annotations.xml")
    dataset.merge_dir(data_path=subset_images_path, labels_path=subset_ann_path, dataset_type=IMPORT_TYPE)

if RUN_FO_SESSION:
    session = fo.launch_app(dataset)
    session.wait()

four.random_split(dataset, {"train": 0.8, "val": 0.1, "test": 0.1})
print(dataset.count_sample_tags())

train_dataset = dataset.match_tags("train")
val_dataset = dataset.match_tags("val")
test_dataset = dataset.match_tags("test")

export_dataset(train_dataset, "train", EXPORT_PATH, use_aug=USE_AUG)
export_dataset(val_dataset, "val", EXPORT_PATH, use_aug=False)
export_dataset(test_dataset, "test", EXPORT_PATH, use_aug=False)

print("Processing ended!")