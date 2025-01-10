# Script used to:
# Augment a selected split of datasets

import albumentations as A
import fiftyone as fo
import os
import cv2
import utils
import shutil

from tqdm import tqdm

DATASET_TYPE = fo.types.YOLOv5Dataset

DATA_PATH = "yolo_datasets"
YAML_FILE_NAME = "dataset.yaml"

TMP_PATH = "augmented_images"
CLEAR_AFT_EXP = True

USE_AUG = False
SPLIT_TO_AUG = "train"
AUG_NUM = 3

RUN_FO_SESSION = False

# Search for subfolders with datasets in the source dir
print(f"Searching for elements in: \"{DATA_PATH}\"")
dataset_paths = os.listdir(DATA_PATH)
dataset_paths = [dataset_path for dataset_path in dataset_paths if os.path.isdir(os.path.join(DATA_PATH, dataset_path))]
print(f"Found subfolders:", dataset_paths)

# Process dataset after dataset
for dataset_name in dataset_paths:

    dataset_path = os.path.join(DATA_PATH, dataset_name)
    originalDataset = fo.Dataset.from_dir(name=dataset_name,
                                          dataset_type=DATASET_TYPE,
                                          dataset_dir=dataset_path,
                                          yaml_path=YAML_FILE_NAME,
                                          split=SPLIT_TO_AUG,
                                          persistent=False)

    # Define the augmentation pipeline
    transform = A.Compose([
        # Spatial-level transforms
        A.HorizontalFlip(p=0.5),
        A.PixelDropout(p=0.2),
        
        # Grid distortion makes the bounding boxes to be misaligned (slightly) with the actual objects
        # A.GridDistortion(interpolation=cv2.INTER_LANCZOS4, border_mode=cv2.BORDER_WRAP, p=0.2),
        
        # Pixel-level transforms
        A.ToGray(p=0.2),
        A.CLAHE(p=0.2),

        # Set of hue jitter makes the images highly unrealistic
        A.ColorJitter(brightness=0.25, contrast=0.4, saturation=0.5, hue=0, p=0.4),

        A.GaussianBlur(p=0.2), 
        A.ImageCompression(quality_lower=75, quality_upper=100, p=0.2), 
        A.Sharpen(p=0.25),
        A.ISONoise(color_shift=(0.1, 0.2), intensity=(0.1, 0.25), p=0.2)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids'], clip=True))

    augmentedDataset = fo.Dataset(name=dataset_name + "_AUG")

    for sample in tqdm(originalDataset, desc="Augmenting dataset samples..."):
        for aug_iter in tqdm(range(AUG_NUM), desc="Iterating augmentations...", leave=False):
            image = cv2.imread(sample.filepath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            bboxes = []
            category_ids = []
            for detection in sample.ground_truth.detections:
                bboxes.append(detection.bounding_box)
                category_ids.append(detection.label)

            yolo_bboxes = utils.convert_to_albumentations_yolo_format(bboxes)
            transformed = transform(image=image, bboxes=yolo_bboxes, category_ids=category_ids)

            filepath = sample.filepath
            filename, ext = os.path.splitext(os.path.basename(filepath))
            augmented_filename = f"{filename}_AUG{aug_iter+1}{ext}"
            augmented_image_path = os.path.join(TMP_PATH, dataset_name, augmented_filename)
            os.makedirs(os.path.dirname(augmented_image_path), exist_ok=True)
            cv2.imwrite(augmented_image_path, cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR))
                
            # Convert augmented bounding boxes back to FiftyOne format
            detections = []
            for bbox, label in zip(transformed['bboxes'], transformed['category_ids']):
                detections.append(
                    fo.Detection(
                        bounding_box=utils.convert_to_fiftyone_yolo_format(bbox),  # YOLO format
                        label=label,
                    )
                )
            
            augmented_sample = fo.Sample(filepath=augmented_image_path)
            augmented_sample["ground_truth"] = fo.Detections(detections=detections)
            augmentedDataset.add_sample(augmented_sample)

    augmentedDataset.export(dataset_type=DATASET_TYPE,
                            export_dir=dataset_path,
                            split=SPLIT_TO_AUG,
                            classes=originalDataset.default_classes)
    
    if CLEAR_AFT_EXP:
        if os.path.exists(TMP_PATH):
            shutil.rmtree(TMP_PATH)
            print(f"Temporary folder '{TMP_PATH}' has been removed.")
        else:
            print(f"Temporary folder '{TMP_PATH}' does not exist.")

    if RUN_FO_SESSION:
        session = fo.launch_app(augmentedDataset)
        session.wait()
