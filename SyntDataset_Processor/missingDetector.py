# Script used to:
# 1. Load a specific CVAT-formatted dataset
# 2. Analyze its annotations file (.xml)
# 3. Go through the samples and search for missing classes
#
# Class is considered missing if it's listed as a label in the annotation file
# and there are no entities of this class found in the labelled dataset samples

import xml.etree.ElementTree as ET
import fiftyone as fo
import os

def get_defined_classes_from_xml(xml_path):
    """
    Extracts all defined classes from the CVAT annotations XML file.

    Args:
        xml_path (str): Path to the CVAT annotations XML file.

    Returns:
        set: Set of all defined classes.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Extract label names from the XML
    defined_classes = set()
    for label in root.findall(".//label"):
        name = label.find("name").text
        defined_classes.add(name)
    
    return defined_classes

def get_present_classes_from_dataset(dataset, detection_field="ground_truth"):
    """
    Extracts all present classes from the dataset.

    Args:
        dataset (fiftyone.core.Dataset): The FiftyOne dataset.
        detection_field (str): The field containing detections (bounding boxes).

    Returns:
        set: Set of all present classes.
    """
    present_classes = set()
    
    for sample in dataset:
        detections = sample[detection_field].detections
        for detection in detections:
            present_classes.add(detection.label)
    
    return present_classes

def find_missing_classes(annotations_path, dataset, detection_field="ground_truth"):
    """
    Identifies classes defined in the annotations but not present in the dataset.

    Args:
        annotations_path (str): Path to the CVAT annotations XML file.
        dataset (fiftyone.core.Dataset): The FiftyOne dataset.
        detection_field (str): The field containing detections (bounding boxes).

    Returns:
        list: List of missing classes.
    """
    defined_classes = get_defined_classes_from_xml(annotations_path)
    present_classes = get_present_classes_from_dataset(dataset, detection_field)
    
    # Find missing classes
    missing_classes = sorted(defined_classes - present_classes)
    return missing_classes

DATA_PATH = "cvat_datasets"
DATASET_NAME = "no_items"
IMPORT_TYPE = fo.types.CVATImageDataset

# Expected CVAT format has images in one folder and all annotations in an .xml file
dataset_images_path = os.path.join(DATA_PATH, DATASET_NAME, "images")
dataset_ann_path = os.path.join(DATA_PATH, DATASET_NAME, "annotations.xml") 

dataset = fo.Dataset.from_dir(name=DATASET_NAME, 
                              dataset_type=IMPORT_TYPE, 
                              data_path=dataset_images_path, 
                              labels_path=dataset_ann_path,
                              persistent=False)

# Find missing classes
missing_classes = find_missing_classes(dataset_ann_path, dataset, "detections")

print(f"Missing classes: {missing_classes}")
