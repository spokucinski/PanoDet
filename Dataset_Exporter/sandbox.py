import os
os.system('pkill mongod')

import fiftyone as fo

unscrolledDataset = fo.Dataset.from_dir(dataset_dir='data/UnscrolledDataset', dataset_type=fo.types.YOLOv5Dataset)
#unscrolledTestDataset = fo.Dataset.from_dir(dataset_dir='data/UnscrolledDataset/test', dataset_type=fo.types.YOLOv5Dataset)

dataset : fo.Dataset = fo.Dataset(name="Test")
subset_images_path = os.path.join('data/12345', "images")
subset_ann_path = os.path.join('data/12345', "annotations.xml")
dataset.merge_dir(data_path=subset_images_path, labels_path=subset_ann_path, dataset_type=fo.types.CVATImageDataset)

session = fo.launch_app(dataset)

labelsInSet: list[str] = []
for sample in dataset:
    
    detectionsCollection = sample.detections
    for detection in detectionsCollection.detections:
        labelsInSet.append(detection.label)

labels = list(set(labelsInSet))
session.wait()

session.close()