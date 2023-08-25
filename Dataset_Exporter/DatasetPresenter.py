from datetime import datetime
import os
import fiftyone as fo
import shutil
import fiftyone.utils.random as four

from Exporter import CSVImageClassificationDatasetExporter

name = f"CODE3K"
dataset_type = fo.types.CVATImageDataset
export_dataset_type = fo.types.TFImageClassificationDataset
DATA_PATH = "C:\CODE\Dataset\DataSources"
EXPORT_PATH = "C:\CODE\Dataset\Export"
data_path_exists = os.path.exists(DATA_PATH)

existing_datasets = fo.list_datasets()
dataset = fo.Dataset(name=name)

data_sources = os.listdir(DATA_PATH)
for data_source in data_sources:
    subset_images_path = os.path.join(DATA_PATH, data_source, "images")
    subset_ann_path = os.path.join(DATA_PATH, data_source, "annotations.xml")
    dataset.merge_dir(data_path=subset_images_path, labels_path=subset_ann_path, dataset_type=dataset_type)

# session = fo.launch_app(dataset)
# session.wait()

four.random_split(dataset, {"train": 0.7, "val": 0.15, "test": 0.15})
train_dataset = dataset.match_tags("train")
val_dataset = dataset.match_tags("val")
test_dataset = dataset.match_tags("test")

for sample in train_dataset:
    try:
        label = sample["classifications"]["classifications"][0]["label"]
        image_path = sample.filepath
        export_folder = EXPORT_PATH + "\\train\\" + label
        if not os.path.exists(export_folder):
            os.makedirs(export_folder)
        shutil.copy(image_path, export_folder)
    except:
        continue

for sample in val_dataset:
    try:
        label = sample["classifications"]["classifications"][0]["label"]
        image_path = sample.filepath
        export_folder = EXPORT_PATH + "\\val\\" + label
        if not os.path.exists(export_folder):
            os.makedirs(export_folder)
        shutil.copy(image_path, export_folder)
    except:
        continue

for sample in test_dataset:
    try:
        label = sample["classifications"]["classifications"][0]["label"]
        image_path = sample.filepath
        export_folder = EXPORT_PATH + "\\test\\" + label
        if not os.path.exists(export_folder):
            os.makedirs(export_folder)
        shutil.copy(image_path, export_folder)
    except:
        continue   


# sample = dataset.first()
# classifications = sample["classifications"]
# internal_classifications = classifications["classifications"]
# actual_classification = internal_classifications[0]["label"]

# dataset.export(export_dir=EXPORT_PATH, dataset_type=export_dataset_type)

i = 5 