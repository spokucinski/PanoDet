from datetime import datetime
import os
import fiftyone as fo
import shutil
import fiftyone.utils.random as four

from Exporter import CSVImageClassificationDatasetExporter
os.chdir("Dataset_Exporter")

FO_DATASET_NAME = f"Red_House_10C"
IMPORT_TYPE = fo.types.CVATImageDataset
EXPORT_TYPE = fo.types.TFImageClassificationDataset
DATA_PATH = "data\DataSources"
EXPORT_PATH = "results"
RUN_FO_SESSION = False

data_path_exists = os.path.exists(DATA_PATH)
existing_datasets = fo.list_datasets()
dataset = fo.Dataset(name=FO_DATASET_NAME)

data_sources = os.listdir(DATA_PATH)
data_sources = [datasource for datasource in data_sources if os.path.isdir(os.path.join(DATA_PATH, datasource))]
for data_source in data_sources:
    subset_images_path = os.path.join(DATA_PATH, data_source, "images")
    subset_ann_path = os.path.join(DATA_PATH, data_source, "annotations.xml")
    dataset.merge_dir(data_path=subset_images_path, labels_path=subset_ann_path, dataset_type=IMPORT_TYPE)

if RUN_FO_SESSION:
    session = fo.launch_app(dataset)
    session.wait()

four.random_split(dataset, {"train": 0.9, "val": 0, "test": 0.1})
print(dataset.count_sample_tags())

train_dataset = dataset.match_tags("train")
val_dataset = dataset.match_tags("val")
test_dataset = dataset.match_tags("test")

print(f"Processing train dataset. Count of elements: {len(train_dataset)}")
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

print(f"Processing val dataset. Count of elements: {len(val_dataset)}")
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

print(f"Processing test dataset. Count of elements: {len(test_dataset)}")
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

print("Processing ended!")