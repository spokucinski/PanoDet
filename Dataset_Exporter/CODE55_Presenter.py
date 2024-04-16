import os
import fiftyone as fo
from fiftyone import ViewField as F

DATASET = 'CODE55'
DATASET_PATH = 'data/CODE55_CVAT11'

print("Starting dataset presentation")

print("Currently available in FiftyOne datasets:")
allDatasets = fo.list_datasets()
print(allDatasets)

dataset : fo.Dataset
if DATASET not in allDatasets:
    print("CODE55 not found in the loaded datasets! Starting import...")
    dataset = fo.Dataset(name=DATASET)
    subset_images_path = os.path.join(DATASET_PATH, 'images')
    subset_ann_path = os.path.join(DATASET_PATH, 'annotations.xml')
    dataset.merge_dir(data_path=subset_images_path, labels_path=subset_ann_path, dataset_type=fo.types.CVATImageDataset)
else:
    print("CODE55 already imported. Loading...")
    dataset = fo.load_dataset(DATASET)

print("Dataset ready!")
print("Setting dataset to persist in the DB.")
dataset.persistent = True

print("Printing dataset metadata:")
print(dataset)

schema = dataset.get_field_schema()
print(schema)
for schemaField in schema:
    print(schemaField)
    field = dataset.get_field(schemaField)
dataset.compute_metadata()

classes = dataset.get_classes("detections.detections.label")

# Define some interesting plots
#plot1 = fo.NumericalHistogram(F("metadata.size_bytes") / 1024, bins=50, xlabel="image size (KB)")
#plot2 = fo.NumericalHistogram("detections.confidence", bins=50, range=[0, 1])
#plot3 = fo.CategoricalHistogram("ground_truth.detections.label", order="frequency")
# plot4 = fo.CategoricalHistogram("detections.detections.label", order="frequency")

# session = fo.launch_app(dataset)

# # Construct a custom dashboard of plots
# plot = fo.ViewGrid([plot4], init_view=dataset)
# plot.show(height=720)
# session.plots.attach(plot)

# session.wait()
# session.close()