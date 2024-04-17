import os
import fiftyone as fo
import json
import matplotlib.pyplot as plt
import numpy as np
import cv2

from fiftyone import ViewField as F

def plot_hist(counts, edges):
    counts = np.asarray(counts)
    edges = np.asarray(edges)
    left_edges = edges[:-1]
    widths = edges[1:] - edges[:-1]
    plt.bar(left_edges, counts, width=widths, align="edge")

DATASET = 'CODE55'
DATASET_PATH = 'data/CODE55_CVAT11'

print("Starting dataset presentation")
print()

print("Currently available in FiftyOne datasets:")
allDatasets = fo.list_datasets()
print(allDatasets)
print()

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
print()
dataset.persistent = True

print("Dataset metadata:")
print(dataset)
print()

print("Schema of the dataset's Sample:")
schema = dataset.get_field_schema()
flattenedSchema = fo.flatten_schema(schema)
fo.pprint(flattenedSchema)
print()

print("First sample details:")
firstSample = dataset.first()
for fieldName in firstSample.field_names:
    if fieldName == 'detections':
        detectionsSubset = firstSample[fieldName][fieldName][:1]
        print("First detection:")
        print(detectionsSubset)
    else:
        print(f"{fieldName}: {firstSample[fieldName]}")
print()

print("Dataset statistics:")
stats = dataset.stats()
fo.pprint(stats)
print()

print("Computing metadata...")
dataset.compute_metadata()
print("First sample metadata:")

firstSampleMetadata = firstSample.get_field("metadata")
fo.pprint(firstSampleMetadata)
print()

print("Available dataset aggregations:")
print(dataset.list_aggregations())

print("Classes in the dataset:")
classes = dataset.distinct("detections.detections.label")
print(classes)
print()

print("Class counts in the dataset:")
labelCounts = dataset.count_values("detections.detections.label")
print(labelCounts)
print()

# Expression that computes the area of a bounding box, in pixels
# Bboxes are in [top-left-x, top-left-y, width, height] format
bbox_width = F("bounding_box")[2] * F("$metadata.width")
bbox_height = F("bounding_box")[3] * F("$metadata.height")
bbox_area = bbox_width * bbox_height

# Expression that computes the area of ground truth bboxes
gt_areas = F("detections.detections[]").apply(bbox_area)

# Compute (min, max, mean) of ground truth bounding boxes
print(dataset.bounds(gt_areas))
print(dataset.mean(gt_areas))

mapa = np.random.randint(256, size=(128, 128), dtype=np.uint8)

result = fo.Heatmap(map=mapa)
#result.save("heatmap.png")
result.export_map("heatmap.png")
# result2 = result.to_array()
# cv2.imwrite("heatmap.png", result2)

widthHistPlot = fo.NumericalHistogram(F("metadata.width"), bins=100, xlabel="Image width")
heightHistPlot = fo.NumericalHistogram(F("metadata.height"), bins=100, xlabel="Image height")
classHistPlot = fo.CategoricalHistogram("detections.detections.label", order="frequency")
bboxAreaHistPlot = fo.NumericalHistogram(gt_areas, bins=100, xlabel="BBox area")
plot = fo.ViewGrid([classHistPlot, widthHistPlot, heightHistPlot, bboxAreaHistPlot], shape=(4, 1), init_view=dataset)
plot.show()

session = fo.launch_app(dataset)
session.plots.attach(plot)

session.wait()
session.close()