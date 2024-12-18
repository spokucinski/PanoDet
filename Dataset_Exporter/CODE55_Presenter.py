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
OUTPUT_PATH = 'output'
HEATMAP_SHAPE = (1024, 2048)

GENERATE_HEATMAPS = False
RUN_SESSION = True

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
    "Alarm Sensor": 54,
    "Living Room": 55,
    "Kitchen": 56,
    "Bathroom": 57,
    "Bedroom": 58,
    "Hall": 59,
    "Garage": 60,
    "Kid's Room": 61,
    "Office": 62,
    "Closet Room": 63,
    "Toilet Room": 64
}

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

print("Image width counts in the dataset:")
widthCounts = dataset.count_values("metadata.width")
print(widthCounts)
print()

print("Image height counts in the dataset:")
heightCounts = dataset.count_values("metadata.height")
print(heightCounts)
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

if GENERATE_HEATMAPS:
    heatmaps = np.zeros(shape=(HEATMAP_SHAPE[0], HEATMAP_SHAPE[1], len(LABELS) + 1))
    for sample in dataset:
        for detection in sample.detections.detections:
            xmin = int(detection.bounding_box[0] * HEATMAP_SHAPE[1])
            xmax = int((detection.bounding_box[0] + detection.bounding_box[2]) * HEATMAP_SHAPE[1])

            ymin = int(detection.bounding_box[1] * HEATMAP_SHAPE[0])
            ymax = int((detection.bounding_box[1] + detection.bounding_box[3]) * HEATMAP_SHAPE[0])
                
            heatmaps[ymin:ymax, xmin:xmax, LABELS[detection.label]] += 1
            heatmaps[ymin:ymax, xmin:xmax, len(LABELS)] += 1

    for label in LABELS:
        print(f"Processing heatmap for label: {label}")
        labelHeatmap = heatmaps[:, :, LABELS[label]]
        normalizedLabelHeatmap = np.interp(labelHeatmap, (labelHeatmap.min(), labelHeatmap.max()), (0, 1))
        normalizedLabelHeatmap = np.multiply(normalizedLabelHeatmap, 255)
        cv2.imwrite(os.path.join(OUTPUT_PATH, f"Normalized{label}Heatmap.png"), normalizedLabelHeatmap)
        coloredNormalizedLabelHeatmap = cv2.applyColorMap(np.uint8(normalizedLabelHeatmap), cv2.COLORMAP_INFERNO)
        cv2.imwrite(os.path.join(OUTPUT_PATH, f"Colored{label}Heatmap.png"), coloredNormalizedLabelHeatmap) 

    print("Processing the master heatmap")
    masterHeatmap = heatmaps[:, :, len(LABELS)]
    normalizedMasterHeatmap = np.interp(masterHeatmap, (masterHeatmap.min(), masterHeatmap.max()), (0, 1))
    normalizedMasterHeatmap = np.multiply(normalizedMasterHeatmap, 255)
    cv2.imwrite(os.path.join(OUTPUT_PATH, "NormalizedMasterHeatmap.png"), normalizedMasterHeatmap)
    coloredMasterHeatmap = cv2.applyColorMap(np.uint8(normalizedMasterHeatmap), cv2.COLORMAP_INFERNO)
    cv2.imwrite(os.path.join(OUTPUT_PATH, "ColoredMasterHeatmap.png"), coloredMasterHeatmap)

if RUN_SESSION:
    widthHistPlot = fo.CategoricalHistogram("metadata.width", xlabel="Image width", order="frequency")
    heightHistPlot = fo.CategoricalHistogram("metadata.height", xlabel="Image height", order="frequency")
    classHistPlot = fo.CategoricalHistogram("detections.detections.label", order="frequency")
    bboxAreaHistPlot = fo.NumericalHistogram(gt_areas, bins=100, xlabel="BBox area")
    plot = fo.ViewGrid([classHistPlot, widthHistPlot, heightHistPlot, bboxAreaHistPlot], shape=(4, 1), init_view=dataset)
    plot.show()

    session = fo.launch_app(dataset)
    session.plots.attach(plot)

    session.wait()
    session.close()