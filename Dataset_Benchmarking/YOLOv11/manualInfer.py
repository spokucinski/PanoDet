import os
from external.ultralytics import YOLO

# Load a model
model = YOLO("yolo11l.pt")  # pretrained YOLO11n model

# Define paths
input_folder = "data/ERP"
output_folder = os.path.join(input_folder, "results")

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get all image files in the input folder
image_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Run batched inference on the list of images
results = model(image_files)

# Process results list
for i, result in enumerate(results):
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen

    # Save the result to the output folder
    output_filename = os.path.join(output_folder, f"result_{i}.jpg")
    result.save(filename=output_filename)
