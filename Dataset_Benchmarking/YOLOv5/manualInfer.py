import torch
import sys
import os
from pathlib import Path
from external.detect import run # Import YOLOv5 detect module

# Paths (Modify these according to your setup)
model_path = "/workspaces/PanoDet/Dataset_Benchmarking/YOLOv5/results/SPHERE_CODE55/Train/ALL_1500_1792_2_yolov5x_SGD/weights/best.pt"  # Change to your model file path
input_dir = "/workspaces/PanoDet/Dataset_Benchmarking/YOLOv5/data/ERP"  # Change to your input image directory
output_dir = "/workspaces/PanoDet/Dataset_Benchmarking/YOLOv5/data/ERP/out"  # Change to your desired output directory


# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Run inference
run(
    weights=model_path,
    source=input_dir,
    project=output_dir,
    name="results",
    save_txt=True,  # Save detection results as text files
    save_conf=True,  # Save confidence scores
    save_crop=True,  # Save cropped detections,
    exist_ok=True,  # Avoid overwriting
    imgsz=[1024, 1792]  # Image size
)

print(f"Inference completed. Results saved to: {output_dir}/results")