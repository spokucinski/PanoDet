import os
import cv2
import py360convert
from pathlib import Path

# Define paths
data_root = Path('data')
output_root = Path('output')
output_root.mkdir(exist_ok=True)

# Supported image extensions
image_extensions = ['.jpg', '.webp']

# Iterate through each dataset
for dataset_folder in sorted(data_root.iterdir()):
    if dataset_folder.is_dir():
        for subset_folder in sorted(dataset_folder.iterdir()):
            if subset_folder.is_dir():
                print(f"\n📂 Processing: {dataset_folder.name}/{subset_folder.name}")
                
                # Process each supported image
                for ext in image_extensions:
                    for img_file in sorted(subset_folder.glob(f"*{ext}")):
                        print(f"  🖼️ Image: {img_file.name}")
                        img = cv2.imread(str(img_file))
                        if img is None:
                            print(f"    ❌ Failed to load image: {img_file.name}")
                            continue

                        # Convert ERP to cubemap
                        cube_faces = py360convert.e2c(img, face_w=512, cube_format='list')

                        # Define per-image output folder
                        img_output_folder = output_root / dataset_folder.name / subset_folder.name / img_file.stem
                        img_output_folder.mkdir(parents=True, exist_ok=True)

                        # Save all 6 cube faces
                        face_names = ['front', 'right', 'back', 'left', 'up', 'down']
                        for i, face in enumerate(cube_faces):
                            face_path = img_output_folder / f"{face_names[i]}.jpg"
                            cv2.imwrite(str(face_path), face)
