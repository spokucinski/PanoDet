import random
import cv2
import albumentations as A
import os
from matplotlib import pyplot as plt


# Define your dataset directory and image list
datasetDir = 'datasets'
datasetDirContent = os.listdir(datasetDir)

# Filter the list to include only directories (folders)
datasets = [os.path.join(datasetDir, item) for item in datasetDirContent if os.path.isdir(os.path.join(datasetDir, item))]

for dataset in datasets:

    print(f"\nLoading input image files, Path: {dataset}")
    imagePaths: list[str] = []
    annotationPaths: list[str] = []
    for root, datasetDirectories, files in os.walk(os.path.join(dataset)):
        for datasetDirectory in datasetDirectories:
            
            if datasetDirectory.lower() == "images":
                directoryPath = os.path.join(dataset, datasetDirectory)
                imagesDirectory = os.listdir(directoryPath)[0]
                imagesPath = os.path.join(directoryPath, imagesDirectory)
                images = os.listdir(imagesPath)
                
                for image in images:
                    if image.lower().endswith(tuple([".jpg", ".png"])):
                        imagePaths.append(os.path.join(dataset, datasetDirectory, image))
            
            if datasetDirectory.lower() == "labels":
                directoryPath = os.path.join(dataset, datasetDirectory)
                annotationsDirectory = os.listdir(directoryPath)[0]
                annotationsPath = os.path.join(directoryPath, annotationsDirectory)
                annotationFile = os.listdir(annotationsPath)
                
                for annotationFile in annotationFile:
                    if annotationFile.lower().endswith(".txt"):
                        annotationPaths.append(os.path.join(dataset, datasetDirectory, annotationFile))
        
    foundImagesCount: int = len(imagePaths)
    foundAnnotationFilesCounts: int = len(annotationPaths)
    print(f"Found: {foundImagesCount} images and {foundAnnotationFilesCounts} annotation files!")

    





# Define the augmentation pipeline
transform = A.Compose([
    A.Resize(416, 416),  # Resize your images to the desired size
    A.HorizontalFlip(p=0.5),  # Horizontal flip with 50% probability
    # Add more augmentations as needed
    ToTensorV2(),  # Convert the image and annotations to PyTorch tensors
], bbox_params=A.BboxParams(format='yolo'))


for image_file in image_list:
    # Load the image
    image_path = os.path.join(datasetDir, 'images', 'AutomaticallyMinScrolledDataset', image_file)
    image = cv2.imread(image_path)
    
    # Load the YOLO annotation file
    annotation_path = os.path.join(datasetDir, 'labels', 'AutomaticallyMinScrolledDataset', image_file.replace('.jpg', '.txt'))
    
    # Read the content of the annotation file
    with open(annotation_path, 'r') as f:
        lines = f.readlines()
    
    bboxes = []
    class_labels = []
    
    for line in lines:
        values = line.strip().split()
        class_id = int(values[0])
        x_center, y_center, width, height = map(float, values[1:])
        x_min = (x_center - width / 2)
        y_min = (y_center - height / 2)
        x_max = (x_center + width / 2)
        y_max = (y_center + height / 2)
        bboxes.append([x_min, y_min, x_max, y_max])
        class_labels.append(class_id)
    
    # Apply augmentations
    augmented = transform(image=image, bboxes=bboxes)
    
    augmented = transform(image=image, bboxes=bboxes)
    augmented_image = augmented['image']
    augmented_bboxes = augmented['bboxes']