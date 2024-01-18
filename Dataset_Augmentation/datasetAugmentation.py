import random
import cv2
import albumentations as A
import os
from matplotlib import pyplot as plt

# Define the augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),  # Horizontal flip with 50% probability
    A.RandomBrightness(0.9, True, 1)
 ])
# ), bbox_params=A.BboxParams(format='yolo'))

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
                        imagePaths.append(os.path.join(dataset, datasetDirectory, imagesDirectory, image))
            
            if datasetDirectory.lower() == "labels":
                directoryPath = os.path.join(dataset, datasetDirectory)
                annotationsDirectory = os.listdir(directoryPath)[0]
                annotationsPath = os.path.join(directoryPath, annotationsDirectory)
                annotationFile = os.listdir(annotationsPath)
                
                for annotationFile in annotationFile:
                    if annotationFile.lower().endswith(".txt"):
                        annotationPaths.append(os.path.join(dataset, datasetDirectory, annotationsDirectory, annotationFile))
        
    foundImagesCount: int = len(imagePaths)
    foundAnnotationFilesCounts: int = len(annotationPaths)
    print(f"Found: {foundImagesCount} images and {foundAnnotationFilesCounts} annotation files!")

    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        augmentedImage = transform(image=image)
        resultsPath = imagePath.replace('datasets', 'augmentedDatasets')
        os.makedirs(os.path.dirname(resultsPath), exist_ok=True)
        cv2.imwrite(resultsPath, augmentedImage['image'])