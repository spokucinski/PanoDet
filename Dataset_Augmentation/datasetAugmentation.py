import random
import cv2
import albumentations as A
import os
from matplotlib import pyplot as plt

class Annotation():

    def __init__(self,
                 objectType: int,
                 xCenter: float,
                 yCenter: float,
                 width: float,
                 height: float):
        
        self.objectType = objectType
        self.xCenter = xCenter
        self.yCenter = yCenter
        self.width = width
        self.height = height

# Define the augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),  # Horizontal flip with 50% probability
    A.RandomBrightness(0.9, True, 1)
 ], bbox_params=A.BboxParams(format='yolo'))

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

        annotations: list[Annotation] = []
        with open(imagePath.replace("images", "labels").replace(".jpg", ".txt").replace(".png", ".txt"), 'r') as reader:                 
            for annotation in reader.readlines():
                objectType, xCenter, yCenter, width, height = annotation.split(' ')
                annotations.append(Annotation(int(objectType), float(xCenter), float(yCenter), float(width), float(height)))

        bboxes = []
        for annotation in annotations:
            bboxes.append([annotation.xCenter, annotation.yCenter, annotation.width, annotation.height, str(annotation.objectType)])

        augmentedImage = transform(image=image, bboxes=bboxes)
        resultsPath = imagePath.replace('datasets', 'augmentedDatasets')
        os.makedirs(os.path.dirname(resultsPath), exist_ok=True)
        augmentedAnnotations: list[Annotation] = []
        for bbox in augmentedImage['bboxes']:
            augmentedAnnotations.append(Annotation(int(bbox[4]), bbox[0], bbox[1], bbox[2], bbox[3]))
        
        cv2.imwrite(resultsPath, augmentedImage['image'])
        resultsPath = resultsPath.replace("images", "labels").replace(".jpg", ".txt").replace(".png", ".txt")
        os.makedirs(os.path.dirname(resultsPath), exist_ok=True)
        with open(resultsPath, "w") as writer:
            writer.writelines(
                list(
                    map(
                        lambda postProcessedAnnotation: 
                            f"{postProcessedAnnotation.objectType} {postProcessedAnnotation.xCenter} {postProcessedAnnotation.yCenter} {postProcessedAnnotation.width} {postProcessedAnnotation.height}\n", 
                            augmentedAnnotations)))