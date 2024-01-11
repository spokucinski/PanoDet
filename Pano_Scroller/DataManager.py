import os
from AnnotationManager import Annotation

def getInput(inputPath: str, acceptedImageFormats: [str]) -> (list[str], list[str]):
    print("\nLoading input files...")
    print(f"Searched path: {inputPath}")
    
    imagePaths: list[str] = []
    for root, _, files in os.walk(inputPath):
        for file in files:
            if file.lower().endswith(tuple(acceptedImageFormats)):
                imagePaths.append(os.path.join(root, file))
    foundImagesCount: int = len(imagePaths)
    print(f"Found: {foundImagesCount} images!")

    annotationPaths: list[Annotation] = []
    for imagePath in imagePaths:
        annotationPath = imagePath.replace("original_images", "original_annotations")
        for imageFormat in acceptedImageFormats:
            annotationPath = annotationPath.replace(imageFormat, ".txt")
        if os.path.exists(annotationPath):
            annotationPaths.append(annotationPath)
    foundAnnotationsCount = len(annotationPaths)
    print(f"Found: {foundAnnotationsCount} according annotation files!")

    if foundImagesCount == 0:
        print("No images found!")
        return
    elif foundImagesCount < 100:
        print("Listing:")
        print('\n '.join("%s" % image for image in imagePaths))
    else:
        print("Found over 100 images, skipping listing.")

    return imagePaths, annotationPaths