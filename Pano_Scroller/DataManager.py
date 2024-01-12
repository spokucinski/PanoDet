import os
from AnnotationManager import Annotation
import Consts

def getInput() -> (list[str], list[str]):
    print(f"\nLoading input image files, Path: {Consts.IMAGES_PATH}")
    imagePaths: list[str] = []
    for root, _, files in os.walk(Consts.IMAGES_PATH):
        for file in files:
            if file.lower().endswith(tuple(Consts.ACCEPTED_IMAGE_FORMATS)):
                imagePaths.append(os.path.join(root, file))
    foundImagesCount: int = len(imagePaths)
    print(f"Found: {foundImagesCount} images!")

    print(f"\nLoading input annotation files, Path: {Consts.ANNOTATIONS_PATH}")
    annotationPaths: list[Annotation] = []
    for imagePath in imagePaths:
        annotationPath = imagePath.replace(Consts.IMAGES_PATH, Consts.ANNOTATIONS_PATH)
        for imageFormat in Consts.ACCEPTED_IMAGE_FORMATS:
            annotationPath = annotationPath.replace(imageFormat, ".txt")
        if os.path.exists(annotationPath):
            annotationPaths.append(annotationPath)
    foundAnnotationsCount = len(annotationPaths)
    print(f"Found: {foundAnnotationsCount} according annotation files!")

    return imagePaths, annotationPaths