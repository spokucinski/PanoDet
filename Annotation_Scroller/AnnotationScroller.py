import cv2
import numpy as np
import os
import argparse
from pathlib import Path
from Scroller import Annotation
import math

def getAnnotationFilePaths(inputPath: str) -> list[str]:
    
    print("\nLoading input files...")
    print(f"Searched path: {inputPath}")
    
    annotationFiles = []
    for root, _, files in os.walk(inputPath):
        for file in files:
            if file.lower().endswith(".txt"):
                annotationFiles.append(os.path.join(root, file))

    foundAnnotationsCount: int = len(annotationFiles)
    print(f"Found: {foundAnnotationsCount} annotation files!")

    if foundAnnotationsCount == 0:
        print("No annotation files found!")
        return
    elif foundAnnotationsCount < 100:
        print("Listing:")
        print('\n '.join("%s" % annotationFile for annotationFile in annotationFiles))
    else:
        print("Found over 100 images, skipping listing.")

    return annotationFiles

def getScrollValue(annotationFilePath: str) -> float:
        filename = Path(annotationFilePath).with_suffix('')
        filename = str(filename) + "_scroll.txt"
        filename = filename.replace("original_annotations", "scrolls")
        
        with open(filename, 'r') as scrollReader:
            scroll = float(scrollReader.readline())
            return scroll

def getFileAnnotations(annotationsFilePath: str) -> list[Annotation]:
    with open(annotationsFilePath, 'r') as reader:      
        annotations: list[Annotation] = []      
        for annotation in reader.readlines():
            objectType, xCenter, yCenter, width, height = annotation.split(' ')
            annotations.append(Annotation(int(objectType), float(xCenter), float(yCenter), float(width), float(height)))
        
        return annotations 

def scrollAnnotations(originalAnnotations: list[Annotation], scroll: float) -> list[Annotation]: 
    scrolledAnnotations: list[Annotation] = []      
    for originalAnnotation in originalAnnotations:

        # Object placed inside picture even after scrolling                
        if originalAnnotation.xCenter - (originalAnnotation.width/2) >= scroll:
            originalAnnotation.xCenter = originalAnnotation.xCenter - scroll
            scrolledAnnotations.append(Annotation(originalAnnotation.objectType, originalAnnotation.xCenter, originalAnnotation.yCenter, originalAnnotation.width, originalAnnotation.height))
        
        # Object fully wrapped and moved to the right part
        elif originalAnnotation.xCenter + (originalAnnotation.width/2) <= scroll:
            originalAnnotation.xCenter = originalAnnotation.xCenter - scroll + 1
            scrolledAnnotations.append(Annotation(originalAnnotation.objectType, originalAnnotation.xCenter, originalAnnotation.yCenter, originalAnnotation.width, originalAnnotation.height))
        
        # Object cut in half
        else:

            # Left part
            leftWidth = (scroll) - (originalAnnotation.xCenter - (originalAnnotation.width/2)) 
            leftX = 1 - (leftWidth/2)
            scrolledAnnotations.append(Annotation(originalAnnotation.objectType, leftX, originalAnnotation.yCenter, leftWidth, originalAnnotation.height))

            # Right part
            rightWidth = (originalAnnotation.xCenter + (originalAnnotation.width/2)) - (scroll) 
            rightX = (rightWidth/2)
            scrolledAnnotations.append(Annotation(originalAnnotation.objectType, rightX, originalAnnotation.yCenter, rightWidth, originalAnnotation.height))
    
    return scrolledAnnotations

def mergeAdjacentObjects(scrolledAnnotations: list[Annotation], scroll: float) -> list[Annotation]:
    
    #leftEdgeObjects = list(filter(lambda annotation: math.isclose((annotation.xCenter - (annotation.width/2)), 0, rel_tol=0.005), originalAnnotations))
    #rightEdgeObjects = list(filter(lambda annotation: (annotation.xCenter + (annotation.width/2)) == 1, originalAnnotations))
    
    for scrolledAnnotation in scrolledAnnotations:
        adjacentAnnotations = list(
            filter(
                lambda analyzedAnnotation: 
                math.isclose((scrolledAnnotation.xCenter + (scrolledAnnotation.width/2)), (analyzedAnnotation.xCenter - (analyzedAnnotation.width/2)), rel_tol=0.0005), 
                scrolledAnnotations))
    
        adjacentAnnotationsOfSameType = list(
            filter(
                lambda adjacentAnnotation: 
                adjacentAnnotation.objectType == scrolledAnnotation.objectType, 
                adjacentAnnotations))
        
        if len(adjacentAnnotationsOfSameType) > 0:
            
            adjacentAnnotationOfSameTypeOriginallyOnTheEdge = []
            for annotation in adjacentAnnotationsOfSameType:
                leftEdge: float = annotation.xCenter - (annotation.width/2)
                if (math.isclose(leftEdge, 1 - scroll, rel_tol=0.005)):
                    adjacentAnnotationOfSameTypeOriginallyOnTheEdge.append(annotation)
        
            if len(adjacentAnnotationOfSameTypeOriginallyOnTheEdge) > 0:
                
                leftMin = min((scrolledAnnotation.xCenter - (scrolledAnnotation.width/2)), min((annotation.xCenter - (annotation.width/2)) for annotation in adjacentAnnotationsOfSameType))
                rightMax = max((scrolledAnnotation.xCenter + (scrolledAnnotation.width/2)), max((annotation.xCenter + (annotation.width/2)) for annotation in adjacentAnnotationsOfSameType))
                
                # Watch-out for different xy axis direction!
                topMin = min((scrolledAnnotation.yCenter - (scrolledAnnotation.height/2)), min((annotation.yCenter - (annotation.height/2)) for annotation in adjacentAnnotationsOfSameType))
                bottomMax = max((scrolledAnnotation.yCenter + (scrolledAnnotation.height/2)), max((annotation.yCenter + (annotation.height/2)) for annotation in adjacentAnnotationsOfSameType))

                newWidth = rightMax - leftMin
                newHeight = bottomMax - topMin

                newXCenter = leftMin + (newWidth/2)
                newYCenter = topMin + (newHeight/2)

                scrolledAnnotations.append(Annotation(scrolledAnnotation.objectType, newXCenter, newYCenter, newWidth, newHeight))
                scrolledAnnotations.remove(scrolledAnnotation)
                result = list(filter(lambda i: i not in adjacentAnnotationOfSameTypeOriginallyOnTheEdge, scrolledAnnotations))
                return result
            
    return scrolledAnnotations

def main():
    inputPath: str = "input\original_annotations"
    
    annotationFilePaths = getAnnotationFilePaths(inputPath=inputPath)
    
    for annotationFilePath in annotationFilePaths:
        scroll = getScrollValue(annotationFilePath)
        originalAnnotations = getFileAnnotations(annotationFilePath)
        scrolledAnnotations = scrollAnnotations(originalAnnotations, scroll)
        postProcessedAnnotations = mergeAdjacentObjects(scrolledAnnotations, scroll)

        resultsAnnotationFile = annotationFilePath.replace("input", "output")
        os.makedirs(os.path.dirname(resultsAnnotationFile), exist_ok=True)
        with open(resultsAnnotationFile, "w") as writer:
            writer.writelines(
                list(
                    map(
                        lambda postProcessedAnnotation: 
                            f"{postProcessedAnnotation.objectType} {postProcessedAnnotation.xCenter} {postProcessedAnnotation.yCenter} {postProcessedAnnotation.width} {postProcessedAnnotation.height}\n", 
                            postProcessedAnnotations)))            

if __name__ == '__main___':
    main()
else:
    main()