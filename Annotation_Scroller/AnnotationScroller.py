import cv2
import numpy as np
import os
import argparse
from pathlib import Path
from Scroller import Annotation
import math

def loadAnnotationFiles(inputPath: str) -> list[str]:
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

def main():
    inputPath: str = "input\original_annotations"
    annotationFiles = loadAnnotationFiles(inputPath=inputPath)
    for annotationFile in annotationFiles:
        filename = Path(annotationFile)
        filename = filename.with_suffix('')
        filename = str(filename) + "_scroll.txt"
        filename = filename.replace("original_annotations", "scrolls")
        scroll = 0.0
        with open(filename, 'r') as scrollReader:
            scroll = float(scrollReader.readline())
        with open(annotationFile, 'r') as reader:
            rawAnnotations = reader.readlines()
            
            parsedAnnotations: [] = []
            for annotation in rawAnnotations:
                type, x, y, width, height = annotation.split(' ')
                parsedAnnotations.append(Annotation(int(type), float(x), float(y), float(width), float(height)))

            leftEdgeObjects = list(filter(lambda annotation: (annotation.xCenter - (annotation.width/2)) == 0, parsedAnnotations))
            rightEdgeObjects = list(filter(lambda annotation: (annotation.xCenter + (annotation.width/2)) == 1, parsedAnnotations))

            scrolledAnnotations = []
            for parsedAnnotation in parsedAnnotations:                
                if parsedAnnotation.xCenter - (parsedAnnotation.width/2) >= scroll:
                    parsedAnnotation.xCenter = parsedAnnotation.xCenter - scroll
                    scrolledAnnotations.append(f"{parsedAnnotation.objectType} {parsedAnnotation.xCenter} {parsedAnnotation.yCenter} {parsedAnnotation.width} {parsedAnnotation.height}\n")
                elif parsedAnnotation.xCenter + (parsedAnnotation.width/2) <= scroll:
                    parsedAnnotation.xCenter = parsedAnnotation.xCenter - scroll + 1
                    scrolledAnnotations.append(f"{parsedAnnotation.objectType} {parsedAnnotation.xCenter} {parsedAnnotation.yCenter} {parsedAnnotation.width} {parsedAnnotation.height}\n")
                else:
                    width1 = (scroll) - (parsedAnnotation.xCenter - (parsedAnnotation.width/2)) 
                    width2 = (parsedAnnotation.xCenter + (parsedAnnotation.width/2)) - (scroll) 
                    
                    x1 = 1 - (width1/2)
                    x2 = (width2/2)

                    scrolledAnnotations.append(f"{parsedAnnotation.objectType} {x1} {parsedAnnotation.yCenter} {width1} {parsedAnnotation.height}\n")
                    scrolledAnnotations.append(f"{parsedAnnotation.objectType} {x2} {parsedAnnotation.yCenter} {width2} {parsedAnnotation.height}\n")
            
            scrolledAnnotationsObjects = []
            for scrolledAnnotation in scrolledAnnotations:
                params = scrolledAnnotation.split(' ')
                scrolledAnnotationsObjects.append(Annotation(int(params[0]), float(params[1]), float(params[2]), float(params[3]), float(params[4])))

            for scrolledAnnotationObject in scrolledAnnotationsObjects:
                adjucentAnnotations = list(filter(lambda scrolledAnnotation: math.isclose((scrolledAnnotation.xCenter + (scrolledAnnotation.width/2)), (scrolledAnnotationObject.xCenter - (scrolledAnnotationObject.width/2)), rel_tol=0.0005), scrolledAnnotationsObjects))
                adjucentAnnotationsOfSameType = list(filter(lambda adjucentAnnotation: adjucentAnnotation.objectType == scrolledAnnotationObject.objectType, adjucentAnnotations))
                if len(adjucentAnnotationsOfSameType) > 0:
                    leftMin = min((scrolledAnnotationObject.xCenter - (scrolledAnnotationObject.width/2)), min((annotation.xCenter - (annotation.width/2)) for annotation in adjucentAnnotationsOfSameType))
                    rightMax = max((scrolledAnnotationObject.xCenter + (scrolledAnnotationObject.width/2)), max((annotation.xCenter + (annotation.width/2)) for annotation in adjucentAnnotationsOfSameType))
                    width = rightMax - leftMin
                    newXCenter = leftMin + (width/2)
                    scrolledAnnotations.append(f"{parsedAnnotation.objectType} {newXCenter} {parsedAnnotation.yCenter} {width} {parsedAnnotation.height}\n")

            resultsAnnotationFile = annotationFile.replace("input", "output")
            os.makedirs(os.path.dirname(resultsAnnotationFile), exist_ok=True)
            with open(resultsAnnotationFile, "w") as writer:
                writer.writelines(scrolledAnnotations)            

if __name__ == '__main___':
    main()
else:
    main()