import cv2
import numpy as np
import os
import argparse
from pathlib import Path

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
            annotations = reader.readlines()

            resultsAnnotationFile = annotationFile.replace("input", "output")
            os.makedirs(os.path.dirname(resultsAnnotationFile), exist_ok=True)
            with open(resultsAnnotationFile, "w") as writer:
                for annotation in annotations:
                    type, x, y, width, height = annotation.split(' ')

                    parsedX = float(x)
                    parsedWidth = float(width)

                    if parsedX - (parsedWidth/2) >= scroll:
                        parsedX = parsedX - scroll
                        writer.writelines(f"{type} {parsedX} {y} {parsedWidth} {height}")
                    elif parsedX + (parsedWidth/2) <= scroll:
                        parsedX = parsedX - scroll + 1
                        writer.writelines(f"{type} {parsedX} {y} {parsedWidth} {height}")
                    else:
                        width1 = (scroll) - (parsedX - (parsedWidth/2)) 
                        width2 = (parsedX + (parsedWidth/2)) - (scroll) 
                        
                        x1 = 1 - (width1/2) #(parsedX - (parsedWidth/2)) + (width1/2) - scroll
                        x2 = (width2/2)

                        writer.writelines(f"{type} {x1} {y} {width1} {height}\n")
                        writer.writelines(f"{type} {x2} {y} {width2} {height}")

if __name__ == '__main___':
    main()
else:
    main()