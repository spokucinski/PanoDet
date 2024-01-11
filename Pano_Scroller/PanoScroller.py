import cv2
import numpy as np
import os
import argparse
from AnnotationManager import Annotation, getFileAnnotations
from PanoScrollerArgs import PanoScrollerArgs
from ProcessMonitoring import SplitProgressMonitor
from pathlib import Path
import math
import matplotlib.pyplot as plt
from itertools import groupby
from operator import itemgetter
import random
import WindowNames as wns
import WindowController
import AnnotationManager
import DataManager

def loadArgs() -> PanoScrollerArgs:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="manual")
    parser.add_argument("--inputPath", default="data\\required_input\\original_images")
    parser.add_argument("--mainWindowName", default=wns.WINDOW_MAIN)
    parser.add_argument("--previewWindowName", default=wns.WINDOW_PREVIEW)
    parser.add_argument("--imageFormats", nargs='+', default=[".jpg", ".png"])

    print("\nLoading program parameters...")
    cmdArgs = parser.parse_args()
    args: PanoScrollerArgs = PanoScrollerArgs(cmdArgs.mode, cmdArgs.inputPath, cmdArgs.mainWindowName, cmdArgs.previewWindowName, cmdArgs.imageFormats)
    print("Parameters loaded, listing:")
    print(', '.join("%s: %s" % item for item in vars(args).items()))

    return args

def getCosSplits(image: cv2.typing.MatLike, annotations: list[Annotation], splitProcessMonitor: SplitProgressMonitor): 

    imageHeightInPixels = image.shape[0]
    imageWidthInPixels = image.shape[1]

    annotationsMatrix = np.zeros((imageHeightInPixels, imageWidthInPixels), dtype=float)
    for annotation in annotations:
        (x1, y1), (x2, y2) = AnnotationManager.denormalizeAnnotation(annotation, imageHeightInPixels, imageWidthInPixels)
        annotationsMatrix[y1:y2+1, x1:x2+1] += 1
    WindowController.updateAnnotationControlView(annotationsMatrix)
    
    # https://stackoverflow.com/questions/43741885/how-to-convert-spherical-coordinates-to-equirectangular-projection-coordinates

    weightsMatrix = np.zeros((imageHeightInPixels, imageWidthInPixels), dtype=float)
    for h in range(imageHeightInPixels):
        pixelsWeight = math.cos(-(h - imageHeightInPixels / 2.0) / imageHeightInPixels * math.pi)
        weightsMatrix[h, :] = pixelsWeight
    WindowController.updateWeightsControlView(weightsMatrix)

    weightedAnnotations = annotationsMatrix * weightsMatrix
    WindowController.updateWeightedAnnotationsControlView(weightedAnnotations)
    WindowController.updateColoredWeightedAnnotationsControlView(weightedAnnotations)
    
    weightedColumns = np.sum(weightedAnnotations, axis=0)
    WindowController.updateWeightedColumnsPlot(weightedColumns, splitProcessMonitor.controlFigure, splitProcessMonitor.controlAxes, splitProcessMonitor.controlPlottedLine)
    
        # Group by value and find ranges
    groups = []
    for k, g in groupby(enumerate(weightedColumns), lambda x: x[1]):
        group = list(map(itemgetter(0), g))
        range_length = group[-1] - group[0]
        groups.append((k, (group[0], group[-1]), range_length))

    # Sort the groups by value
    groups.sort(key=lambda x: (x[0], -x[2]))

    # Print the value and the simplified range (lower and upper border)
    for value, borders, range_length in groups:
        print(f"Value: {value}, Range: {borders}, Actual Range: {range_length}")

    return groups

def suggestSplitStd(splitProgressMonitor: SplitProgressMonitor):
    return

def scrollImage(image: cv2.typing.MatLike, scrollX: int):
    leftPart = image[:, 0:scrollX]
    rightPart = image[:, scrollX:]

    result = np.concatenate((rightPart, leftPart), axis=1)
    image[:, :] = result[:,:]
    # cv2.imshow("test", result)
    # image.data = result.data
    # return result

def markSplitPoint(processMonitor: SplitProgressMonitor):
    cv2.line(processMonitor.main_img, (processMonitor.last_suggested_c_split, 0), (processMonitor.last_suggested_c_split, processMonitor.main_img.shape[0]), (255, 0, 0), processMonitor.line_thickness)

def copyImageParts(sourceImage: cv2.typing.MatLike, targetImage: cv2.typing.MatLike, widthRange: (int, int), heightRange: (int, int)):
    targetImage[heightRange[0]:heightRange[1], widthRange[0]:widthRange[1]] = sourceImage[heightRange[0]:heightRange[1], widthRange[0]:widthRange[1]]

def addVerticalLine(image: cv2.typing.MatLike, x: int, color: (int, int, int) = (255, 0, 0), lineThickness:int = 3):
    cv2.line(image, (x, 0), (x, image.shape[0]), color, lineThickness)

def removeVerticalLine(image: cv2.typing.MatLike, x: int, sourceImage: cv2.typing.MatLike, lineThickness:int = 3):
    image[:, x-lineThickness:x+lineThickness] = sourceImage[:, x-lineThickness:x+lineThickness]

def addStatusInfo(image: cv2.typing.MatLike, x: int, imageIndex: int, maxImageIndex: int):
    cv2.putText(img=image, text=f"X = {x}, Image: {imageIndex} / {maxImageIndex}", org=(100, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255), thickness=3, lineType=cv2.LINE_AA)

def removeStatusInfo(image: cv2.typing.MatLike, sourceImage: cv2.typing.MatLike):
    image[0:int(0.1*image.shape[0]), 0:int(0.25*image.shape[1])] = sourceImage[0:int(0.1*image.shape[0]), 0:int(0.25*image.shape[1])]
   
def processEvent(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            removeVerticalLine(param.main_img, param.last_known_x, param.original_unchanged_img)
            removeStatusInfo(param.main_img, param.original_unchanged_img)
            
            AnnotationManager.addAnnotationsToImage(param.main_img, param.original_img_annotations) 

            addVerticalLine(param.main_img, x)
            addStatusInfo(param.main_img, x, param.loaded_image_index, param.max_image_index)        
            param.last_known_x = x 

        elif event == cv2.EVENT_LBUTTONDOWN:
            param.preview_img = np.copy(param.original_unchanged_img)
            scrollImage(param.preview_img, x)
            param.scrolled_resulting_img = np.copy(param.preview_img)
            param.last_scroll = x/param.main_img.shape[1]

        elif event == cv2.EVENT_RBUTTONDOWN:
            param.original_img_path = param.imagePaths[param.loaded_image_index]
            updated_file_path = param.original_img_path.replace("required_input", "scrolled_images").replace("original_images\\", "")
            os.makedirs(os.path.dirname(updated_file_path), exist_ok=True)
            cv2.imwrite(updated_file_path, param.scrolled_img)
            
            filename = Path(updated_file_path)
            filename = filename.with_suffix('')
            filename = str(filename) + "_scroll.txt"
            filename = filename.replace("scrolled_images", "scroll_values")
            with open(filename, 'w') as writer:
                writer.write(str(param.last_scroll))

            if param.loaded_image_index >= param.max_image_index:
                param.processing = False
            else:
                param.loaded_image_index = param.loaded_image_index + 1
                param.original_unchanged_img = cv2.imread(param.imagePaths[param.loaded_image_index])
                param.original_img_annotations = getFileAnnotations(param.annotationPaths[param.loaded_image_index])
                param.marked_img = param.original_unchanged_img.copy()
                param.scrolled_img = param.original_unchanged_img.copy()
                param.last_scroll = 0.0
                cv2.imshow(param.previewWindowName, param.scrolled_img)
                param.last_suggested_c_split = 0
                param.calculated_c_ranges = None

def main():
    print("Starting PanoScroller!")

    args: PanoScrollerArgs = loadArgs() 
    imagePaths, annotationPaths = DataManager.getInput(args.inputPath, args.imageFormats)

    original_first_image = cv2.imread(imagePaths[0])
    marked_first_image = original_first_image.copy()
    first_image_annotations = getFileAnnotations(annotationPaths[0])
    AnnotationManager.addAnnotationsToImage(marked_first_image, first_image_annotations)

    if not imagePaths or len(imagePaths) == 0:
        print("No input files detected, ending processing")
    else:
        processParams = SplitProgressMonitor(imagePaths,
                                             annotationPaths,
                                             0, 
                                             len(imagePaths) - 1, 
                                             original_first_image,
                                             first_image_annotations,
                                             marked_first_image, 
                                             cv2.imread(imagePaths[0]),
                                             cv2.imread(imagePaths[0]), 
                                             0, 
                                             3, 
                                             True,
                                             0.0, 
                                             0, 
                                             0,
                                             None,
                                             None,
                                             False,
                                             None,
                                             None,
                                             None)

        WindowController.initializeBaseWindows()
        cv2.setMouseCallback(wns.WINDOW_MAIN, processEvent, processParams)

        while (processParams.processing):
            cv2.imshow(args.mainWindowName, processParams.main_img)
            cv2.imshow(args.previewWindowName, processParams.preview_img)
            
            k = cv2.waitKey(20) & 0xFF
            # C suggests split point with COS(f)
            if k == 99:
                if not processParams.controlWindowsInitialized:
                    processParams.controlWindowsInitialized = True
                    WindowController.initializeControlWindows()
                    figure, axes, plottedLine = WindowController.initializeWeightsPlot()
                    processParams.controlAxes = axes
                    processParams.controlFigure = figure
                    processParams.controlPlottedLine = plottedLine

                if not processParams.calculated_c_ranges:
                    processParams.calculated_c_ranges = getCosSplits(processParams.original_unchanged_img, processParams.original_img_annotations, processParams)

                if processParams.last_suggested_c_split >= len(processParams.calculated_c_ranges):
                    processParams.last_suggested_c_split = 0
                    
                suggestedSplitGroup = processParams.calculated_c_ranges[processParams.last_suggested_c_split]
                suggestedSplitX = random.randint(suggestedSplitGroup[1][0], suggestedSplitGroup[1][1])
                print(suggestedSplitX)
                processParams.last_suggested_c_split += 1
                splitImage(processParams, suggestedSplitX)
                markSplitPoint(processParams)

            if k == 115:
                suggestSplitStd(processParams)

            # ESC escapes
            if k == 27:
                break

    print("Ending PanoScroller, closing the app!")

if __name__ == '__main___':
    main()
else:
    main()