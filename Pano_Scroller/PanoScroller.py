import cv2
import numpy as np
import os
import argparse
from Scroller import Annotation, getFileAnnotations
from PanoScrollerArgs import PanoScrollerArgs
from ProcessMonitoring import SplitProgressMonitor
from pathlib import Path
import math
from PIL import Image
import matplotlib.pyplot as plt
from itertools import groupby
from operator import itemgetter
import random

def loadArgs() -> PanoScrollerArgs:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="manual")
    parser.add_argument("--inputPath", default="input\\original_images")
    parser.add_argument("--mainWindowName", default="Source image")
    parser.add_argument("--previewWindowName", default="Preview")
    parser.add_argument("--imageFormats", nargs='+', default=[".jpg", ".png"])

    print("\nLoading program parameters...")
    cmdArgs = parser.parse_args()
    args: PanoScrollerArgs = PanoScrollerArgs(cmdArgs.mode, cmdArgs.inputPath, cmdArgs.mainWindowName, cmdArgs.previewWindowName, cmdArgs.imageFormats)
    print("Parameters loaded, listing:")
    print(', '.join("%s: %s" % item for item in vars(args).items()))

    return args

def loadInput(inputPath: str, imageFormats: [str]) -> (list[str], list[str]):
    print("\nLoading input files...")
    print(f"Searched path: {inputPath}")
    
    imagePaths: list[str] = []
    for root, _, files in os.walk(inputPath):
        for file in files:
            if file.lower().endswith(tuple(imageFormats)):
                imagePaths.append(os.path.join(root, file))
    foundImagesCount: int = len(imagePaths)
    print(f"Found: {foundImagesCount} images!")

    annotationPaths: list[Annotation] = []
    for imagePath in imagePaths:
        annotationPath = imagePath.replace("original_images", "original_annotations")
        for imageFormat in imageFormats:
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

def initializeWindows(processParams: SplitProgressMonitor, mainWinName: str, previewWinName: str):  
    cv2.namedWindow(mainWinName, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(mainWinName, 600, 300)
    cv2.setMouseCallback(mainWinName, split_image, processParams)

    cv2.namedWindow(previewWinName, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(previewWinName, 600, 300)

def showAnnotationControlView(annotationsMatrix: np.ndarray):
    colorMap = {
        0: [0, 0, 0],      # Black
        1: [0, 255, 0],    # Green
        2: [255, 255, 0],  # Yellow
        3: [255, 0, 0],    # Red
        4: [128, 0, 128],  # Purple
        5: [150, 75, 0]    # Brown
    }

    controlImage = np.zeros((annotationsMatrix.shape[0], annotationsMatrix.shape[1], 3), dtype=np.uint8)
    for value, color in colorMap.items():
        controlImage[annotationsMatrix == value] = color

    controlImage = cv2.cvtColor(controlImage, cv2.COLOR_RGB2BGR)
    annotationsControlViewName = 'AnnotationsControlView'
    cv2.namedWindow(annotationsControlViewName, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(annotationsControlViewName, 400, 200)
    cv2.imshow(annotationsControlViewName, controlImage)

def showWeightsControlView(weightsMatrix: np.ndarray):

    weightsMatrixCopy = weightsMatrix.copy()
    weightsMatrixCopy = (weightsMatrixCopy * 255).astype(np.uint8)

    weightsControlViewName = 'WeightsControlView'
    cv2.namedWindow(weightsControlViewName, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(weightsControlViewName, 400, 200)
    cv2.imshow(weightsControlViewName, weightsMatrixCopy)

def showWeightedAnnotationsControlView(weightedAnnotationsSource: np.ndarray):

    weightedAnnotations = weightedAnnotationsSource.copy()
    weightedAnnotations = (weightedAnnotations - np.min(weightedAnnotations)) / (np.max(weightedAnnotations) - np.min(weightedAnnotations))
    weightedAnnotations = (weightedAnnotations * 255).astype(np.uint8)

    viewName = 'WeightedAnnotationsControlView'
    cv2.namedWindow(viewName, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(viewName, 400, 200)
    cv2.imshow(viewName, weightedAnnotations)

def showColoredWeightedAnnotationsControlView(weightedAnnotationsSource: np.ndarray):
    weightedAnnotations = np.zeros((weightedAnnotationsSource.shape[0], weightedAnnotationsSource.shape[1], 3), dtype=np.uint8)
    
    weightsColorMap = {
        0: [255, 0, 0],         # Red
        0.1: [0, 255, 0],       # Green
        0.2: [0, 0, 255],       # Blue
        0.3: [255, 255, 0],     # Yellow
        0.4: [0, 255, 255],     # Cyan
        0.5: [255, 0, 255],     # Magenta
        0.6: [128, 0, 0],       # Dark Red
        0.7: [0, 128, 0],       # Dark Green
        0.8: [0, 0, 128],       # Dark Blue
        0.9: [128, 128, 128]    # Grey
    }

    for threshold, color in weightsColorMap.items():
        weightedAnnotations[weightedAnnotationsSource > threshold] = color 
    weightedAnnotations = cv2.cvtColor(weightedAnnotations, cv2.COLOR_RGB2BGR)
    
    viewName = 'ColoredWeightedAnnotationsControlView'
    cv2.namedWindow(viewName, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(viewName, 400, 200)
    cv2.imshow(viewName, weightedAnnotations)

def showWeightedColumnsPlot(weightedColumns: np.ndarray):
    xAxis = np.arange(len(weightedColumns))
    plt.figure(figsize=(10, 6))  # You can adjust the size of the figure
    plt.plot(xAxis, weightedColumns)
    plt.xlim(min(xAxis), max(xAxis))
    plt.title('Weights plot by column')
    plt.show(block=False)

def getCosSplits(image: cv2.typing.MatLike, annotations: list[Annotation]): 

    imageHeightInPixels = image.shape[0]
    imageWidthInPixels = image.shape[1]

    annotationsMatrix = np.zeros((imageHeightInPixels, imageWidthInPixels), dtype=float)
    for annotation in annotations:
        (x1, y1), (x2, y2) = denormalizeAnnotation(annotation, imageHeightInPixels, imageWidthInPixels)
        annotationsMatrix[y1:y2+1, x1:x2+1] += 1
    showAnnotationControlView(annotationsMatrix)
    
    # https://stackoverflow.com/questions/43741885/how-to-convert-spherical-coordinates-to-equirectangular-projection-coordinates

    weightsMatrix = np.zeros((imageHeightInPixels, imageWidthInPixels), dtype=float)
    for h in range(imageHeightInPixels):
        pixelsWeight = math.cos(-(h - imageHeightInPixels / 2.0) / imageHeightInPixels * math.pi)
        weightsMatrix[h, :] = pixelsWeight
    showWeightsControlView(weightsMatrix)

    weightedAnnotations = annotationsMatrix * weightsMatrix
    showWeightedAnnotationsControlView(weightedAnnotations)
    showColoredWeightedAnnotationsControlView(weightedAnnotations)
    
    weightedColumns = np.sum(weightedAnnotations, axis=0)
    showWeightedColumnsPlot(weightedColumns)
    
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

def split_image(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            param.marked_img[:, param.last_known_x-param.line_thickness:param.last_known_x+param.line_thickness] = param.original_img[:, param.last_known_x-param.line_thickness:param.last_known_x+param.line_thickness]
            addAnnotations(param.marked_img, param.original_img_annotations)
            param.last_known_x = x
            cv2.line(param.marked_img, (x, 0), (x, param.marked_img.shape[0]), (255, 0, 0), param.line_thickness)
            cv2.putText(img=param.marked_img, text=f"X = {x}, Image: {param.loaded_image_index} / {param.max_image_index}", org=(100, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255), thickness=3, lineType=cv2.LINE_AA)

        elif event == cv2.EVENT_LBUTTONDOWN:
            param.marked_img = param.original_img.copy()
            left_part = param.marked_img[0:param.marked_img.shape[0], 0:x]
            right_part = param.marked_img[0:param.marked_img.shape[0], x:param.marked_img.shape[1]]
            param.scrolled_img = np.concatenate((right_part, left_part), axis=1)
            param.last_scroll = x/param.marked_img.shape[1]
            cv2.imshow(param.previewWindowName, param.scrolled_img)

        elif event == cv2.EVENT_RBUTTONDOWN:
            param.original_img_path = param.imagePaths[param.loaded_image_index]
            updated_file_path = param.original_img_path.replace("input", "output")
            os.makedirs(os.path.dirname(updated_file_path), exist_ok=True)
            cv2.imwrite(updated_file_path, param.scrolled_img)
            
            filename = Path(updated_file_path)
            filename = filename.with_suffix('')
            filename = str(filename) + "_scroll.txt"
            with open(filename, 'w') as writer:
                writer.write(str(param.last_scroll))

            if param.loaded_image_index >= param.max_image_index:
                param.processing = False
            else:
                param.loaded_image_index = param.loaded_image_index + 1
                param.original_img = cv2.imread(param.imagePaths[param.loaded_image_index])
                param.original_img_annotations = getFileAnnotations(param.annotationPaths[param.loaded_image_index])
                param.marked_img = param.original_img.copy()
                param.scrolled_img = param.original_img.copy()
                param.last_scroll = 0.0
                cv2.imshow(param.previewWindowName, param.scrolled_img)

def denormalizeAnnotation(annotation: Annotation, imageHeight: int, imageWidth: int) -> ((int, int), (int, int)):
    # Recalculate annotation from normalized
    xCenter = annotation.xCenter * imageWidth
    yCenter = annotation.yCenter * imageHeight
    width = annotation.width * imageWidth
    height = annotation.height * imageHeight

    x1 = int(xCenter - (width/2))
    y1 = int(yCenter - (height/2))

    x2 = int(xCenter + (width/2))
    y2 = int(yCenter + (height/2))

    return (x1, y1), (x2, y2)

def addAnnotations(image: cv2.typing.MatLike, annotations: list[Annotation]):
    image_width, image_height = image.shape[1], image.shape[0]

    for annotation in annotations:
        (x1, y1), (x2, y2) = denormalizeAnnotation(annotation, image_height, image_width)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
   
def main():
    print("Starting PanoScroller!")

    args: PanoScrollerArgs = loadArgs() 
    imagePaths, annotationPaths = loadInput(args.inputPath, args.imageFormats)

    original_first_image = cv2.imread(imagePaths[0])
    marked_first_image = original_first_image.copy()
    first_image_annotations = getFileAnnotations(annotationPaths[0])
    addAnnotations(marked_first_image, first_image_annotations)

    if not imagePaths or len(imagePaths) == 0:
        print("No input files detected, ending processing")
    else:
        processParams = SplitProgressMonitor(imagePaths,
                                             annotationPaths,
                                             args.previewWindowName, 
                                             0, 
                                             len(imagePaths) - 1, 
                                             original_first_image,
                                             first_image_annotations,
                                             marked_first_image, 
                                             cv2.imread(imagePaths[0]), 
                                             0, 
                                             3, 
                                             True,
                                             0.0, 
                                             0, 
                                             0,
                                             None,
                                             None)

        initializeWindows(processParams, args.mainWindowName, args.previewWindowName)

        while (processParams.processing):
            cv2.imshow(args.mainWindowName, processParams.marked_img)
            cv2.imshow(args.previewWindowName, processParams.scrolled_img)
            
            k = cv2.waitKey(20) & 0xFF
            # C suggests split point with COS(f)
            if k == 99:
                processParams.calculated_c_ranges = getCosSplits(processParams.original_img, processParams.original_img_annotations)
                suggestedSplitGroup = processParams.calculated_c_ranges[processParams.last_suggested_c_split]
                suggestedSplitX = random.randint(suggestedSplitGroup[1][0], suggestedSplitGroup[1][1])
                print(suggestedSplitX)
                processParams.last_suggested_c_split += 1

            if k == 115:
                suggestSplitStd(processParams)

            # ESC escapes
            if k == 27:
                break

    print("Ending PanoScroller, closing the app!")

    for value, borders, range_length in groups:
        print(f"Value: {value}, Range: {borders}, Actual Range: {range_length}")

if __name__ == '__main___':
    main()
else:
    main()