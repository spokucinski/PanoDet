import random
import cv2
import os
import numpy as np
import AnnotationManager
import WindowManager
import math
import ImageManager
import Consts
from operator import itemgetter
from itertools import groupby

class ScrollingProcess:

    def __init__(self,
                 imagePaths: list[str],
                 annotationPaths: list[str]):
        
        self.imagePaths = imagePaths
        self.annotationPaths = annotationPaths
        self.loaded_image_index = -1 # -1 -> no image loaded
        self.max_image_index = len(imagePaths) - 1
        self.last_known_x = 0
        self.line_thickness = 3
        self.processing = True
        self.last_scroll = 0.0
        self.controlWindowsInitialized = False
        self.original_img_annotations = None
        self.scrolledAnnotations = None

        self.controlFigure = None
        self.controlAxes = None
        self.controlPlottedLine = None

        self.filteredFigure = None
        self.filteredAxes = None
        self.filteredPlottedLine = None

        self.cosFigure = None
        self.cosAxes = None
        self.cosPlottedLine = None

        self.last_suggested_c_split = 0
        self.last_suggested_maximum_split = 0
        self.last_suggested_unified_split = 0
        
        self.last_suggested_c_split_x = 0
        self.last_suggested_maximum_split_x = 0
        self.last_suggested_unified_split_x = 0

        self.calculated_c_ranges = None
        self.calculated_maximum_ranges = None
        self.calculated_unified_ranges = None

    def clearParameters(self):
        self.last_scroll = 0.0
        self.last_known_x = 0

        self.last_suggested_c_split = 0
        self.last_suggested_maximum_split = 0
        self.last_suggested_unified_split = 0

        self.calculated_c_ranges = None
        self.calculated_maximum_ranges = None
        self.calculated_unified_ranges = None

    def loadNextImage(self):
        self.loaded_image_index += 1           
        self.original_unchanged_img = cv2.imread(self.imagePaths[self.loaded_image_index])
        self.main_img = np.copy(self.original_unchanged_img)
        self.preview_img = np.copy(self.original_unchanged_img)
        self.scrolled_resulting_img = np.copy(self.original_unchanged_img)
        self.clearParameters()
        self.original_img_annotations = AnnotationManager.getFileAnnotations(self.annotationPaths[self.loaded_image_index])
        self.scrolledAnnotations = self.original_img_annotations.copy()

        AnnotationManager.addAnnotationsToImage(self.main_img, self.original_img_annotations)        
        ImageManager.addVerticalLine(self.main_img, self.last_known_x)
        ImageManager.addStatusInfo(self.main_img, self.last_known_x, self.loaded_image_index, self.max_image_index)

    def initializeMainFlow(self, eventProcessingFunction):
        WindowManager.initializeBaseWindows(eventProcessingFunction, self)

    def initializeControlFlow(self):
        self.controlWindowsInitialized = True
        WindowManager.initializeControlWindows()
        
        figure, axes, plottedLine = WindowManager.initializePlot()    
        self.controlAxes = axes
        self.controlFigure = figure
        self.controlPlottedLine = plottedLine

        filteredFigure, filteredAxes, filteredPlottedLine = WindowManager.initializePlot()
        self.filteredAxes = filteredAxes
        self.filteredFigure = filteredFigure
        self.filteredPlottedLine = filteredPlottedLine

        cosFigure, cosAxes, cosPlottedLine = WindowManager.initializePlot()
        self.cosFigure = cosFigure
        self.cosAxes = cosAxes
        self.cosPlottedLine = cosPlottedLine

    def getCosSplits(self, image: cv2.typing.MatLike, annotations: list[AnnotationManager.Annotation]): 
        imageHeightInPixels = image.shape[0]
        imageWidthInPixels = image.shape[1]

        annotationsMatrix = np.zeros((imageHeightInPixels, imageWidthInPixels), dtype=float)
        for annotation in annotations:
            (x1, y1), (x2, y2) = AnnotationManager.denormalizeAnnotation(annotation, imageHeightInPixels, imageWidthInPixels)
            annotationsMatrix[y1:y2+1, x1:x2+1] += 1
        WindowManager.updateAnnotationControlView(annotationsMatrix)
        
        # https://stackoverflow.com/questions/43741885/how-to-convert-spherical-coordinates-to-equirectangular-projection-coordinates

        weightsMatrix = np.zeros((imageHeightInPixels, imageWidthInPixels), dtype=float)
        for h in range(imageHeightInPixels):
            pixelsWeight = math.cos(-(h - imageHeightInPixels / 2.0) / imageHeightInPixels * math.pi)
            weightsMatrix[h, :] = pixelsWeight
        WindowManager.updateWeightsControlView(weightsMatrix)

        weightedAnnotations = annotationsMatrix * weightsMatrix
        WindowManager.updateWeightedAnnotationsControlView(weightedAnnotations)
        WindowManager.updateColoredWeightedAnnotationsControlView(weightedAnnotations)
        
        weightedColumns = np.sum(weightedAnnotations, axis=0)
        WindowManager.updatePlot(weightedColumns, self.controlFigure, self.controlAxes, self.controlPlottedLine)
        
            # Group by value and find ranges
        groups = []
        for k, g in groupby(enumerate(weightedColumns), lambda x: x[1]):
            group = list(map(itemgetter(0), g))
            range_length = group[-1] - group[0]
            groups.append((k, (group[0], group[-1]), range_length))

        # Sort the groups by value
        groups.sort(key=lambda x: (x[0], -x[2]))

        # # Print the value and the simplified range (lower and upper border)
        # for value, borders, range_length in groups:
        #     print(f"Value: {value}, Range: {borders}, Actual Range: {range_length}")

        return groups
    
    def getUnifiedSplits(self, image: cv2.typing.MatLike, annotations: list[AnnotationManager.Annotation]):
        imageHeightInPixels = image.shape[0]
        imageWidthInPixels = image.shape[1]

        annotationsMatrix = np.zeros((imageHeightInPixels, imageWidthInPixels), dtype=float)
        for annotation in annotations:
            (x1, y1), (x2, y2) = AnnotationManager.denormalizeAnnotation(annotation, imageHeightInPixels, imageWidthInPixels)
            annotationsMatrix[y1:y2+1, x1:x2+1] += 1
        WindowManager.updateAnnotationControlView(annotationsMatrix)
        
        # https://stackoverflow.com/questions/43741885/how-to-convert-spherical-coordinates-to-equirectangular-projection-coordinates

        weightsMatrix = np.ones((imageHeightInPixels, imageWidthInPixels), dtype=float)
        WindowManager.updateWeightsControlView(weightsMatrix)

        weightedAnnotations = annotationsMatrix * weightsMatrix
        WindowManager.updateWeightedAnnotationsControlView(weightedAnnotations)
        WindowManager.updateColoredWeightedAnnotationsControlView(weightedAnnotations)
        
        weightedColumns = np.sum(weightedAnnotations, axis=0)
        WindowManager.updatePlot(weightedColumns, self.controlFigure, self.controlAxes, self.controlPlottedLine)
        
        # Group by value and find ranges
        groups = []
        for k, g in groupby(enumerate(weightedColumns), lambda x: x[1]):
            group = list(map(itemgetter(0), g))
            range_length = group[-1] - group[0]
            groups.append((k, (group[0], group[-1]), range_length))

        # Sort the groups by value
        groups.sort(key=lambda x: (x[0], -x[2]))

        # # Print the value and the simplified range (lower and upper border)
        # for value, borders, range_length in groups:
        #     print(f"Value: {value}, Range: {borders}, Actual Range: {range_length}")

        return groups
    
    def getMaximumSplits(self, image: cv2.typing.MatLike, annotations: list[AnnotationManager.Annotation]):
        imageHeightInPixels = image.shape[0]
        imageWidthInPixels = image.shape[1]

        annotationsMatrix = np.zeros((imageHeightInPixels, imageWidthInPixels), dtype=float)
        for annotation in annotations:
            (x1, y1), (x2, y2) = AnnotationManager.denormalizeAnnotation(annotation, imageHeightInPixels, imageWidthInPixels)
            annotationsMatrix[y1:y2+1, x1:x2+1] += 1
        WindowManager.updateAnnotationControlView(annotationsMatrix)
        
        # https://stackoverflow.com/questions/43741885/how-to-convert-spherical-coordinates-to-equirectangular-projection-coordinates

        weightsMatrix = np.ones((imageHeightInPixels, imageWidthInPixels), dtype=float)
        for h in range(imageHeightInPixels):
            pixelsWeight = math.cos(-(h - imageHeightInPixels / 2.0) / imageHeightInPixels * math.pi)
            weightsMatrix[h, :] = pixelsWeight
        WindowManager.updateWeightsControlView(weightsMatrix)

        weightedAnnotations = annotationsMatrix * weightsMatrix
        WindowManager.updateWeightedAnnotationsControlView(weightedAnnotations)
        WindowManager.updateColoredWeightedAnnotationsControlView(weightedAnnotations)
        
        onesWeightedColumns = np.max(annotationsMatrix, axis=0)
        WindowManager.updatePlot(onesWeightedColumns, self.controlFigure, self.controlAxes, self.controlPlottedLine)
        
        cosWeightedColumns = np.sum(weightedAnnotations, axis=0)
        WindowManager.updatePlot(cosWeightedColumns, self.cosFigure, self.cosAxes, self.cosPlottedLine)

        # Group by value and find ranges
        groups = []
        for k, g in groupby(enumerate(onesWeightedColumns), lambda x: x[1]):
            group = list(map(itemgetter(0), g))
            range_length = group[-1] - group[0]
            groups.append((k, (group[0], group[-1]), range_length))

        filteredPlot = [0] * imageWidthInPixels
        filteredGroups = [group for group in groups if group[2] >= 0.001 * imageWidthInPixels]
        for filteredGroup in filteredGroups:
            startIndex = filteredGroup[1][0]
            endIndex = filteredGroup[1][1]
            rangeValue = filteredGroup[0]
            rangeLength = filteredGroup[2]
            rangeValues = [rangeValue] * rangeLength
            filteredPlot[startIndex:endIndex] = rangeValues

        WindowManager.updatePlot(filteredPlot, self.filteredFigure, self.filteredAxes, self.filteredPlottedLine)
        
        # Sort the groups by value
        filteredGroups.sort(key=lambda x: -x[0])

        extendedGroups = []
        for filteredGroup in filteredGroups:
            startIndex = filteredGroup[1][0]
            endIndex = filteredGroup[1][1]
            
            partOfWeightedColumns = cosWeightedColumns[startIndex:endIndex]
            partOfWeightedColumnsGroups = []
            for k, g in groupby(enumerate(partOfWeightedColumns), lambda x: x[1]):
                partOfWeightedColumnsGroup = list(map(itemgetter(0), g))
                partOfWeightedColumnsGroupRangeLength = partOfWeightedColumnsGroup[-1] - partOfWeightedColumnsGroup[0]
                partOfWeightedColumnsGroups.append((k, (partOfWeightedColumnsGroup[0], partOfWeightedColumnsGroup[-1]), partOfWeightedColumnsGroupRangeLength))
            partOfWeightedColumnsGroups.sort(key=lambda x: -x[0])

            maxPartOfWeightedColumnsGroup = partOfWeightedColumnsGroups[0]
            maxCosSum = maxPartOfWeightedColumnsGroup[0]
            relativeMaxCosSumX = (maxPartOfWeightedColumnsGroup[1][0] + maxPartOfWeightedColumnsGroup[1][1]) // 2
            absolutMaxCosSumX = relativeMaxCosSumX + filteredGroup[1][0]
            extendedGroups.append((filteredGroup[0], filteredGroup[1], filteredGroup[2], maxCosSum, absolutMaxCosSumX))

        extendedGroups.sort(key=lambda x: (-x[0], -x[3]))
        
        # Print the value and the simplified range (lower and upper border)
        for value, borders, range_length, maxCosValue, maxCosValueX in extendedGroups:
            print(f"Value: {value}, Range: {borders}, Actual Range: {range_length}, Max Cos Value: {maxCosValue}, Max Cos Value X: {maxCosValueX}")

        return extendedGroups

    def calculateCosinusRanges(self):
        self.calculated_c_ranges = self.getCosSplits(self.original_unchanged_img, self.original_img_annotations)

    def calculateMaximumRanges(self):
        self.calculated_maximum_ranges = self.getMaximumSplits(self.original_unchanged_img, self.original_img_annotations)

    def calculateUnifiedRanges(self):
        self.calculated_unified_ranges = self.getUnifiedSplits(self.original_unchanged_img, self.original_img_annotations)

    def scrollImage(self, requestedSplitX: int):
        self.preview_img = np.copy(self.original_unchanged_img)      
        ImageManager.scrollImage(self.preview_img, requestedSplitX)          
        self.scrolled_resulting_img = np.copy(self.preview_img)
        self.last_scroll = requestedSplitX/self.main_img.shape[1]
        
    def proposeNextCosinusSplit(self):
        ImageManager.removeVerticalLine(self.main_img, self.last_suggested_c_split_x, self.original_unchanged_img)
        value, borders, range_length = self.calculated_c_ranges[self.last_suggested_c_split]
        suggestedSplitX = int((borders[0] + borders[1]) / 2)
        self.last_suggested_c_split += 1
        ImageManager.addVerticalLine(self.main_img, suggestedSplitX, (0, 255, 0))
        self.last_suggested_c_split_x = suggestedSplitX
        self.last_scroll = suggestedSplitX/self.main_img.shape[1]
        self.scrollImage(suggestedSplitX)
        self.scrolledAnnotations = AnnotationManager.scrollAnnotations(self.original_img_annotations, self.last_scroll)
        self.scrolledAnnotations = AnnotationManager.mergeAdjacentObjects(self.scrolledAnnotations, self.last_scroll)
        AnnotationManager.addAnnotationsToImage(self.preview_img, self.scrolledAnnotations, annotationColor=(0, 255, 0))

    def proposeNextMaxSplit(self):
        ImageManager.removeVerticalLine(self.main_img, self.last_suggested_maximum_split_x, self.original_unchanged_img)
        value, borders, range_length, maxCosValue, maxCosValueX = self.calculated_maximum_ranges[self.last_suggested_maximum_split]
        suggestedSplitX = maxCosValueX
        self.last_suggested_maximum_split += 1
        ImageManager.addVerticalLine(self.main_img, suggestedSplitX, (0, 255, 0))
        self.last_suggested_maximum_split_x = suggestedSplitX
        self.last_scroll = suggestedSplitX/self.main_img.shape[1]
        self.scrollImage(suggestedSplitX)
        self.scrolledAnnotations = AnnotationManager.scrollAnnotations(self.original_img_annotations, self.last_scroll)
        self.scrolledAnnotations = AnnotationManager.mergeAdjacentObjects(self.scrolledAnnotations, self.last_scroll)
        AnnotationManager.addAnnotationsToImage(self.preview_img, self.scrolledAnnotations, annotationColor=(0, 255, 0))

    def proposeNextUnifiedSplit(self):
        ImageManager.removeVerticalLine(self.main_img, self.last_suggested_unified_split_x, self.original_unchanged_img)
        suggestedSplitX = random.randint(0, self.main_img.shape[1])
        ImageManager.addVerticalLine(self.main_img, suggestedSplitX, (0, 255, 0))
        self.last_suggested_unified_split_x = suggestedSplitX
        self.last_scroll = suggestedSplitX/self.main_img.shape[1]
        self.scrollImage(suggestedSplitX)
        self.scrolledAnnotations = AnnotationManager.scrollAnnotations(self.original_img_annotations, self.last_scroll)
        self.scrolledAnnotations = AnnotationManager.mergeAdjacentObjects(self.scrolledAnnotations, self.last_scroll)
        AnnotationManager.addAnnotationsToImage(self.preview_img, self.scrolledAnnotations, annotationColor=(0, 255, 0))

    def saveProcessedImage(self):
        pathToProcessedImage = self.imagePaths[self.loaded_image_index]
        scrolledImgOutPath = pathToProcessedImage.replace(Consts.IMAGES_PATH, Consts.SCROLLED_IMAGES_PATH)
        os.makedirs(os.path.dirname(scrolledImgOutPath), exist_ok=True)
        cv2.imwrite(scrolledImgOutPath, self.scrolled_resulting_img)

    def saveScrollValue(self):
        pathToProcessedImage = self.imagePaths[self.loaded_image_index]
        scrolledImgOutPath = pathToProcessedImage.replace(Consts.IMAGES_PATH, Consts.SCROLLED_IMAGES_PATH)
        scrollValOutPath = scrolledImgOutPath.replace(Consts.SCROLLED_IMAGES_PATH, Consts.SCROLL_VALUES_PATH)
        for acceptedImageFormat in Consts.ACCEPTED_IMAGE_FORMATS:
            scrollValOutPath = scrollValOutPath.replace(acceptedImageFormat, "_scroll.txt")
        with open(scrollValOutPath, 'w') as writer:
            writer.write(str(self.last_scroll))

    def saveScrolledAnnotations(self):
        pathToProcessedAnnotations = self.annotationPaths[self.loaded_image_index]
        pathToProcessedAnnotations = pathToProcessedAnnotations.replace(Consts.ANNOTATIONS_PATH, Consts.SCROLLED_ANNOTATIONS_PATH)

        os.makedirs(os.path.dirname(pathToProcessedAnnotations), exist_ok=True)
        with open(pathToProcessedAnnotations, "w") as writer:
            writer.writelines(
                list(
                    map(
                        lambda postProcessedAnnotation: 
                            f"{postProcessedAnnotation.objectType} {postProcessedAnnotation.xCenter} {postProcessedAnnotation.yCenter} {postProcessedAnnotation.width} {postProcessedAnnotation.height}\n", 
                            self.scrolledAnnotations)))