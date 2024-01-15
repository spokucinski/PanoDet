import math
import cv2
import ImageManager

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

def addAnnotationsToImage(image: cv2.typing.MatLike, annotations: list[Annotation], lineThickness: int = 3, annotationColor: (int, int, int) = ((0, 0, 255))):
    image_width, image_height = image.shape[1], image.shape[0]

    for annotation in annotations:
        (x1, y1), (x2, y2) = denormalizeAnnotation(annotation, image_height, image_width)
        cv2.rectangle(image, (x1, y1), (x2, y2), annotationColor, lineThickness)

def denormalizeAnnotation(annotation: Annotation, imageHeight: int, imageWidth: int) -> ((int, int), (int, int)):
    xCenter = annotation.xCenter * imageWidth
    yCenter = annotation.yCenter * imageHeight
    
    width = annotation.width * imageWidth
    height = annotation.height * imageHeight

    x1 = int(xCenter - (width/2))
    y1 = int(yCenter - (height/2))

    x2 = int(xCenter + (width/2))
    y2 = int(yCenter + (height/2))

    return (x1, y1), (x2, y2)

def getFileAnnotations(annotationsFilePath: str) -> list[Annotation]:
    annotations: list[Annotation] = []
    try:
        with open(annotationsFilePath, 'r') as reader:                 
            for annotation in reader.readlines():
                objectType, xCenter, yCenter, width, height = annotation.split(' ')
                annotations.append(Annotation(int(objectType), float(xCenter), float(yCenter), float(width), float(height)))
    finally:
        return annotations

def scrollAnnotations(originalAnnotations: list[Annotation], scroll: float) -> list[Annotation]: 
    scrolledAnnotations: list[Annotation] = []      
    for originalAnnotation in originalAnnotations:

        # Object placed inside picture even after scrolling                
        if originalAnnotation.xCenter - (originalAnnotation.width/2) >= scroll:
            scrolledAnnotations.append(Annotation(originalAnnotation.objectType, originalAnnotation.xCenter - scroll, originalAnnotation.yCenter, originalAnnotation.width, originalAnnotation.height))
        
        # Object fully wrapped and moved to the right part
        elif originalAnnotation.xCenter + (originalAnnotation.width/2) <= scroll:
            scrolledAnnotations.append(Annotation(originalAnnotation.objectType, originalAnnotation.xCenter - scroll + 1, originalAnnotation.yCenter, originalAnnotation.width, originalAnnotation.height))
        
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
    mergingResult: list[Annotation] = scrolledAnnotations
    for scrolledAnnotation in scrolledAnnotations:
        adjacentAnnotations = list(
            filter(
                lambda analyzedAnnotation: 
                math.isclose((scrolledAnnotation.xCenter + (scrolledAnnotation.width/2)), (analyzedAnnotation.xCenter - (analyzedAnnotation.width/2)), rel_tol=0.0005) 
                and
                    ((scrolledAnnotation.yCenter - (scrolledAnnotation.height/2)) < (analyzedAnnotation.yCenter + (analyzedAnnotation.height/2))) 
                    and 
                    ((scrolledAnnotation.yCenter + (scrolledAnnotation.height/2)) > (analyzedAnnotation.yCenter - (analyzedAnnotation.height/2))), 
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

                mergingResult.append(Annotation(scrolledAnnotation.objectType, newXCenter, newYCenter, newWidth, newHeight))
                mergingResult.remove(scrolledAnnotation)
                mergingResult = list(filter(lambda i: i not in adjacentAnnotationOfSameTypeOriginallyOnTheEdge, mergingResult))
            
    return mergingResult