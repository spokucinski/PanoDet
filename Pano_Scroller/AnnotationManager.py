import cv2

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