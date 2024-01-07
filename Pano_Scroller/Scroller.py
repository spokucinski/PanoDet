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

def getFileAnnotations(annotationsFilePath: str) -> list[Annotation]:
    annotations: list[Annotation] = []
    try:
        with open(annotationsFilePath, 'r') as reader:                 
            for annotation in reader.readlines():
                objectType, xCenter, yCenter, width, height = annotation.split(' ')
                annotations.append(Annotation(int(objectType), float(xCenter), float(yCenter), float(width), float(height)))
    finally:
        return annotations