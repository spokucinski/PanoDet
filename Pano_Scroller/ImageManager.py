import cv2
import numpy as np

def addVerticalLine(image: cv2.typing.MatLike, x: int, color: (int, int, int) = (255, 0, 0), lineThickness:int = 3):
    cv2.line(image, (x, 0), (x, image.shape[0]), color, lineThickness)

def removeVerticalLine(image: cv2.typing.MatLike, x: int, sourceImage: cv2.typing.MatLike, lineThickness:int = 3):
    image[:, x-lineThickness:x+lineThickness] = sourceImage[:, x-lineThickness:x+lineThickness]

def addStatusInfo(image: cv2.typing.MatLike, x: int, imageIndex: int, maxImageIndex: int):
    cv2.putText(img=image, text=f"X = {x}, Image: {imageIndex} / {maxImageIndex}", org=(100, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255), thickness=3, lineType=cv2.LINE_AA)

def removeStatusInfo(image: cv2.typing.MatLike, sourceImage: cv2.typing.MatLike):
    image[0:int(0.1*image.shape[0]), 0:int(0.25*image.shape[1])] = sourceImage[0:int(0.1*image.shape[0]), 0:int(0.25*image.shape[1])]

def scrollImage(image: cv2.typing.MatLike, scrollX: int):
    leftPart = image[:, 0:scrollX]
    rightPart = image[:, scrollX:]

    result = np.concatenate((rightPart, leftPart), axis=1)
    image[:, :] = result[:,:]