from dataclasses import dataclass

@dataclass
class Detection:
    detectionId: int
    relativeDetectionNumber: int
    label: str
    distance: float
    phi: float
    theta: float
    xglobal: float
    yglobal: float
    zglobal: float
    xradio: float
    yradio: float
    zradio: float
    room: str
    imageId: str
    detectedObjectId: str
    detectionCorrect: bool
    detectedInRoom: str
