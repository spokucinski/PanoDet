from dataclasses import dataclass

@dataclass
class Detection:
    detection_id: int
    rel_det_number: int
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
    image_id: str
    detected_object_id: str
    detection_correct: bool
    detected_in_room: str
