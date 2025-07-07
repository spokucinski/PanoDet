from dataclasses import dataclass

@dataclass
class RadioDetection:
    radio_id: int
    experiment: str
    room: str
    tracker_id: str
    object_id: str
    xr: float
    yr: float
    zr: float
    xrgt: float
    yrgt: float
    zrgt: float
