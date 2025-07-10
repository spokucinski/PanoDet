from dataclasses import dataclass

@dataclass
class RadioPrediction:
    radioId: int
    experiment: str
    room: str
    trackerId: str
    objectId: str
    xr: float
    yr: float
    zr: float
    xrgt: float
    yrgt: float
    zrgt: float
