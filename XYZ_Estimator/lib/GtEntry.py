from dataclasses import dataclass

@dataclass
class GtEntry:
    groundTruthId: int
    room: str
    collection: str
    objectId: str
    code55Class: str
    xgt: float
    ygt: float
    zgt: float
