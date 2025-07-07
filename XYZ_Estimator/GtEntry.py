from dataclasses import dataclass

@dataclass
class GtEntry:
    ground_truth_id: int
    room: str
    collection: str
    object_id: str
    code55: str
    xgt: float
    ygt: float
    zgt: float
