class DistanceMeasurement:
    def __init__(self, anchor_id: str, distance: float):
        self.anchor_id = anchor_id
        self.distance = distance

    def to_dict(self):
        return {"anchor_id": self.anchor_id, "distance": self.distance}