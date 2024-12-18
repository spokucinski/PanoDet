class DistanceMeasurement:
    def __init__(self, anchorId: str, distance: float, executionTime: float):
        self.anchorId = anchorId
        self.distance = distance
        self.executionTime = executionTime