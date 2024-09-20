class Error:
    def __init__(self, errorType: str, executionTime: float):
        self.errorType = errorType
        self.executionTime = executionTime