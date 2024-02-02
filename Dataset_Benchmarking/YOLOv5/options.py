class TrainingOptions():
    def __init__(self,
                 resultsPath: str,
                 logName: str,
                 projectName: str,
                 datasetsPath: str,
                 datasetDefsPath: str,
                 epochs: list[int],
                 patience: int,
                 models: list[str],
                 batchSizes: list[str],
                 imageSizes: list[str],
                 rectangularTraining: bool,
                 hyperParameters: str):
        
        self.resultsPath: str = resultsPath
        self.logName: str = logName
        self.projectName: str = projectName
        self.datasetsPath: str = datasetsPath
        self.datasetDefsPath: str = datasetDefsPath
        self.epochs: list[int] = epochs
        self.patience: int = patience
        self.models: list[str] = models
        self.batchSizes: list[int] = batchSizes
        self.imageSizes: list[int] = imageSizes
        self.rectangularTraining: bool = rectangularTraining
        self.hyperParameters: str = hyperParameters