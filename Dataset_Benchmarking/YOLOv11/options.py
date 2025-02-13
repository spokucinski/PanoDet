class Options():
    def __init__(self,
                 resultsPath: str,
                 logName: str,
                 projectName: str,
                 datasetsPath: str,
                 datasetDefsPath: str,
                 datasets: list[str],
                 epochs: list[int],
                 patience: int,
                 models: list[str],
                 batchSizes: list[str],
                 imageSizes: list[str],
                 rectangularTraining: bool,
                 hyperParameters: str,
                 optimizers: list[str]):
        
        self.resultsPath: str = resultsPath
        self.logName: str = logName
        self.projectName: str = projectName
        self.datasetsPath: str = datasetsPath
        self.datasetDefsPath: str = datasetDefsPath
        self.datasets: list[str] = datasets
        self.epochs: list[int] = epochs
        self.patience: int = patience
        self.models: list[str] = models
        self.batchSizes: list[int] = batchSizes
        self.imageSizes: list[int] = imageSizes
        self.rectangularTraining: bool = rectangularTraining
        self.hyperParameters: str = hyperParameters
        self.optimizers: list[str] = optimizers