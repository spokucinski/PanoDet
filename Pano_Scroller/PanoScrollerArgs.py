class PanoScrollerArgs():
    
    def __init__(self,
                 mode: str,
                 inputPath: str,
                 mainWindowName: str,
                 previewWindowName: str,
                 imageFormats: list[str]):
        
        self.mode = mode
        self.inputPath = inputPath
        self.mainWindow = mainWindowName
        self.previewWindow = previewWindowName
        self.imageFormats = imageFormats