class PanoScrollerArgs():
    
    def __init__(self,
                 mode: str,
                 inputPath: str,
                 mainWindowName: str,
                 previewWindowName: str,
                 imageFormats: list[str]):
        
        self.mode = mode
        self.inputPath = inputPath
        self.mainWindowName = mainWindowName
        self.previewWindowName = previewWindowName
        self.imageFormats = imageFormats