import cv2

class SplitProgressMonitor():

    def __init__(self,
                 images: list[str],
                 previewWindowName: str,
                 loaded_image_index: int,
                 max_image_index: int,
                 original_img: cv2.typing.MatLike,
                 marked_img: cv2.typing.MatLike,
                 scrolled_img: cv2.typing.MatLike,
                 last_known_x: int,
                 line_thickness: int,
                 processing: bool):
        
        self.images = images
        self.previewWindowName = previewWindowName
        self.loaded_image_index = loaded_image_index
        self.max_image_index = max_image_index
        self.original_img = original_img
        self.marked_img = marked_img
        self.scrolled_img = scrolled_img
        self.last_known_x = last_known_x
        self.line_thickness = line_thickness
        self.processing = processing