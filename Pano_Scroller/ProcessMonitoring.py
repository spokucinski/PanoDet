import cv2
from Scroller import Annotation

class SplitProgressMonitor():

    def __init__(self,
                 imagePaths: list[str],
                 annotationPaths: list[str],
                 previewWindowName: str,
                 loaded_image_index: int,
                 max_image_index: int,
                 original_img: cv2.typing.MatLike,
                 original_img_annotations: list[Annotation],
                 marked_img: cv2.typing.MatLike,
                 scrolled_img: cv2.typing.MatLike,
                 last_known_x: int,
                 line_thickness: int,
                 processing: bool,
                 last_scroll: float,
                 last_suggested_c_split: int,
                 last_suggested_std_split: int,
                 calculated_c_ranges,
                 calculated_std_ranges):
        
        self.imagePaths = imagePaths
        self.annotationPaths = annotationPaths
        self.previewWindowName = previewWindowName
        self.loaded_image_index = loaded_image_index
        self.max_image_index = max_image_index
        self.original_img = original_img
        self.original_img_annotations = original_img_annotations
        self.marked_img = marked_img
        self.scrolled_img = scrolled_img
        self.last_known_x = last_known_x
        self.line_thickness = line_thickness
        self.processing = processing
        self.last_scroll = last_scroll
        self.last_suggested_c_split = last_suggested_c_split
        self.last_suggested_std_split = last_suggested_std_split
        self.calculated_c_ranges = calculated_c_ranges
        self.calculated_std_ranges = calculated_std_ranges