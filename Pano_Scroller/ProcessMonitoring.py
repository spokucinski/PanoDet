import cv2
from AnnotationManager import Annotation

class SplitProgressMonitor():

    def __init__(self,
                 imagePaths: list[str],
                 annotationPaths: list[str],
                 loaded_image_index: int,
                 max_image_index: int,
                 original_unchanged_img: cv2.typing.MatLike,
                 original_img_annotations: list[Annotation],
                 main_img: cv2.typing.MatLike,
                 preview_img: cv2.typing.MatLike,
                 scrolled_resulting_img: cv2.typing.MatLike,
                 last_known_x: int,
                 line_thickness: int,
                 processing: bool,
                 last_scroll: float,
                 last_suggested_c_split: int,
                 last_suggested_std_split: int,
                 calculated_c_ranges,
                 calculated_std_ranges,
                 controlWindowsInitialized: bool,
                 controlFigure,
                 controlAxes,
                 controlPlottedLine):
        
        self.imagePaths = imagePaths
        self.annotationPaths = annotationPaths
        self.loaded_image_index = loaded_image_index
        self.max_image_index = max_image_index
        self.original_unchanged_img = original_unchanged_img
        self.original_img_annotations = original_img_annotations
        self.main_img = main_img
        self.preview_img = preview_img
        self.scrolled_resulting_img = scrolled_resulting_img
        self.last_known_x = last_known_x
        self.line_thickness = line_thickness
        self.processing = processing
        self.last_scroll = last_scroll
        self.last_suggested_c_split = last_suggested_c_split
        self.last_suggested_std_split = last_suggested_std_split
        self.calculated_c_ranges = calculated_c_ranges
        self.calculated_std_ranges = calculated_std_ranges
        self.controlWindowsInitialized = controlWindowsInitialized
        self.controlFigure = controlFigure
        self.controlAxes = controlAxes
        self.controlPlottedLine = controlPlottedLine