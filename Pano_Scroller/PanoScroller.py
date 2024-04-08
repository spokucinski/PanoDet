import cv2
from ProcessMonitoring import ScrollingProcess
import AnnotationManager
import DataManager
import ImageManager
import Consts
   
def mainWindowCallback(event, x, y, flags, scrollingProcess):
        if event == cv2.EVENT_MOUSEMOVE:
            ImageManager.removeVerticalLine(scrollingProcess.main_img, scrollingProcess.last_known_x, scrollingProcess.original_unchanged_img)
            ImageManager.removeStatusInfo(scrollingProcess.main_img, scrollingProcess.original_unchanged_img)
            
            AnnotationManager.addAnnotationsToImage(scrollingProcess.main_img, scrollingProcess.original_img_annotations)
            
            ImageManager.addVerticalLine(scrollingProcess.main_img, x)

            lastSuggestedX = max(scrollingProcess.last_suggested_c_split_x,
                                 scrollingProcess.last_suggested_maximum_split_x,
                                 scrollingProcess.last_suggested_unified_split_x)
            
            if lastSuggestedX > 0:
                ImageManager.addStatusInfo(scrollingProcess.main_img, x, scrollingProcess.loaded_image_index, scrollingProcess.max_image_index, lastSuggestedX)
            else:
                ImageManager.addStatusInfo(scrollingProcess.main_img, x, scrollingProcess.loaded_image_index, scrollingProcess.max_image_index)
           
            scrollingProcess.last_known_x = x 

        elif event == cv2.EVENT_LBUTTONDOWN:
            scrollingProcess.scrollImage(x)
            scrollingProcess.scrolledAnnotations = AnnotationManager.scrollAnnotations(scrollingProcess.original_img_annotations, scrollingProcess.last_scroll)
            scrollingProcess.scrolledAnnotations = AnnotationManager.mergeAdjacentObjects(scrollingProcess.scrolledAnnotations, scrollingProcess.last_scroll)
            AnnotationManager.addAnnotationsToImage(scrollingProcess.preview_img, scrollingProcess.scrolledAnnotations, annotationColor=(0, 255, 0))

        elif event == cv2.EVENT_RBUTTONDOWN:
            scrollingProcess.saveProcessedImage()           
            scrollingProcess.saveScrollValue()
            scrollingProcess.saveScrolledAnnotations()

            if scrollingProcess.loaded_image_index >= scrollingProcess.max_image_index:
                scrollingProcess.processing = False
            else:
                scrollingProcess.loadNextImage()            

def main():
    print("Starting PanoScroller!")
    imagePaths, annotationPaths = DataManager.getInput()
    
    if not imagePaths or len(imagePaths) == 0:
        raise Exception("No input files detected, ending processing")

    scrollingProcess = ScrollingProcess(imagePaths, annotationPaths)
    scrollingProcess.loadNextImage()
    scrollingProcess.initializeMainFlow(mainWindowCallback)

    # Main Program Loop
    while (scrollingProcess.processing):
        cv2.imshow(Consts.WINDOW_MAIN, scrollingProcess.main_img)
        cv2.imshow(Consts.WINDOW_PREVIEW, scrollingProcess.preview_img)
        k = cv2.waitKey(20) & 0xFF

        # c suggests split point with COS(f)
        if k == 99:
            if not scrollingProcess.controlWindowsInitialized:
                scrollingProcess.initializeControlFlow()

            if not scrollingProcess.calculated_c_ranges:
                scrollingProcess.calculateCosinusRanges()

            if scrollingProcess.last_suggested_c_split >= len(scrollingProcess.calculated_c_ranges):
                scrollingProcess.last_suggested_c_split = 0

            scrollingProcess.proposeNextCosinusSplit()

        # b suggests split point with MAX
        if k == 98:
            if not scrollingProcess.controlWindowsInitialized:
                scrollingProcess.initializeControlFlow()
            
            if not scrollingProcess.calculated_maximum_ranges:
                scrollingProcess.calculateMaximumRanges()

            if scrollingProcess.last_suggested_maximum_split >= len(scrollingProcess.calculated_maximum_ranges):
                scrollingProcess.last_suggested_maximum_split = 0

            scrollingProcess.proposeNextMaxSplit()

        # r suggests split point with uniform distribution
        if k == 114:
            scrollingProcess.proposeNextUnifiedSplit()

        # ESC escapes
        if k == 27:
            break

        # Enter for Auto-Min Scrolling
        if k == 13:
            while (scrollingProcess.processing):
                if not scrollingProcess.controlWindowsInitialized:
                    scrollingProcess.initializeControlFlow()

                if not scrollingProcess.calculated_c_ranges:
                    scrollingProcess.calculateCosinusRanges()

                if scrollingProcess.last_suggested_c_split >= len(scrollingProcess.calculated_c_ranges):
                    scrollingProcess.last_suggested_c_split = 0

                scrollingProcess.proposeNextCosinusSplit()

                scrollingProcess.saveProcessedImage()           
                scrollingProcess.saveScrollValue()
                scrollingProcess.saveScrolledAnnotations()

                if scrollingProcess.loaded_image_index >= scrollingProcess.max_image_index:
                    scrollingProcess.processing = False
                else:
                    scrollingProcess.loadNextImage()

        # t for auto-unified scrolling
        if k == 116:
            while (scrollingProcess.processing):
                scrollingProcess.proposeNextUnifiedSplit()
                scrollingProcess.saveProcessedImage()
                scrollingProcess.saveScrollValue()
                scrollingProcess.saveScrolledAnnotations()

                if scrollingProcess.loaded_image_index >= scrollingProcess.max_image_index:
                    scrollingProcess.processing = False
                else:
                    scrollingProcess.loadNextImage()

        # Space for auto max scrolling
        if k == 32:
            while (scrollingProcess.processing):
                if not scrollingProcess.controlWindowsInitialized:
                    scrollingProcess.initializeControlFlow()

                if not scrollingProcess.calculated_maximum_ranges:
                    scrollingProcess.calculateMaximumRanges()

                if scrollingProcess.last_suggested_maximum_split >= len(scrollingProcess.calculated_maximum_ranges):
                    scrollingProcess.last_suggested_maximum_split = 0

                scrollingProcess.proposeNextMaxSplit()

                scrollingProcess.saveProcessedImage()           
                scrollingProcess.saveScrollValue()
                scrollingProcess.saveScrolledAnnotations()

                if scrollingProcess.loaded_image_index >= scrollingProcess.max_image_index:
                    scrollingProcess.processing = False
                else:
                    scrollingProcess.loadNextImage()

    print("Ending PanoScroller, closing the app!")

if __name__ == '__main___':
    main()
else:
    main()