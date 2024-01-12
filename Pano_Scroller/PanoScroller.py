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
            AnnotationManager.addAnnotationsOverlay(scrollingProcess.main_img, x, scrollingProcess.original_img_annotations, scrollingProcess.loaded_image_index, scrollingProcess.max_image_index) 
            scrollingProcess.last_known_x = x 

        elif event == cv2.EVENT_LBUTTONDOWN:
            scrollingProcess.scrollImage(x)

        elif event == cv2.EVENT_RBUTTONDOWN:
            scrollingProcess.saveProcessedImage()           
            scrollingProcess.saveScrollValue()

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

        # C suggests split point with COS(f)
        if k == 99:
            if not scrollingProcess.controlWindowsInitialized:
                scrollingProcess.initializeControlFlow()

            if not scrollingProcess.calculated_c_ranges:
                scrollingProcess.calculateCosinusRanges()

            if scrollingProcess.last_suggested_c_split >= len(scrollingProcess.calculated_c_ranges):
                scrollingProcess.last_suggested_c_split = 0

            scrollingProcess.proposeNextCosinusSplit()
            
        # if k == 115:
        #     suggestSplitStd(scrollingProcess)

        # ESC escapes
        if k == 27:
            break

    print("Ending PanoScroller, closing the app!")

if __name__ == '__main___':
    main()
else:
    main()