import cv2
import Consts
import numpy as np
import matplotlib.pyplot as plt

from screeninfo import get_monitors

def initializeBaseWindows(processEvent, processParams):
    # Creates empty windows for base pano scrolling images

    # App is an UI-based app, so at least one screen is required
    monitors = get_monitors()
    if len(monitors) < 1:
        raise Exception("No available monitors detected! UI is required to use this app!")

    # Base windows are expected to be placed on the very first screen, on top
    firstMonitorWidth = monitors[0].width
    defaultWindowWidth: int = int(firstMonitorWidth/2)
    defaultWindowHeight: int = int(firstMonitorWidth/4)

    # Main window in the top-left corner
    emptyMainWindowImage = np.zeros((defaultWindowHeight, defaultWindowWidth, 3), np.uint8)  
    cv2.namedWindow(Consts.WINDOW_MAIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(Consts.WINDOW_MAIN, defaultWindowWidth, defaultWindowHeight)
    cv2.moveWindow(Consts.WINDOW_MAIN, 0, 0)
    cv2.imshow(Consts.WINDOW_MAIN, emptyMainWindowImage) 

    # Preview window in the top-right corner
    emptyPreviewWindowImage = np.zeros((defaultWindowHeight, defaultWindowWidth, 3), np.uint8)
    cv2.namedWindow(Consts.WINDOW_PREVIEW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(Consts.WINDOW_PREVIEW, defaultWindowWidth, defaultWindowHeight)
    cv2.moveWindow(Consts.WINDOW_PREVIEW, defaultWindowWidth, 0)
    cv2.imshow(Consts.WINDOW_PREVIEW, emptyPreviewWindowImage)

    cv2.setMouseCallback(Consts.WINDOW_MAIN, processEvent, processParams)

def initializeControlWindows():
    # Creates empty windows for pano scrolling overview monitoring.

    # App is an UI-based app, so at least one screen is required
    monitors = get_monitors()
    if len(monitors) < 1:
        raise Exception("No available monitors detected! UI is required to use this app!")
    
    # By default control windows are placed in the corners of screen.
    # If only one screen is available - in the corners of the first screen
    # If more screens are avaialable - in the corners of the second screen
    firstMonitorWidth: int = monitors[0].width
    defaultWindowWidth: int = int(firstMonitorWidth/2)
    defaultWindowHeight: int = int(firstMonitorWidth/4)

    # Initialize and move windows to the default position   
    emptyAnnotationImage = np.zeros((defaultWindowHeight, defaultWindowWidth, 3), np.uint8)
    cv2.namedWindow(Consts.WINDOW_ANN_CTR, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(Consts.WINDOW_ANN_CTR, defaultWindowWidth, defaultWindowHeight)
    cv2.moveWindow(Consts.WINDOW_ANN_CTR, 0, 0)
    cv2.imshow(Consts.WINDOW_ANN_CTR, emptyAnnotationImage)

    emptyWeightsImage = np.zeros((defaultWindowHeight, defaultWindowWidth, 3), np.uint8)
    cv2.namedWindow(Consts.WINDOW_W_CTR, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(Consts.WINDOW_W_CTR, defaultWindowWidth, defaultWindowHeight)
    cv2.moveWindow(Consts.WINDOW_W_CTR, defaultWindowWidth, 0)
    cv2.imshow(Consts.WINDOW_W_CTR, emptyWeightsImage)

    emptyWeightedAnnotationImage = np.zeros((defaultWindowHeight, defaultWindowWidth, 3), np.uint8)
    cv2.namedWindow(Consts.WINDOW_W_ANN_CTR, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(Consts.WINDOW_W_ANN_CTR, defaultWindowWidth, defaultWindowHeight)
    cv2.moveWindow(Consts.WINDOW_W_ANN_CTR, 0, defaultWindowHeight)
    cv2.imshow(Consts.WINDOW_W_ANN_CTR, emptyWeightedAnnotationImage)

    emptyColoredWeightedAnnotationImage = np.zeros((defaultWindowHeight, defaultWindowWidth, 3), np.uint8)
    cv2.namedWindow(Consts.WINDOW_C_W_ANN_CTR, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(Consts.WINDOW_C_W_ANN_CTR, defaultWindowWidth, defaultWindowHeight)
    cv2.moveWindow(Consts.WINDOW_C_W_ANN_CTR, defaultWindowWidth, defaultWindowHeight)
    cv2.imshow(Consts.WINDOW_C_W_ANN_CTR, emptyColoredWeightedAnnotationImage)
    
    if len(monitors) > 1:
        # Just move the already initialized windows to an updated position
        secondMonitorWidth: int = monitors[1].width
        adjustedWindowWidth: int = int(secondMonitorWidth/2)
        adjustedWindowHeight: int = int(secondMonitorWidth/4)

        cv2.resizeWindow(Consts.WINDOW_ANN_CTR, adjustedWindowWidth, adjustedWindowHeight)
        cv2.moveWindow(Consts.WINDOW_ANN_CTR, firstMonitorWidth + 0, 0)

        cv2.resizeWindow(Consts.WINDOW_W_CTR, adjustedWindowWidth, adjustedWindowHeight)
        cv2.moveWindow(Consts.WINDOW_W_CTR, firstMonitorWidth + adjustedWindowWidth, 0)

        cv2.resizeWindow(Consts.WINDOW_W_ANN_CTR, adjustedWindowWidth, adjustedWindowHeight)
        cv2.moveWindow(Consts.WINDOW_W_ANN_CTR, firstMonitorWidth + 0, adjustedWindowHeight)

        cv2.resizeWindow(Consts.WINDOW_C_W_ANN_CTR, adjustedWindowWidth, adjustedWindowHeight)
        cv2.moveWindow(Consts.WINDOW_C_W_ANN_CTR, firstMonitorWidth + adjustedWindowWidth, adjustedWindowHeight)

    
def initializePlot(xLabel: str = None, yLabel: str = None, plotName: str = None):
    # Initialize plot
    figure, axes = plt.subplots()

    if xLabel:
        axes.set_xlabel(xLabel)

    if yLabel:
        axes.set_ylabel(yLabel)

    if xLabel or yLabel:
        axes.legend()

    # Plot a placeholding function
    plotedLine, = axes.plot(range(10), range(10))

    if plotName:
        plotedLine.set_label(plotName)

    # Unbound from the UI thread
    plt.show(block=False)

    # App is an UI-based app, so at least one screen is required
    monitors = get_monitors()
    if len(monitors) < 1:
        raise Exception("No available monitors detected! UI is required to use this app!")

    # By default put the figure in the middle of the first screen
    figureManager = plt.get_current_fig_manager()
    windowGeometry = figureManager.window.geometry()
    x, y, plotWindowWidth, plotWindowHeight = windowGeometry.getRect()

    defaultPlotX: int = int(monitors[0].width/2) - int(plotWindowWidth/2)
    defaultPlotY: int = int(monitors[0].height/2) - int(plotWindowHeight/2)
    figureManager.window.setGeometry(defaultPlotX, defaultPlotY, plotWindowWidth, plotWindowHeight)
    
    if len(monitors) > 1:
        # If more screens available - move the figure to the middle of the second screen
        adjustedPlotX: int = int(monitors[1].width/2) - int(plotWindowWidth/2) + monitors[0].width
        adjustedPlotY: int = int(monitors[1].height/2) - int(plotWindowHeight/2)
        figureManager.window.setGeometry(adjustedPlotX, adjustedPlotY, plotWindowWidth, plotWindowHeight)
        
    return figure, axes, plotedLine

def updateAnnotationControlView(annotationsMatrix: np.ndarray):
    colorMap = {
        0: [0, 0, 0],      # Black
        1: [0, 255, 0],    # Green
        2: [255, 255, 0],  # Yellow
        3: [255, 0, 0],    # Red
        4: [128, 0, 128],  # Purple
        5: [150, 75, 0]    # Brown
    }

    controlImage = np.zeros((annotationsMatrix.shape[0], annotationsMatrix.shape[1], 3), dtype=np.uint8)
    for value, color in colorMap.items():
        controlImage[annotationsMatrix == value] = color

    controlImage = cv2.cvtColor(controlImage, cv2.COLOR_RGB2BGR)
    cv2.imshow(Consts.WINDOW_ANN_CTR, controlImage)

def updateWeightsControlView(weightsMatrixSource: np.ndarray):
    # Shows copied matrix as a black-white image
    weightsMatrixCopy = np.copy(weightsMatrixSource)
    weightsMatrixCopy = np.multiply(weightsMatrixCopy, 255)
    cv2.imshow(Consts.WINDOW_W_CTR, weightsMatrixCopy.astype(np.uint8))

def updateWeightedAnnotationsControlView(weightedAnnotationsSource: np.ndarray):
    # Shows copied matrix as a black-white image
    weightedAnnotationsCopy = np.copy(weightedAnnotationsSource)

    # Normalize to the range 0-1
    weightedAnnotationsCopy = np.interp(weightedAnnotationsCopy, (weightedAnnotationsCopy.min(), weightedAnnotationsCopy.max()), (0, 1))
    weightedAnnotationsCopy = np.multiply(weightedAnnotationsCopy, 255)
    cv2.imshow(Consts.WINDOW_W_ANN_CTR, weightedAnnotationsCopy.astype(np.uint8))

def updateColoredWeightedAnnotationsControlView(weightedAnnotationsSource: np.ndarray):
    weightedAnnotationsImage = np.zeros((weightedAnnotationsSource.shape[0], weightedAnnotationsSource.shape[1], 3), dtype=np.uint8)
    
    weightsColorMap = {
        0:   [10, 0, 0],         # Light Red
        0.1: [76, 0, 0],         # Salmon Red
        0.2: [144, 0, 0],        # Soft Red
        0.3: [210, 0, 0],        # Tomato Red
        0.4: [255, 23, 0],       # Indian Red
        0.5: [255, 91, 0],       # Pure Red
        0.6: [255, 157, 0],      # Firebrick Red
        0.7: [255, 225, 0],      # Crimson Red
        0.8: [255, 255, 54],     # Dark Red
        0.9: [255, 255, 156]     # Deep Red
    }
    
    for threshold, color in weightsColorMap.items():
        weightedAnnotationsImage[weightedAnnotationsSource > threshold] = color 
    weightedAnnotationsImage = cv2.cvtColor(weightedAnnotationsImage, cv2.COLOR_RGB2BGR)
    cv2.imshow(Consts.WINDOW_C_W_ANN_CTR, weightedAnnotationsImage)

def updatePlot(weightedColumns: np.ndarray, figure, axes, line, lastSuggestedX):
    updatedValuesX = np.arange(len(weightedColumns))
    
    line.set_xdata(updatedValuesX)
    line.set_ydata(weightedColumns)
    
    if lastSuggestedX > 0:
        # Debug - mark suggested line
        axes.axvline(x=lastSuggestedX, color='red', linestyle=':', label=f'Suggested X: {lastSuggestedX}')

    np.savetxt('array.txt', weightedColumns, fmt='%d')
    axes.set_xlim(0, max(updatedValuesX))
    axes.set_ylim(0, max(weightedColumns))
    axes.autoscale_view()

    figure.canvas.draw()
    figure.canvas.flush_events()