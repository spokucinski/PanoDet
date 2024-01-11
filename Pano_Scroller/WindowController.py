import cv2
import WindowNames as wns
import numpy as np
import matplotlib.pyplot as plt

from screeninfo import get_monitors

def initializeBaseWindows():
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
    cv2.namedWindow(wns.WINDOW_MAIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(wns.WINDOW_MAIN, defaultWindowWidth, defaultWindowHeight)
    cv2.moveWindow(wns.WINDOW_MAIN, 0, 0)
    cv2.imshow(wns.WINDOW_MAIN, emptyMainWindowImage) 

    # Preview window in the top-right corner
    emptyPreviewWindowImage = np.zeros((defaultWindowHeight, defaultWindowWidth, 3), np.uint8)
    cv2.namedWindow(wns.WINDOW_PREVIEW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(wns.WINDOW_PREVIEW, defaultWindowWidth, defaultWindowHeight)
    cv2.moveWindow(wns.WINDOW_PREVIEW, defaultWindowWidth, 0)
    cv2.imshow(wns.WINDOW_PREVIEW, emptyPreviewWindowImage)

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
    cv2.namedWindow(wns.WINDOW_ANN_CTR, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(wns.WINDOW_ANN_CTR, defaultWindowWidth, defaultWindowHeight)
    cv2.moveWindow(wns.WINDOW_ANN_CTR, 0, 0)
    cv2.imshow(wns.WINDOW_ANN_CTR, emptyAnnotationImage)

    emptyWeightsImage = np.zeros((defaultWindowHeight, defaultWindowWidth, 3), np.uint8)
    cv2.namedWindow(wns.WINDOW_W_CTR, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(wns.WINDOW_W_CTR, defaultWindowWidth, defaultWindowHeight)
    cv2.moveWindow(wns.WINDOW_W_CTR, defaultWindowWidth, 0)
    cv2.imshow(wns.WINDOW_W_CTR, emptyWeightsImage)

    emptyWeightedAnnotationImage = np.zeros((defaultWindowHeight, defaultWindowWidth, 3), np.uint8)
    cv2.namedWindow(wns.WINDOW_W_ANN_CTR, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(wns.WINDOW_W_ANN_CTR, defaultWindowWidth, defaultWindowHeight)
    cv2.moveWindow(wns.WINDOW_W_ANN_CTR, 0, defaultWindowHeight)
    cv2.imshow(wns.WINDOW_W_ANN_CTR, emptyWeightedAnnotationImage)

    emptyColoredWeightedAnnotationImage = np.zeros((defaultWindowHeight, defaultWindowWidth, 3), np.uint8)
    cv2.namedWindow(wns.WINDOW_C_W_ANN_CTR, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(wns.WINDOW_C_W_ANN_CTR, defaultWindowWidth, defaultWindowHeight)
    cv2.moveWindow(wns.WINDOW_C_W_ANN_CTR, defaultWindowWidth, defaultWindowHeight)
    cv2.imshow(wns.WINDOW_C_W_ANN_CTR, emptyColoredWeightedAnnotationImage)
    
    if len(monitors) > 1:
        # Just move the already initialized windows to an updated position
        secondMonitorWidth: int = monitors[1].width
        adjustedWindowWidth: int = int(secondMonitorWidth/2)
        adjustedWindowHeight: int = int(secondMonitorWidth/4)

        cv2.resizeWindow(wns.WINDOW_ANN_CTR, adjustedWindowWidth, adjustedWindowHeight)
        cv2.moveWindow(wns.WINDOW_ANN_CTR, firstMonitorWidth + 0, 0)

        cv2.resizeWindow(wns.WINDOW_W_CTR, adjustedWindowWidth, adjustedWindowHeight)
        cv2.moveWindow(wns.WINDOW_W_CTR, firstMonitorWidth + adjustedWindowWidth, 0)

        cv2.resizeWindow(wns.WINDOW_W_ANN_CTR, adjustedWindowWidth, adjustedWindowHeight)
        cv2.moveWindow(wns.WINDOW_W_ANN_CTR, firstMonitorWidth + 0, adjustedWindowHeight)

        cv2.resizeWindow(wns.WINDOW_C_W_ANN_CTR, adjustedWindowWidth, adjustedWindowHeight)
        cv2.moveWindow(wns.WINDOW_C_W_ANN_CTR, firstMonitorWidth + adjustedWindowWidth, adjustedWindowHeight)

    
def initializeWeightsPlot():
    # Initialize plot
    figure, axes = plt.subplots()
    # Plot a placeholding function
    plotedLine, = axes.plot(range(10), range(10))
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