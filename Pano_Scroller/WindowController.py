import cv2
import Consts as consts
import numpy as np
import matplotlib.pyplot as plt

from screeninfo import get_monitors

def initializeBaseWindows():
    monitorParams = get_monitors()
    if len(monitorParams) > 0:
        monitorWidth = monitorParams[0].width
        windowWidth: int = int(monitorWidth/2)
        windowHeight: int = int(monitorWidth/4)

        emptyMainWindowImage = np.zeros((windowHeight, windowWidth, 3), np.uint8)  
        cv2.namedWindow(consts.WINDOW_MAIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(consts.WINDOW_MAIN, windowWidth, windowHeight)
        cv2.moveWindow(consts.WINDOW_MAIN, 0, 0)
        cv2.imshow(consts.WINDOW_MAIN, emptyMainWindowImage) 

        emptyPreviewWindowImage = np.zeros((windowHeight, windowWidth, 3), np.uint8)
        cv2.namedWindow(consts.WINDOW_PREVIEW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(consts.WINDOW_PREVIEW, windowWidth, windowHeight)
        cv2.moveWindow(consts.WINDOW_PREVIEW, windowWidth, 0)
        cv2.imshow(consts.WINDOW_PREVIEW, emptyPreviewWindowImage)
    
    else:
        raise Exception("No available monitors detected! UI is required to use this app!")

def initializeControlWindows(debugMode: bool = True):
    monitorParams = get_monitors()
    if len(monitorParams) > 1:
        firstMonitorWidth: int = monitorParams[0].width
        secondMonitorWidth: int = monitorParams[1].width
        windowWidth: int = int(secondMonitorWidth/2)
        windowHeight: int = int(secondMonitorWidth/4)

        if debugMode:
            emptyAnnotationImage = np.zeros((windowHeight, windowWidth, 3), np.uint8)
            cv2.namedWindow(consts.WINDOW_ANN_CTR, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(consts.WINDOW_ANN_CTR, windowWidth, windowHeight)
            cv2.moveWindow(consts.WINDOW_ANN_CTR, firstMonitorWidth + 0, 0)
            cv2.imshow(consts.WINDOW_ANN_CTR, emptyAnnotationImage)

            emptyWeightsImage = np.zeros((windowHeight, windowWidth, 3), np.uint8)
            cv2.namedWindow(consts.WINDOW_W_CTR, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(consts.WINDOW_W_CTR, windowWidth, windowHeight)
            cv2.moveWindow(consts.WINDOW_W_CTR, firstMonitorWidth + windowWidth, 0)
            cv2.imshow(consts.WINDOW_W_CTR, emptyWeightsImage)

            emptyWeightedAnnotationImage = np.zeros((windowHeight, windowWidth, 3), np.uint8)
            cv2.namedWindow(consts.WINDOW_W_ANN_CTR, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(consts.WINDOW_W_ANN_CTR, windowWidth, windowHeight)
            cv2.moveWindow(consts.WINDOW_W_ANN_CTR, firstMonitorWidth + 0, windowHeight)
            cv2.imshow(consts.WINDOW_W_ANN_CTR, emptyWeightedAnnotationImage)

            emptyColoredWeightedAnnotationImage = np.zeros((windowHeight, windowWidth, 3), np.uint8)
            cv2.namedWindow(consts.WINDOW_C_W_ANN_CTR, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(consts.WINDOW_C_W_ANN_CTR, windowWidth, windowHeight)
            cv2.moveWindow(consts.WINDOW_C_W_ANN_CTR, firstMonitorWidth + windowWidth, windowHeight)
            cv2.imshow(consts.WINDOW_C_W_ANN_CTR, emptyColoredWeightedAnnotationImage)
        else:
            emptyStatusMonitorImage = np.zeros((int(secondMonitorWidth/2), secondMonitorWidth, 3), np.uint8)
            cv2.namedWindow(consts.WINDOW_CONTROL, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(consts.WINDOW_CONTROL, secondMonitorWidth, int(secondMonitorWidth/2))
            cv2.moveWindow(consts.WINDOW_CONTROL, firstMonitorWidth, 0)
            cv2.imshow(consts.WINDOW_CONTROL, emptyStatusMonitorImage)

def initializeWeightsPlot():
    figure, axes = plt.subplots()
    xAxis = range(10)
    figure.set_label("Empty figure")
    plotedLine, = axes.plot(xAxis, xAxis)
    plt.show(block=False)

    monitorParams = get_monitors()
    if len(monitorParams) > 1:
        addX: int = monitorParams[1].width    
        mngr = plt.get_current_fig_manager()
        geom = mngr.window.geometry()
        x,y,dx,dy = geom.getRect()
        mngr.window.setGeometry(x + addX, y, dx, dy)
        

    return figure, axes, plotedLine