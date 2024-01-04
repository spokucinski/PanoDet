import cv2
import numpy as np
import os
import argparse
from PanoScrollerArgs import PanoScrollerArgs
from ProcessMonitoring import SplitProgressMonitor

def loadArgs() -> PanoScrollerArgs:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="manual")
    parser.add_argument("--inputPath", default="input")
    parser.add_argument("--mainWindowName", default="Source image")
    parser.add_argument("--previewWindowName", default="Preview")
    parser.add_argument("--imageFormats", nargs='+', default=[".jpg", ".png"])

    print("\nLoading program parameters...")
    cmdArgs = parser.parse_args()
    args: PanoScrollerArgs = PanoScrollerArgs(cmdArgs.mode, cmdArgs.inputPath, cmdArgs.mainWindowName, cmdArgs.previewWindowName, cmdArgs.imageFormats)
    print("Parameters loaded, listing:")
    print(', '.join("%s: %s" % item for item in vars(args).items()))

    return args

def loadImages(inputPath: str, imageFormats: [str]) -> list[str]:
    print("\nLoading input files...")
    print(f"Searched path: {inputPath}")
    
    images = []
    for root, _, files in os.walk(inputPath):
        for file in files:
            if file.lower().endswith(tuple(imageFormats)):
                images.append(os.path.join(root, file))
    foundImagesCount: int = len(images)
    print(f"Found: {foundImagesCount} images!")

    if foundImagesCount == 0:
        print("No images found! Ending PanoScroller.")
        return
    elif foundImagesCount < 100:
        print("Listing:")
        print('\n '.join("%s" % image for image in images))
    else:
        print("Found over 100 images, skipping listing.")

    return images

def initializeWindows(processParams: SplitProgressMonitor, mainWinName: str, previewWinName: str):  
    cv2.namedWindow(mainWinName, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(mainWinName, 1800, 900)
    cv2.setMouseCallback(mainWinName, split_image, processParams)

    cv2.namedWindow(previewWinName, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(previewWinName, 1800, 900)

def split_image(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            param.marked_img[:, param.last_known_x-param.line_thickness:param.last_known_x+param.line_thickness] = param.original_img[:, param.last_known_x-param.line_thickness:param.last_known_x+param.line_thickness]
            param.last_known_x = x
            cv2.line(param.marked_img, (x, 0), (x, param.marked_img.shape[0]), (0, 0, 255), param.line_thickness)
            cv2.putText(img=param.marked_img, text=f"X = {x}, Image: {param.loaded_image_index} / {param.max_image_index}", org=(100, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255), thickness=3, lineType=cv2.LINE_AA)

        elif event == cv2.EVENT_LBUTTONDOWN:
            param.marked_img = param.original_img.copy()
            left_part = param.marked_img[0:param.marked_img.shape[0], 0:x]
            right_part = param.marked_img[0:param.marked_img.shape[0], x:param.marked_img.shape[1]]
            param.scrolled_img = np.concatenate((right_part, left_part), axis=1)    
            cv2.imshow(param.previewWindowName, param.scrolled_img)

        elif event == cv2.EVENT_RBUTTONDOWN:
            param.original_img_path = param.images[param.loaded_image_index]
            updated_file_path = param.original_img_path.replace("input", "output")
            os.makedirs(os.path.dirname(updated_file_path), exist_ok=True)
            cv2.imwrite(updated_file_path, param.scrolled_img)

            if param.loaded_image_index >= param.max_image_index:
                param.processing = False
            else:
                param.loaded_image_index = param.loaded_image_index + 1
                param.original_img = cv2.imread(param.images[param.loaded_image_index])
                param.marked_img = param.original_img.copy()
                param.scrolled_img = param.original_img.copy()
                cv2.imshow(param.previewWindowName, param.scrolled_img)

def main():
    print("Starting PanoScroller!")

    args: PanoScrollerArgs = loadArgs() 
    images: list[str] = loadImages(args.inputPath, args.imageFormats)
        
    processParams = SplitProgressMonitor(images,
                                         args.previewWindowName, 
                                         0, 
                                         len(images) - 1, 
                                         cv2.imread(images[0]), 
                                         cv2.imread(images[0]), 
                                         cv2.imread(images[0]), 
                                         0, 
                                         3, 
                                         True)

    initializeWindows(processParams, args.mainWindowName, args.previewWindowName)

    while (processParams.processing):
        cv2.imshow(args.mainWindowName, processParams.marked_img)
        cv2.imshow(args.previewWindowName, processParams.scrolled_img)
        
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break

    print("Ending PanoScroller!")

if __name__ == '__main___':
    main()
else:
    main()