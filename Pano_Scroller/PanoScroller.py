import cv2
import numpy as np
import glob
import os
import argparse
import PanoScrollerArgs

original_img = None
marked_img = None
last_known_x = 0
loaded_image_index = 0
original_img_path : str = ""
processing = True
scrolled_img = None
line_thickness = 3
images = []
max_image_index = 0
previewWindow : str = ""

def loadArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="manual")
    parser.add_argument("--inputPath", default="input")
    parser.add_argument("--mainWindowName", default="Source image")
    parser.add_argument("--previewWindowName", default="Preview")
    parser.add_argument("--imageFormats", nargs='+', default=[".jpg", ".png"])

    return parser.parse_args()

def split_image(event, x, y, flags, param):
        global original_img, marked_img, last_known_x, loaded_image_index, original_img_path, processing, scrolled_img, line_thickness, images, max_image_index, previewWindow

        if event == cv2.EVENT_MOUSEMOVE:
            marked_img[:, last_known_x-line_thickness:last_known_x+line_thickness] = original_img[:, last_known_x-line_thickness:last_known_x+line_thickness]
            last_known_x = x
            cv2.line(marked_img, (x, 0), (x, marked_img.shape[0]), (0, 0, 255), line_thickness)

        elif event == cv2.EVENT_LBUTTONDOWN:
            marked_img = original_img.copy()
            left_part = marked_img[0:marked_img.shape[0], 0:x]
            right_part = marked_img[0:marked_img.shape[0], x:marked_img.shape[1]]
            scrolled_img = np.concatenate((right_part, left_part), axis=1)
            cv2.namedWindow(previewWindow, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(previewWindow, 1800, 900)
            cv2.imshow(previewWindow, scrolled_img)

        elif event == cv2.EVENT_RBUTTONDOWN:
            original_img_path = images[loaded_image_index]
            updated_file_path = original_img_path.replace("input", "output")
            os.makedirs(os.path.dirname(updated_file_path), exist_ok=True)
            cv2.imwrite(updated_file_path, scrolled_img)

            if loaded_image_index >= max_image_index:
                processing = False
            else:
                loaded_image_index = loaded_image_index + 1
                original_img = cv2.imread(images[loaded_image_index])
                marked_img = original_img.copy()
                scrolled_img = original_img.copy()
                cv2.imshow(previewWindow, scrolled_img)

def main():
    global original_img, marked_img, last_known_x, loaded_image_index, original_img_path, processing, scrolled_img, line_thickness, images, max_image_index, previewWindow

    print("Starting PanoScroller!")
    print("\nLoading program parameters...")
    cmdArgs = loadArgs()
    args: PanoScrollerArgs.PanoScrollerArgs = PanoScrollerArgs.PanoScrollerArgs(cmdArgs.mode, cmdArgs.inputPath, cmdArgs.mainWindowName, cmdArgs.previewWindowName, cmdArgs.imageFormats)
    previewWindow = args.previewWindow
    print("Parameters loaded, listing:")
    print(', '.join("%s: %s" % item for item in vars(args).items()))

    print("\nLoading input files...")
    print(f"Searched path: {args.inputPath}")
    images = []
    for root, dirs, files in os.walk(args.inputPath):
        for file in files:
            if file.lower().endswith(tuple(args.imageFormats)):
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

    loaded_image_index = 0
    max_image_index = len(images) - 1

    original_img = cv2.imread(images[loaded_image_index])
    original_img_path = ""
    marked_img = original_img.copy()
    scrolled_img = original_img.copy()
    last_known_x = 0
    line_thickness = 3
    processing = True

    cv2.namedWindow(args.mainWindow, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(args.mainWindow, 1800, 900)
    cv2.setMouseCallback(args.mainWindow, split_image)

    while (processing):
        cv2.imshow(args.mainWindow, marked_img)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break

if __name__ == '__main___':
    main()
else:
    main()