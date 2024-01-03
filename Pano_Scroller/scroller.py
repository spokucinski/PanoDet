import cv2
import numpy as np
import glob
import os

MAIN_WIN = "main"
RES_WIN = "result"
INPUT_PATH = "input"

res = []
for root, dirs, files in os.walk(INPUT_PATH):
    for file in files:
        if file.lower().endswith(".jpg") or file.lower().endswith(".png"):
            res.append(os.path.join(root, file))

loaded_image_index = 0
max_image_index = len(res)

original_img = cv2.imread(res[loaded_image_index])
original_img_path : str = ""
marked_img = original_img.copy()
scrolled_img = original_img.copy()
last_known_x = 0
line_thickness = 3
processing = True

def split_image(event, x, y, flags, param):
    global original_img, marked_img, last_known_x, loaded_image_index, original_img_path, processing, scrolled_img

    if event == cv2.EVENT_MOUSEMOVE:
        marked_img[:, last_known_x-line_thickness:last_known_x+line_thickness] = original_img[:, last_known_x-line_thickness:last_known_x+line_thickness]
        last_known_x = x
        cv2.line(marked_img, (x, 0), (x, marked_img.shape[0]), (0, 0, 255), line_thickness)

    elif event == cv2.EVENT_LBUTTONDOWN:
        marked_img = original_img.copy()
        left_part = marked_img[0:marked_img.shape[0], 0:x]
        right_part = marked_img[0:marked_img.shape[0], x:marked_img.shape[1]]
        scrolled_img = np.concatenate((right_part, left_part), axis=1)
        cv2.namedWindow(RES_WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(RES_WIN, 1800, 900)
        cv2.imshow(RES_WIN, scrolled_img)

    elif event == cv2.EVENT_RBUTTONDOWN:
        original_img_path = res[loaded_image_index]
        updated_file_path = original_img_path.replace("input", "output")
        os.makedirs(os.path.dirname(updated_file_path), exist_ok=True)
        cv2.imwrite(updated_file_path, scrolled_img)

        if loaded_image_index >= max_image_index:
            processing = False
        else:
            loaded_image_index = loaded_image_index + 1
            original_img = cv2.imread(res[loaded_image_index])
            marked_img = original_img.copy()
            scrolled_img = original_img.copy()
            cv2.imshow(RES_WIN, scrolled_img)

cv2.namedWindow(MAIN_WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(MAIN_WIN, 1800, 900)
cv2.setMouseCallback(MAIN_WIN, split_image)

while (processing):
    cv2.imshow(MAIN_WIN, marked_img)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
