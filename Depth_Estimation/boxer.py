import cv2
import os

# ======= KONFIGURACJA =======
input_folder = "D:\\Repos\\PanoDet\\Depth_Estimation\\Depth-Anything-V2\\assets\\C3\\images"   # <- <-- ZMIEŃ
output_folder = "D:\\Repos\\PanoDet\\Depth_Estimation\\Depth-Anything-V2\\assets\\C3\\images"     # <- <-- ZMIEŃ
max_display_width = 1920
max_display_height = 1080
# ============================

os.makedirs(output_folder, exist_ok=True)
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

pt1, pt2 = (-1, -1), (-1, -1)
drawing = False
finished = False

def draw_box(event, x, y, flags, param):
    global pt1, pt2, drawing, finished
    if event == cv2.EVENT_LBUTTONDOWN:
        pt1 = (x, y)
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        pt2 = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        pt2 = (x, y)
        drawing = False
        finished = True

for image_name in image_files:
    image_path = os.path.join(input_folder, image_name)
    original_img = cv2.imread(image_path)
    orig_h, orig_w = original_img.shape[:2]

    # Oblicz skalę do resize
    scale = min(max_display_width / orig_w, max_display_height / orig_h, 1.0)
    resized_w, resized_h = int(orig_w * scale), int(orig_h * scale)
    img = cv2.resize(original_img, (resized_w, resized_h))

    pt1, pt2 = (-1, -1), (-1, -1)
    finished = False

    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", draw_box)

    while True:
        display = img.copy()
        if drawing and pt1 != (-1, -1):
            cv2.rectangle(display, pt1, pt2, (0, 255, 0), 2)
        cv2.imshow("Image", display)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or finished:  # ESC or mouse finished
            break

    if pt1 != (-1, -1) and pt2 != (-1, -1):
        # Przelicz współrzędne do oryginalnej rozdzielczości
        x1, y1 = int(pt1[0] / scale), int(pt1[1] / scale)
        x2, y2 = int(pt2[0] / scale), int(pt2[1] / scale)

        output_txt = os.path.join(output_folder, os.path.splitext(image_name)[0] + ".txt")
        with open(output_txt, "w") as f:
            f.write(f"{x1},{y1},{x2},{y2}\n")

cv2.destroyAllWindows()
