from ultralytics import YOLO

model = YOLO("results/FullCODE55/Train/ScrolledFullCode_500_1920_3_yolo11x_auto_default/weights/best.pt")

results = model.predict(
    source="datasets/Rotated",       # może być lista plików lub kamera (0)
    imgsz=1920,
    conf=0.25,
    iou=0.7,
    device="cuda:0",
    save=True,                  # zapisuje obrazy/wideo
    save_txt=True,              # zapisuje .txt
    save_conf=True,             # dodaje confidence do .txt
    save_json=True,             # COCO-style JSON
    project="results/predict",
    name="rotated"
)