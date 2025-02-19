from external.ultralytics import RTDETR

model = RTDETR('results/SPHERE_CODE55/Train/ALL_250_1792_3_rtdetr-l_SGD/weights/best.pt')

model.info()

results = model.val(split='test', 
                    imgsz=1792, 
                    data='datasets/ALL/dataset.yaml', 
                    name='ALL_250_1792_3_rtdetr-l_SGD', 
                    project='results/SPHERE_CODE55/Test',
                    save_json=True,
                    batch=1,
                    rect=True,
                    plots=True)