import json
import cv2
import numpy as np

import sys
sys.path.append('external/SphericalObjectDetection/PRDA')

from external.SphericalObjectDetection.PRDA import PRDA
img = cv2.imread('data/input/PANDORA/images/000001.jpg')
img_size = img.shape
erp_w, erp_h = img_size[1], img_size[0]

with open('data/input/PANDORA/annotations/test.json') as f:
    res = json.load(f)
    annotations = res["annotations"]
    cor_ann = []
    for ann in annotations:
        if ann["image_id"] == 1:
            cor_ann.append(ann)

    for ann in cor_ann:
        PRDA.visualization(img, np.array([ann["bbox"]]), erp_w=erp_w, erp_h=erp_h)
        cv2.imwrite('data/output/test1.jpg', img)
gt = np.array([[-2.9321531433504737, -0.18171150631480926, 7, 8, 0]])


# Visualization results
PRDA.visualization(img, gt, erp_w=erp_w, erp_h=erp_h)

cv2.imwrite('data/output/test1upd2.jpg', img)