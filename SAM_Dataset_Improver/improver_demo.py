import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('external/SphericalObjectDetection/PRDA')
sys.path.append('external/SAM')

from external.SphericalObjectDetection.PRDA import PRDA
from segment_anything import SamPredictor, sam_model_registry

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

img = cv2.imread('data/input/PANDORA/images/000001.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_size = img.shape
erp_w, erp_h = img_size[1], img_size[0]

SAM_CHECKPOINT_PATH = "models/sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"
DEVICE = "cuda"

sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH)
sam = sam.to(device=DEVICE)
predictor = SamPredictor(sam)
predictor.set_image(img)

with open('data/input/PANDORA/annotations/test.json') as testAnnotationsFile:
    allAnnotationsJson = json.load(testAnnotationsFile)
    allTestAnnotations = allAnnotationsJson["annotations"]
    firstImageAnnotations = list(filter(lambda annotation: annotation["image_id"] == 1, allTestAnnotations))

    for ann in firstImageAnnotations:
        minX, minY, maxX, maxY = PRDA.getBBoxCoords(np.array([ann["bbox"]]), erp_w=erp_w, erp_h=erp_h)
        input_box = np.array([int(minX), int(minY), int(maxX), int(maxY)])
        masks, _, _ = predictor.predict(point_coords=None,
                                        point_labels=None,
                                        box=input_box[None, :],
                                        multimask_output=False)
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        show_mask(masks[0], plt.gca())
        show_box(input_box, plt.gca())
        plt.axis('off')
        plt.show()
        plt.savefig('data/output/test1.jpg')