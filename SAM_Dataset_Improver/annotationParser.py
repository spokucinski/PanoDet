import json
import cv2
import numpy as np


import sys
sys.path.append('external/SphericalObjectDetection/PRDA')

from lib.tools import ro_Shpbbox
from lib.ImageRecorder import ImageRecorder

file_paths = ['data/input/PANDORA/annotations/test.json',
              'data/input/PANDORA/annotations/train.json']

for file_path in file_paths:
    with open(file_path) as f:
        erp_w = 1920
        erp_h = 960
        res = json.load(f)
        annotations = res["annotations"]

        for ann in annotations:
            print(f'Processing annotation with ID: {ann["id"]}')
            new_gt = np.array([ann["bbox"]])
            BFoV = ImageRecorder(erp_w, erp_h, view_angle_w=new_gt[0][2], view_angle_h=new_gt[0][3],
                                 long_side=erp_w)
            Px, Py = BFoV._sample_points(new_gt[0][0], new_gt[0][1], border_only=True)
            Px, Py = ro_Shpbbox(new_gt, Px, Py, erp_w=erp_w, erp_h=erp_h)

            parsedPoints = []
            for i in range(len(Px)):
                parsedPoints.append([int(Px[i]), int(Py[i])])

            ctr = np.array(parsedPoints).reshape((-1, 1, 2)).astype(np.int32)

            x, y, w, h = cv2.boundingRect(ctr)

            ann["bbox"] = [x, y, w, h]

        output_file_path = file_path.split('.')[0] + 'Parsed.json'
        with open(output_file_path, "w") as outfile:
            json.dump(res, outfile)