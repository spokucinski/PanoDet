def convert_to_albumentations_yolo_format(bboxes):
    yolo_bboxes = []
    
    for bbox in bboxes:
        x_min, y_min, width, height = bbox
        x_center = x_min + (width / 2)
        y_center = y_min + (height / 2)

        yolo_bboxes.append([x_center, y_center, width, height])

    return yolo_bboxes

def convert_to_fiftyone_yolo_format(bbox):
    x_center, y_center, width, height = bbox
    x_min = x_center - (width / 2)
    y_min = y_center - (height / 2)

    return [x_min, y_min, width, height]