import numpy as np
import cv2


def to_segmentation_image(classes_ids, masks):
    height, width = masks.shape[1:]
    segmentation_image = np.zeros((height, width), dtype=np.uint16)
    for i, (class_id, mask) in enumerate(zip(classes_ids, masks)):
        obj = (class_id << 8) + i + 1
        segmentation_image[mask != 0] = obj
    return segmentation_image


def from_segmentation_image(segmentation_image, default_score=0.9):
    objects = np.unique(segmentation_image)
    objects = objects[objects != 0]
    masks = list()
    boxes = list()
    scores = list()
    classes_ids = list()
    for obj in objects:
        mask = (segmentation_image == obj)
        indices = np.where(mask)
        min_x = indices[1].min()
        min_y = indices[0].min()
        max_x = indices[1].max()
        max_y = indices[0].max()
        box = np.array([min_x, min_y, max_x, max_y], dtype=int)
        score = default_score
        class_id = (obj >> 8) & 0xFF

        masks.append(mask.astype(np.uint8))
        boxes.append(box)
        scores.append(score)
        classes_ids.append(class_id)
    masks = np.array(masks)
    boxes = np.array(boxes)
    scores = np.array(scores)
    classes_ids = np.array(classes_ids)
    return classes_ids, scores, boxes, masks


def to_tracking_image(classes_ids, tracking_ids, masks):
    height, width = masks.shape[1:]
    tracking_image = np.zeros((height, width), dtype=np.uint16)
    for class_id, tracking_id, mask in zip(classes_ids, tracking_ids, masks):
        obj = (class_id << 8) + tracking_id + 1
        tracking_image[mask != 0] = obj
    return tracking_image
