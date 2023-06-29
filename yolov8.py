import torch
from ultralytics import YOLO
from ultralytics.yolo.utils.ops import preprocess_results
from ultralytics.yolo.utils.visualization import draw_detections
import numpy as np
import cv2


class YOLOv8:
    def __init__(self, model_file, weights_file, min_score=0.7):
        self.model_file = model_file
        self.weights_file = weights_file
        self.min_score = min_score

        self.model = YOLO(self.model_file)
        weights = torch.load(self.weights_file)['model']
        self.model.model.load(weights)

    def run(self, image):
        if isinstance(image, str):
            image = cv2.imread(image)
        results = self.model(image, save=False, show=False, verbose=False)
        height, width = image.shape[:2]
        scores, classes_ids, boxes, masks = preprocess_results(results, (height, width))

        selected = (scores >= self.min_score)
        scores = scores[selected]
        classes_ids = classes_ids[selected]
        boxes = boxes[selected]
        masks = masks[selected]
        return scores, classes_ids, boxes, masks

    @staticmethod
    def draw_detections(image, scores, classes_ids, boxes, masks, palette=((0, 0, 255),)):
        draw_detections(image, scores, classes_ids, boxes, masks,
            palette=((0, 0, 255),), min_score=0)