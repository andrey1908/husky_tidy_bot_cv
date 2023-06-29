import cv2
import numpy as np
from yolox.tracker.byte_tracker import BYTETracker


class ByteTrack:
    class BYTETracker_args:
        def __init__(self, track_thresh, track_buffer, match_thresh, mot20):
            self.track_thresh = track_thresh
            self.track_buffer = track_buffer
            self.match_thresh = match_thresh
            self.mot20 = mot20

    def __init__(self, frame_rate=10, track_thresh=0.1, track_buffer=10, match_thresh=0.9):
        self.frame_rate = frame_rate
        mot20 = False
        self.args = ByteTrack.BYTETracker_args(track_thresh, track_buffer, match_thresh, mot20)
        self.trackers = dict()

    def track(self, boxes, scores, classes_ids, masks=None):
        unique_classes_ids = np.unique(classes_ids)
        tracked_objects = list()
        for class_id in unique_classes_ids:
            if class_id not in self.trackers:
                self.trackers[class_id] = BYTETracker(self.args, self.frame_rate)
            tracker = self.trackers[class_id]

            selected = (classes_ids == class_id)
            selected_boxes = boxes[selected]
            selected_scores = scores[selected]
            selected_boxes_scores = \
                np.hstack((selected_boxes, selected_scores[..., np.newaxis]))
            tracked_objects_single = tracker.update(selected_boxes_scores,
                (1, 1), (1, 1), masks=masks)
            tracked_objects_single = \
                zip(tracked_objects_single, [class_id] * len(tracked_objects_single))
            tracked_objects.extend(tracked_objects_single)
        return tracked_objects
    
    @staticmethod
    def draw_tracked_objects(image, tracked_objects, draw_boxes=False,
            palette=((0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255), (0, 255, 255))):
        for tracked_object, class_id in tracked_objects:
            color = palette[tracked_object.track_id % len(palette)]
            mask = tracked_object.mask
            image[mask != 0] = color

            if draw_boxes:
                x1, y1, x2, y2 = list(map(int, tracked_object.tlbr))
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
