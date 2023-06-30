import cv2
import numpy as np
from yolox.tracker.byte_tracker import BYTETracker
from conversions import from_segmentation_image, to_tracking_image


class ByteTrack:
    def __init__(self, track_thresh=0.1, track_buffer=30, match_thresh=0.9, frame_rate=30):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate

        self.trackers = dict()

    def track(self, boxes, scores, classes_ids, masks=None):
        unique_classes_ids = np.unique(classes_ids)
        tracked_objects = list()
        for class_id in unique_classes_ids:
            if class_id not in self.trackers:
                self.trackers[class_id] = BYTETracker(track_thresh=self.track_thresh,
                    track_buffer=self.track_buffer, match_thresh=self.match_thresh,
                    mot20=False, frame_rate=self.frame_rate)
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

    @staticmethod
    def from_segmentation_image(segmentation_image, default_score=0.9):
        classes_ids, scores, boxes, masks = \
            from_segmentation_image(segmentation_image, default_score=default_score)
        return boxes, scores, classes_ids, masks

    @staticmethod
    def to_tracking_image(tracked_objects, out_shape):
        if len(tracked_objects) == 0:
            classes_ids = list()
            tracking_ids = list()
            masks = np.empty((0, *out_shape))
        else:
            classes_ids, tracking_ids, masks = \
                zip(*[(class_id, tracked_object.track_id, tracked_object.mask)
                    for tracked_object, class_id in tracked_objects])
            masks = np.array(masks)
            assert tuple(out_shape) == masks.shape[1:]
        tracking_image = to_tracking_image(classes_ids, tracking_ids, masks)
        return tracking_image
