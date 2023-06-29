import rospy
import rostopic
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
from byte_track import ByteTrack


class ByteTrack_node(ByteTrack):
    def __init__(self, segmentation_topic, out_tracking_topic, default_score=0.9,
            frame_rate=10, track_thresh=0.1, track_buffer=10, match_thresh=0.9,
            visualization=False,
            palette=((0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255), (0, 255, 255))):
        super().__init__(frame_rate, track_thresh, track_buffer, match_thresh)

        self.segmentation_topic = segmentation_topic
        self.out_tracking_topic = out_tracking_topic
        self.default_score = default_score
        self.visualization = visualization
        self.palette = palette

        self.tracking_pub = rospy.Publisher(self.out_tracking_topic, Image, queue_size=10)
        topic_type, _, _ = rostopic.get_topic_class(self.segmentation_topic)
        rospy.Subscriber(self.segmentation_topic, topic_type, self.callback)

        self.bridge = CvBridge()

    def callback(self, segmentation_msg):
        if segmentation_msg._type == "sensor_msgs/Image":
            segmentation_image = self.bridge.imgmsg_to_cv2(segmentation_msg, desired_encoding='passthrough')
        elif segmentation_msg._type == "sensor_msgs/CompressedImage":
            segmentation_image = self.bridge.compressed_imgmsg_to_cv2(segmentation_msg, desired_encoding='passthrough')
        else:
            raise RuntimeError("Unkown message type")

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
            score = self.default_score
            class_id = (obj >> 8) & 0xFF

            masks.append(mask.astype(np.uint8))
            boxes.append(box)
            scores.append(score)
            classes_ids.append(class_id)
        masks = np.array(masks)
        boxes = np.array(boxes)
        scores = np.array(scores)

        tracked_objects = self.track(boxes, scores, classes_ids, masks=masks)

        if self.visualization:
            tracking_image = np.zeros((*segmentation_image.shape, 3), dtype=np.uint8)
            ByteTrack.draw_tracked_objects(tracking_image, tracked_objects,
                palette=self.palette)
        else:
            tracking_image = np.zeros_like(segmentation_image)
            for tracked_object, class_id in tracked_objects:
                obj = (class_id << 8) + tracked_object.track_id + 1
                mask = tracked_object.mask
                tracking_image[mask != 0] = obj

        tracking_msg = self.bridge.cv2_to_imgmsg(tracking_image, encoding='passthrough')
        tracking_msg.header = segmentation_msg.header
        self.tracking_pub.publish(tracking_msg)


if __name__ == "__main__":
    rospy.init_node("tracking")
    tracking_node = ByteTrack_node("/segmentation", "/tracking",
        visualization=False)
    print("Spinning...")
    rospy.spin()
