import rospy
import rostopic
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
from yolov8 import YOLOv8
from conversions import to_segmentation_image


class YOLOv8_node(YOLOv8):
    def __init__(self, model_file, weights_file, image_topic, out_segmentation_topic,
            min_score=0.7, visualization=False, palette=((0, 0, 255),)):
        super().__init__(model_file, weights_file, min_score=min_score)

        self.image_topic = image_topic
        self.out_segmentation_topic = out_segmentation_topic
        self.visualization = visualization
        self.palette = palette

        self.segmentation_pub = rospy.Publisher(self.out_segmentation_topic, Image, queue_size=10)
        topic_type, _, _ = rostopic.get_topic_class(self.image_topic)
        rospy.Subscriber(self.image_topic, topic_type, self.callback)

        self.bridge = CvBridge()

    def callback(self, image_msg):
        if image_msg._type == "sensor_msgs/Image":
            image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')
        elif image_msg._type == "sensor_msgs/CompressedImage":
            image = self.bridge.compressed_imgmsg_to_cv2(image_msg, desired_encoding='passthrough')
        else:
            raise RuntimeError("Unkown message type")

        scores, classes_ids, boxes, masks = self.segment(image)

        if self.visualization:
            YOLOv8.draw_segmentation(image, scores, classes_ids, boxes, masks, palette=self.palette)
            segmentation_image = image
        else:
            segmentation_image = YOLOv8.to_segmentation_image(classes_ids, masks)

        segmentation_msg = self.bridge.cv2_to_imgmsg(segmentation_image, encoding='passthrough')
        segmentation_msg.header = image_msg.header
        self.segmentation_pub.publish(segmentation_msg)


if __name__ == "__main__":
    rospy.init_node("segmentation")
    segmentation_node = YOLOv8_node(
        "/home/cds-jetson-host/ultralytics/yolov8n-seg-1class.yaml",
        "/home/cds-jetson-host/ultralytics/runs/segment/train2/weights/last.pt",
        "/camera/compressed", "/segmentation", visualization=False)
    print("Spinning...")
    rospy.spin()
