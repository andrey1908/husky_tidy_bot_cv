import rospy
import rostopic
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch
from ultralytics import YOLO
from ultralytics.yolo.utils.ops import preprocess_results
import numpy as np
import cv2


class YOLOv8_segmentation:
    def __init__(self, model_file, weights_file, image_topic, out_segmentation_topic,
            min_score=0.7, colorful=False, palette=((0, 0, 255),)):
        self.model_file = model_file
        self.weights_file = weights_file
        self.image_topic = image_topic
        self.out_segmentation_topic = out_segmentation_topic
        self.min_score = min_score
        self.colorful = colorful
        self.palette = palette

        self.model = YOLO(self.model_file)
        weights = torch.load(self.weights_file)['model']
        self.model.model.load(weights)

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

        results = self.model(image, save=False, show=False, verbose=False)
        height, width = image.shape[:2]
        scores, classes_ids, masks = preprocess_results(results, (height, width))

        if self.colorful:
            segmentation = np.zeros_like(image)
        else:
            segmentation = np.zeros((height, width), dtype=np.uint16)
        for i, (score, class_id, mask) in enumerate(zip(scores, classes_ids, masks)):
            if score < self.min_score:
                continue
            if self.colorful:
                obj = np.array(self.palette[i % len(self.palette)], dtype=np.uint8)
            else:
                obj = (class_id << 8) + i + 1
            segmentation[mask != 0] = obj

        segmentation_msg = self.bridge.cv2_to_imgmsg(segmentation, encoding='passthrough')
        segmentation_msg.header = image_msg.header
        self.segmentation_pub.publish(segmentation_msg)


if __name__ == "__main__":
    rospy.init_node("segmentation")
    segmentation_node = YOLOv8_segmentation(
        "/home/cds-jetson-host/ultralytics/yolov8n-seg-1class.yaml",
        "/home/cds-jetson-host/ultralytics/runs/segment/train2/weights/last.pt",
        "/camera/compressed", "/segmentation", colorful=False)
    print("Spinning...")
    rospy.spin()
