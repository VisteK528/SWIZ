import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from ultralytics import YOLO

image_qos = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


class SegmentationNode(Node):
    def __init__(self):
        super().__init__("swiz_seg_node")

        self.bridge = CvBridge()
        self.get_logger().info("Loading YOLO model...")
        self.model = YOLO("custom-yolo26x-seg.pt")
        self.get_logger().info("Model loaded successfully!")

        self._sub = self.create_subscription(
            Image, "camera/color/image_raw", self.image_callback, image_qos
        )
        self._pub = self.create_publisher(Image, "segmentation_output", image_qos)
        self._masks_pub = self.create_publisher(
            Image, "segmentation_output_masks", image_qos
        )

    def _image_callback(self, msg: Image):

        self._pub.publish(msg)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            results = self.model(cv_image, retina_masks=True, verbose=False, conf=0.10)
            result = results[0]

            height, width = cv_image.shape[:2]
            colored_instance_mask = np.zeros((height, width, 3), dtype=np.uint8)

            if result.masks is not None:
                masks_array = result.masks.data.cpu().numpy()
                num_instances = masks_array.shape[0]

                for i in range(num_instances):
                    binary_mask = masks_array[i] > 0.5
                    random_color = [int(x) for x in np.random.randint(50, 255, size=3)]

                    colored_instance_mask[binary_mask] = random_color

            ros_mask_msg = self.bridge.cv2_to_imgmsg(
                colored_instance_mask, encoding="bgr8"
            )
            ros_mask_msg.header = msg.header
            self._masks_pub.publish(ros_mask_msg)

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = SegmentationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
