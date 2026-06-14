#!/usr/bin/env python3

import os
import random
import threading
import time

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class DataCollectorNode(Node):
    def __init__(self):
        super().__init__("dataset_collector")

        self.subscription = self.create_subscription(
            Image,
            "camera/color/image_raw",
            self.image_callback,
            10,
        )
        self.bridge = CvBridge()
        self.latest_image = None

        self.save_dir = "yolo_dataset/images"
        os.makedirs(self.save_dir, exist_ok=True)
        self.image_counter = 0

        self.get_logger().info("Starting data collection thread...")
        self.collection_thread = threading.Thread(target=self.collect_data_loop)
        self.collection_thread.start()

    def image_callback(self, msg):

        self.latest_image = msg

    def collect_data_loop(self):
        BASE_X = 0.0
        BASE_Y = 0.62
        BASE_Z = 1.2
        MODEL_PATH = "/ws/install/swiz_binpicking/share/swiz_binpicking/worlds/ur/models/drc_practice_blue_cylinder/model.sdf"

        time.sleep(2)

        TOTAL_IMAGES = 100
        for iteration in range(TOTAL_IMAGES):
            self.get_logger().info(f"--- Iteration {iteration + 1}/{TOTAL_IMAGES} ---")

            num_cylinders = random.randint(3, 20)
            spawned_names = []

            self.get_logger().info(f"Spawning {num_cylinders} cylinders...")
            for i in range(num_cylinders):
                x = BASE_X + random.uniform(-0.1, 0.1)
                y = BASE_Y + random.uniform(-0.1, 0.1)
                z = BASE_Z + (i * 0.15)
                r = random.uniform(0, 3.14)
                p = random.uniform(0, 3.14)
                y_rot = random.uniform(0, 3.14)

                name = f"blue_cylinder_iter{iteration}_{i}"
                spawned_names.append(name)

                req_string = (
                    f"sdf_filename: '{MODEL_PATH}', "
                    f"name: '{name}', "
                    f"pose: {{ "
                    f"  position: {{ x: {x:.3f}, y: {y:.3f}, z: {z:.3f} }}, "
                    f"  orientation: {{ x: {r:.3f}, y: {p:.3f}, z: {y_rot:.3f} }} "
                    f"}}"
                )

                cmd = f'gz service -s /world/default/create --reqtype gz.msgs.EntityFactory --reptype gz.msgs.Boolean --timeout 300 --req "{req_string}"'
                os.system(cmd)
                time.sleep(0.05)

            self.get_logger().info("Waiting 5 seconds for cylinders to settle...")
            time.sleep(5.0)

            if self.latest_image is not None:
                try:
                    cv_image = self.bridge.imgmsg_to_cv2(
                        self.latest_image, desired_encoding="bgr8"
                    )
                    img_name = os.path.join(
                        self.save_dir, f"frame_{self.image_counter:04d}.jpg"
                    )

                    cv2.imwrite(img_name, cv_image)
                    self.get_logger().info(f"Saved {img_name} successfully!")
                    self.image_counter += 1
                except Exception as e:
                    self.get_logger().error(f"Failed to process/save image: {e}")
            else:
                self.get_logger().warning(
                    "No image received yet! Is the camera topic correct?"
                )

            self.get_logger().info("Removing cylinders for next iteration...")
            for name in spawned_names:
                del_req = f"name: '{name}' type: MODEL"
                del_cmd = f'gz service -s /world/default/remove --reqtype gz.msgs.Entity --reptype gz.msgs.Boolean --timeout 300 --req "{del_req}"'
                os.system(del_cmd)
                time.sleep(0.05)

            time.sleep(1.0)

        self.get_logger().info("Data collection complete! You can kill the node now.")


def main(args=None):
    rclpy.init(args=args)
    node = DataCollectorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down data collector...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
