import re
import subprocess
import threading
import traceback

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


class GlobalPoseBypass(Node):
    def __init__(self):
        super().__init__("global_pose_bypass")

        self.tf_broadcaster = TransformBroadcaster(self)

        self.gz_process = subprocess.Popen(
            ["gz", "topic", "-e", "-t", "/world/default/pose/info"],
            stdout=subprocess.PIPE,
            text=True,
        )

        self.parse_thread = threading.Thread(target=self.parse_stream)
        self.parse_thread.daemon = True
        self.parse_thread.start()

    def parse_stream(self):
        try:
            self._parse_stream_impl()
        except Exception as e:
            self.get_logger().error(
                f"Parser thread crashed: {e}\n{traceback.format_exc()}"
            )

    def _parse_stream_impl(self):
        in_pose = False
        in_position = False
        in_orientation = False

        current_name = ""
        current_pos = {"x": 0.0, "y": 0.0, "z": 0.0}
        current_ori = {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}

        for line in iter(self.gz_process.stdout.readline, ""):
            line = line.strip()

            if not line:
                continue

            if line == "pose {":
                in_pose = True
                in_position = False
                in_orientation = False
                current_name = ""
                current_pos = {"x": 0.0, "y": 0.0, "z": 0.0}
                current_ori = {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
                continue

            if not in_pose:
                continue

            if line.startswith("name:"):
                name_match = re.search(r'name:\s*"([^"]+)"', line)
                if name_match:
                    current_name = name_match.group(1)
                continue

            if line == "position {":
                in_position = True
                continue

            if line == "orientation {":
                in_orientation = True
                continue

            if line == "}":
                if in_position:
                    in_position = False
                elif in_orientation:
                    in_orientation = False
                else:
                    in_pose = False

                    if "blue_cylinder" in current_name:
                        self.publish_tf(current_name, current_pos, current_ori)
                        self.get_logger().info(f"Published TF -> {current_name}")

                    current_name = ""
                continue

            if in_position or in_orientation:
                if (
                    line.startswith("x:")
                    or line.startswith("y:")
                    or line.startswith("z:")
                    or line.startswith("w:")
                ):
                    parts = line.split(":")
                    if len(parts) == 2:
                        axis = parts[0].strip()
                        try:
                            value = float(parts[1].strip())
                            if in_position and axis in current_pos:
                                current_pos[axis] = value
                            elif in_orientation and axis in current_ori:
                                current_ori[axis] = value
                        except ValueError:
                            pass

    def publish_tf(self, child_name, pos, ori):
        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "world"
        t.child_frame_id = child_name

        t.transform.translation.x = pos["x"]
        t.transform.translation.y = pos["y"]
        t.transform.translation.z = pos["z"]

        t.transform.rotation.x = ori["x"]
        t.transform.rotation.y = ori["y"]
        t.transform.rotation.z = ori["z"]
        t.transform.rotation.w = ori["w"]

        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = GlobalPoseBypass()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
