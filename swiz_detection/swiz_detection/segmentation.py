import message_filters
import numpy as np
import open3d as o3d
import pyransac3d as pyr
import rclpy
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from ultralytics import YOLO
from visualization_msgs.msg import Marker, MarkerArray

image_qos = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


def create_cylinder_point_cloud(radius, height, total_points):
    area_side = 2 * np.pi * radius * height
    area_cap = np.pi * (radius**2)
    total_area = area_side + (2 * area_cap)

    n_side = int(total_points * (area_side / total_area))
    n_cap = int(total_points * (area_cap / total_area))

    theta_side = np.random.uniform(0, 2 * np.pi, n_side)
    z_side = np.random.uniform(0, height, n_side)
    x_side = radius * np.cos(theta_side)
    y_side = radius * np.sin(theta_side)

    def generate_cap(z_val):
        theta = np.random.uniform(0, 2 * np.pi, n_cap)
        r = radius * np.sqrt(np.random.uniform(0, 1, n_cap))
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.full(n_cap, z_val)
        return x, y, z

    x_bottom, y_bottom, z_bottom = generate_cap(0)
    x_top, y_top, z_top = generate_cap(height)

    x = np.concatenate([x_side, x_bottom, x_top])
    y = np.concatenate([y_side, y_bottom, y_top])
    z = np.concatenate([z_side, z_bottom, z_top])

    points = np.column_stack((x, y, z))

    points -= points.mean(axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


class SegmentationNode(Node):
    def __init__(self):
        super().__init__("swiz_seg_node")

        self.bridge = CvBridge()
        self.get_logger().info("Loading YOLO model...")
        self.model = YOLO("custom-yolo26x-seg.pt")
        self.get_logger().info("Model loaded successfully!")

        self._image_sub = message_filters.Subscriber(
            self, Image, "camera/color/image_raw"
        )

        self.create_subscription(
            CameraInfo, "camera/color/camera_info", self.info_cb, 10
        )

        self._pointcloud_sub = message_filters.Subscriber(
            self, PointCloud2, "camera/depth/points"
        )

        self._pub = self.create_publisher(Image, "segmentation_output", image_qos)
        self._masks_pub = self.create_publisher(
            Image, "segmentation_output_masks", image_qos
        )

        self._downsampled_cloud_pub = self.create_publisher(
            PointCloud2, "/downsampled_cloud", 10
        )

        self._segmented_point_cloud_pub = self.create_publisher(
            PointCloud2, "/segmented_cloud", 10
        )

        self._marker_pub = self.create_publisher(MarkerArray, "/cylinder_markers", 10)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self._image_sub, self._pointcloud_sub], queue_size=10, slop=0.1
        )
        self.get_logger().info("Starting common callback!")
        self.ts.registerCallback(self._sync_callback)

        self.fx, self.fy, self.cx, self.cy = None, None, None, None

        self.cylinder_radius = 0.02232
        self.cylinder_height = 0.06904
        self.cylinder_model = create_cylinder_point_cloud(0.02232, 0.06904, 2000)

    def info_cb(self, msg):
        self.fx, self.fy = msg.k[0], msg.k[4]
        self.cx, self.cy = msg.k[2], msg.k[5]

    def mask_pointcloud_with_2d(self, pcd, mask, fx, fy, cx, cy):
        points = np.asarray(pcd.points)

        # ROS robot frame -> camera optical frame
        x_cam = points[:, 1] * -1
        y_cam = points[:, 2] * -1
        z_cam = points[:, 0]

        valid_z = np.isfinite(z_cam) & (z_cam > 0.01) & (z_cam < 10.0)

        u = np.round((x_cam * fx / np.where(valid_z, z_cam, 1)) + cx).astype(int)
        v = np.round((y_cam * fy / np.where(valid_z, z_cam, 1)) + cy).astype(int)

        h, w = mask.shape[:2]
        in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h) & valid_z

        in_mask = np.zeros(len(points), dtype=bool)
        in_mask[in_bounds] = mask[v[in_bounds], u[in_bounds]] > 0

        return pcd.select_by_index(np.where(in_mask)[0])

    def run_ransac_cylinder(
        self, scene_pcd: o3d.geometry.PointCloud, stamp, frame_id: str, object_id: int
    ):
        if len(scene_pcd.points) < 50:
            return

        labels = np.array(
            scene_pcd.cluster_dbscan(eps=0.02, min_points=15, print_progress=False)
        )
        if len(labels) == 0 or labels.max() < 0:
            return

        valid_labels = labels[labels >= 0]
        if len(valid_labels) == 0:
            return

        largest_cluster_idx = np.argmax(np.bincount(valid_labels))
        inlier_indices = np.where(labels == largest_cluster_idx)[0]
        clean_pcd = scene_pcd.select_by_index(inlier_indices)
        clean_pcd, _ = clean_pcd.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=2.0
        )

        points = np.asarray(clean_pcd.points)
        if len(points) < 30:
            return

        cyl = pyr.Cylinder()
        ransac_center, axis, radius, inliers = cyl.fit(
            points, thresh=0.002, maxIteration=2000
        )

        if radius > self.cylinder_radius * 2.5:
            return

        if len(inliers) < 20:
            return

        inlier_points = points[inliers]
        projections = np.dot(inlier_points - ransac_center, axis)
        t_min, t_max = np.min(projections), np.max(projections)
        true_center = ransac_center + ((t_max + t_min) / 2.0) * axis

        v1 = np.array([0.0, 0.0, 1.0])
        v2 = np.array(axis)
        v2 = v2 / np.linalg.norm(v2)

        if np.allclose(v1, v2):
            rot_matrix = np.eye(3)
        elif np.allclose(v1, -v2):
            rot_matrix = np.diag([1, -1, -1])
        else:
            cross_prod = np.cross(v1, v2)
            cross_norm = np.linalg.norm(cross_prod)
            if cross_norm < 1e-6:
                rot_matrix = np.eye(3)
            else:
                axis_rot = cross_prod / cross_norm
                angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
                rot_matrix = Rotation.from_rotvec(angle * axis_rot).as_matrix()

        init_transform = np.eye(4)
        init_transform[:3, :3] = rot_matrix
        init_transform[:3, 3] = true_center

        icp_result = o3d.pipelines.registration.registration_icp(
            self.cylinder_model,
            scene_pcd,
            max_correspondence_distance=0.02,
            init=init_transform,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=300
            ),
        )

        T = icp_result.transformation
        final_translation = T[:3, 3]
        final_quat = Rotation.from_matrix(T[:3, :3].copy()).as_quat()

        return self._create_marker(
            object_id, frame_id, final_translation, final_quat, stamp
        )

    def _create_marker(
        self,
        object_id: int,
        frame_id: str,
        position: np.ndarray,
        quat: np.ndarray,
        stamp,
    ):
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = frame_id
        marker.ns = "cylinders"
        marker.id = object_id
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD

        marker.pose.position.x = float(position[0])
        marker.pose.position.y = float(position[1])
        marker.pose.position.z = float(position[2])
        marker.pose.orientation.x = float(quat[0])
        marker.pose.orientation.y = float(quat[1])
        marker.pose.orientation.z = float(quat[2])
        marker.pose.orientation.w = float(quat[3])

        marker.scale.x = self.cylinder_radius * 2
        marker.scale.y = self.cylinder_radius * 2
        marker.scale.z = self.cylinder_height

        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.7

        marker.lifetime.sec = 5

        return marker

    def _sync_callback(self, image_msg, pointcloud_msg):
        colored_instance_mask = None
        result = None
        track_ids = []

        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")

            results = self.model.track(
                cv_image, persist=True, retina_masks=True, verbose=False, conf=0.5
            )
            result = results[0]

            height, width = cv_image.shape[:2]
            colored_instance_mask = np.zeros((height, width, 3), dtype=np.uint8)

            if result.masks is not None and result.boxes.id is not None:
                masks_array = result.masks.data.cpu().numpy()

                track_ids = result.boxes.id.int().cpu().tolist()

                for i, track_id in enumerate(track_ids):
                    binary_mask = masks_array[i] > 0.5

                    np.random.seed(track_id)
                    consistent_color = [
                        int(x) for x in np.random.randint(50, 255, size=3)
                    ]

                    colored_instance_mask[binary_mask] = consistent_color

            ros_mask_msg = self.bridge.cv2_to_imgmsg(
                colored_instance_mask, encoding="bgr8"
            )
            ros_mask_msg.header = image_msg.header
            self._masks_pub.publish(ros_mask_msg)

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

        # Point cloud segmentation
        structured_array = pc2.read_points(
            pointcloud_msg, field_names=("x", "y", "z"), skip_nans=True
        )

        points = np.empty((structured_array.shape[0], 3), dtype=np.float32)

        points[:, 0] = structured_array["x"]
        points[:, 1] = structured_array["y"]
        points[:, 2] = structured_array["z"]

        if points.shape[0] == 0:
            return

        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(points)

        # downsampled_pcd = o3d_pcd.voxel_down_sample(voxel_size=0.00001)
        downsampled_pcd = o3d_pcd.voxel_down_sample(voxel_size=0.005)

        downsampled_points = np.asarray(downsampled_pcd.points, dtype=np.float32)
        out_msg = pc2.create_cloud_xyz32(pointcloud_msg.header, downsampled_points)
        self._downsampled_cloud_pub.publish(out_msg)

        if self.fx is not None and result is not None and result.masks is not None:
            masks_array = result.masks.data.cpu().numpy()

            marker_array = MarkerArray()

            for i in range(len(track_ids)):
                binary_mask = (masks_array[i] > 0.5).astype(np.uint8) * 255

                segmented_pcd = self.mask_pointcloud_with_2d(
                    downsampled_pcd, binary_mask, self.fx, self.fy, self.cx, self.cy
                )
                self.get_logger().info(
                    f"Segmented point cloud has {len(segmented_pcd.points)} points after masking."
                )
                segmented_points = np.asarray(segmented_pcd.points, dtype=np.float32)
                self._segmented_point_cloud_pub.publish(
                    pc2.create_cloud_xyz32(pointcloud_msg.header, segmented_points)
                )

                marker = self.run_ransac_cylinder(
                    segmented_pcd,
                    pointcloud_msg.header.stamp,
                    pointcloud_msg.header.frame_id,
                    object_id=track_ids[i],
                )

                if marker is not None:
                    marker_array.markers.append(marker)

        if len(marker_array.markers) > 0:
            self._marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = SegmentationNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
