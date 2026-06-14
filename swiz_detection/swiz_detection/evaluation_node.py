import csv
import math
import pathlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
import tf2_ros
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker, MarkerArray

CYLINDER_RADIUS = 0.02232
CYLINDER_HEIGHT = 0.06904


def transform_to_matrix(transform_stamped) -> np.ndarray:
    """geometry_msgs/TransformStamped -> 4x4 homogeneous matrix."""
    T = np.eye(4)
    q = transform_stamped.transform.rotation
    T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    tr = transform_stamped.transform.translation
    T[:3, 3] = [tr.x, tr.y, tr.z]
    return T


def axis_angle_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between two axes, ignoring direction (handles the 180 deg flip)."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    c = abs(float(np.dot(a, b)))
    return math.degrees(math.acos(min(1.0, c)))


class PoseValidationNode(Node):
    def __init__(self):
        super().__init__("pose_validation_node")

        self.declare_parameter("world_frame", "world")
        self.declare_parameter("marker_topic", "/cylinder_markers")
        self.declare_parameter("gt_tf_topic", "/tf")
        # Explicit ground-truth frame names (child_frame_id on /tf); if empty, fall
        # back to substring filter.
        # Ground-truth cylinder frames follow the pattern f"{prefix}{i}" for
        # i in 1..num_cylinders (e.g. blue_cylinder_1 .. blue_cylinder_N).
        self.declare_parameter("num_cylinders", 4)
        self.declare_parameter("cylinder_name_prefix", "blue_cylinder_")
        # Substring fallback, used only when num_cylinders <= 0 (auto-discover on /tf).
        self.declare_parameter("gt_name_filter", "cylinder")
        # Local axis of the cylinder geometry inside the Gazebo model (usually +Z).
        self.declare_parameter("cylinder_axis_in_model", [0.0, 0.0, 1.0])
        # Shift the GT centre by this many metres ALONG the GT cylinder axis before
        # comparing. Use it to reconcile a frame-origin convention mismatch: if the
        # GT /tf frame sits at a cap while the estimate reports the geometric centre,
        # set this to +/- CYLINDER_HEIGHT/2 (sign chosen to minimise along-axis err).
        self.declare_parameter("gt_centre_axial_offset", CYLINDER_HEIGHT / 2.0)

        self.declare_parameter("match_distance_threshold", 0.10)
        self.declare_parameter("csv_path", "/tmp/pose_validation.csv")
        self.declare_parameter("summary_period_sec", 5.0)
        self.declare_parameter("publish_gt_markers", True)
        self.declare_parameter("tf_timeout_sec", 0.2)

        self._world_frame = self.get_parameter("world_frame").value
        n_cyl = int(self.get_parameter("num_cylinders").value)
        prefix = self.get_parameter("cylinder_name_prefix").value
        self._gt_names = (
            [f"{prefix}{i}" for i in range(1, n_cyl + 1)] if n_cyl > 0 else []
        )
        self._gt_filter = self.get_parameter("gt_name_filter").value
        self._axis_local = np.array(
            self.get_parameter("cylinder_axis_in_model").value, dtype=float
        )
        self._axial_offset = float(self.get_parameter("gt_centre_axial_offset").value)
        self._match_thresh = float(self.get_parameter("match_distance_threshold").value)
        self._csv_path = pathlib.Path(self.get_parameter("csv_path").value)
        self._publish_gt = bool(self.get_parameter("publish_gt_markers").value)
        self._tf_timeout = float(self.get_parameter("tf_timeout_sec").value)

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self.create_subscription(
            TFMessage,
            self.get_parameter("gt_tf_topic").value,
            self.tf_discovery_cb,
            10,
        )
        self.create_subscription(
            MarkerArray,
            self.get_parameter("marker_topic").value,
            self.markers_cb,
            10,
        )
        self._gt_marker_pub = self.create_publisher(
            MarkerArray, "/gt_cylinder_markers", 10
        )

        self._gt_frames: set = set(self._gt_names)
        self._samples: List[dict] = []
        self._logged_names = False

        self.create_timer(
            float(self.get_parameter("summary_period_sec").value), self._log_summary
        )

        self.get_logger().info(
            f"Pose validation up. world_frame='{self._world_frame}', "
            f"axial_offset={self._axial_offset * 1000:.2f} mm, "
            f"match_threshold={self._match_thresh * 1000:.0f} mm, "
            f"csv='{self._csv_path}'"
        )

    def _is_gt_model(self, name: str) -> bool:
        if self._gt_names:
            return name in self._gt_names
        return self._gt_filter.lower() in name.lower()

    def tf_discovery_cb(self, msg: TFMessage):
        new = False
        for t in msg.transforms:
            name = t.child_frame_id
            if self._is_gt_model(name) and name not in self._gt_frames:
                self._gt_frames.add(name)
                new = True

        if new or (not self._logged_names and msg.transforms):
            all_children = sorted({t.child_frame_id for t in msg.transforms})
            self.get_logger().info("child_frame_ids on /tf: " + ", ".join(all_children))
            self.get_logger().info(
                f"tracking GT frames: {sorted(self._gt_frames)} "
                f"(from num_cylinders/cylinder_name_prefix; "
                f"filter fallback '{self._gt_filter}')"
            )
            self._logged_names = True

    def _current_gt(self, stamp) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Resolve each tracked cylinder frame to (position, R) in the world frame."""
        gt: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for frame in self._gt_frames:
            T = self._lookup_world_transform(frame, stamp)
            if T is None:
                continue
            gt[frame] = (T[:3, 3].copy(), T[:3, :3].copy())
        return gt

    def _gt_axis_world(self, R: np.ndarray) -> np.ndarray:
        a = R @ self._axis_local
        return a / (np.linalg.norm(a) + 1e-12)

    def markers_cb(self, msg: MarkerArray):
        if not self._gt_frames:
            return

        estimates = []
        ref_stamp = None
        for m in msg.markers:
            if m.action != Marker.ADD:
                continue
            if ref_stamp is None:
                ref_stamp = m.header.stamp
            T = self._lookup_world_transform(m.header.frame_id, m.header.stamp)
            if T is None:
                continue

            p_local = np.array(
                [m.pose.position.x, m.pose.position.y, m.pose.position.z]
            )
            q = m.pose.orientation
            R_local = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            axis_local = R_local @ np.array([0.0, 0.0, 1.0])

            p_world = (T[:3, :3] @ p_local) + T[:3, 3]
            axis_world = T[:3, :3] @ axis_local
            axis_world /= np.linalg.norm(axis_world) + 1e-12

            estimates.append({"id": int(m.id), "pos": p_world, "axis": axis_world})

        if not estimates:
            return

        gt = self._current_gt(ref_stamp)
        if not gt:
            self.get_logger().warn(
                "No GT cylinder transforms resolvable from /tf "
                f"(tracking {sorted(self._gt_frames)}); check world_frame and TF tree."
            )
            return
        if self._publish_gt:
            self._publish_gt_markers(gt)

        gt_items = []
        for n, (p, R) in gt.items():
            ax = self._gt_axis_world(R)
            gt_items.append({"name": n, "pos": p + self._axial_offset * ax, "axis": ax})

        for est_idx, gt_idx, _ in self.associate(estimates, gt_items):
            e = estimates[est_idx]
            g = gt_items[gt_idx]

            pos_err_vec = e["pos"] - g["pos"]
            pos_err = float(np.linalg.norm(pos_err_vec))

            axis = g["axis"]
            along = float(np.dot(pos_err_vec, axis))
            perp_vec = pos_err_vec - along * axis
            perp = float(np.linalg.norm(perp_vec))

            ang_err = axis_angle_error_deg(e["axis"], axis)

            self._samples.append(
                {
                    "track_id": e["id"],
                    "gt_name": g["name"],
                    "ex": e["pos"][0],
                    "ey": e["pos"][1],
                    "ez": e["pos"][2],
                    "gx": g["pos"][0],
                    "gy": g["pos"][1],
                    "gz": g["pos"][2],
                    "err_x": pos_err_vec[0],
                    "err_y": pos_err_vec[1],
                    "err_z": pos_err_vec[2],
                    "pos_err": pos_err,
                    "along_err": along,
                    "perp_err": perp,
                    "ang_err_deg": ang_err,
                }
            )

    def _lookup_world_transform(self, frame_id: str, stamp) -> Optional[np.ndarray]:
        if not frame_id:
            return None
        timeout = rclpy.duration.Duration(seconds=self._tf_timeout)
        try:
            tf = self._tf_buffer.lookup_transform(
                self._world_frame, frame_id, stamp, timeout
            )
        except Exception:
            try:
                tf = self._tf_buffer.lookup_transform(
                    self._world_frame, frame_id, rclpy.time.Time(), timeout
                )
            except Exception as exc:
                self.get_logger().warn(
                    f"TF {self._world_frame} <- {frame_id} unavailable: {exc}"
                )
                return None
        return transform_to_matrix(tf)

    @staticmethod
    def _associate(estimates: List[dict], gts: List[dict], thresh: float = None):
        """Greedy nearest-neighbour matching by world-frame centre distance."""
        thresh = thresh if thresh is not None else float("inf")
        pairs_by_dist = []
        for ei, e in enumerate(estimates):
            for gi, g in enumerate(gts):
                d = float(np.linalg.norm(e["pos"] - g["pos"]))
                pairs_by_dist.append((d, ei, gi))
        pairs_by_dist.sort(key=lambda x: x[0])

        used_e, used_g, matches = set(), set(), []
        for d, ei, gi in pairs_by_dist:
            if d > thresh:
                break
            if ei in used_e or gi in used_g:
                continue
            used_e.add(ei)
            used_g.add(gi)
            matches.append((ei, gi, d))
        return matches

    def associate(self, estimates, gts):
        return self._associate(estimates, gts, self._match_thresh)

    def _log_summary(self):
        if not self._samples:
            return
        s = self._summary()
        self.get_logger().info(
            f"[{s['n']} samples] "
            f"pos RMSE={s['pos_rmse_mm']:.2f} mm "
            f"(perp={s['perp_rmse_mm']:.2f}, along={s['along_rmse_mm']:.2f}) | "
            f"axis err mean={s['ang_mean_deg']:.2f} deg "
            f"(RMS={s['ang_rms_deg']:.2f}, max={s['ang_max_deg']:.2f})"
        )

    def _summary(self) -> dict:
        arr = self._samples
        ex = np.array([r["err_x"] for r in arr])
        ey = np.array([r["err_y"] for r in arr])
        ez = np.array([r["err_z"] for r in arr])
        pe = np.array([r["pos_err"] for r in arr])
        al = np.array([r["along_err"] for r in arr])
        pp = np.array([r["perp_err"] for r in arr])
        ae = np.array([r["ang_err_deg"] for r in arr])
        return {
            "n": len(arr),
            "mse_x_mm2": float(np.mean(ex**2)) * 1e6,
            "mse_y_mm2": float(np.mean(ey**2)) * 1e6,
            "mse_z_mm2": float(np.mean(ez**2)) * 1e6,
            "pos_rmse_mm": float(np.sqrt(np.mean(pe**2))) * 1000.0,
            "pos_mean_mm": float(np.mean(pe)) * 1000.0,
            "pos_median_mm": float(np.median(pe)) * 1000.0,
            "pos_max_mm": float(np.max(pe)) * 1000.0,
            "perp_rmse_mm": float(np.sqrt(np.mean(pp**2))) * 1000.0,
            "perp_mean_mm": float(np.mean(pp)) * 1000.0,
            "perp_max_mm": float(np.max(pp)) * 1000.0,
            "along_rmse_mm": float(np.sqrt(np.mean(al**2))) * 1000.0,
            "along_mean_mm": float(np.mean(al)) * 1000.0,
            "along_max_mm": float(np.max(np.abs(al))) * 1000.0,
            "ang_mean_deg": float(np.mean(ae)),
            "ang_rms_deg": float(np.sqrt(np.mean(ae**2))),
            "ang_max_deg": float(np.max(ae)),
        }

    def write_outputs(self):
        if not self._samples:
            self.get_logger().warn("No samples collected; nothing written.")
            return

        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "track_id",
            "gt_name",
            "ex",
            "ey",
            "ez",
            "gx",
            "gy",
            "gz",
            "err_x",
            "err_y",
            "err_z",
            "pos_err",
            "along_err",
            "perp_err",
            "ang_err_deg",
        ]
        with self._csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self._samples)

        s = self._summary()
        summary_path = self._csv_path.with_suffix(".summary.txt")
        lines = [
            f"samples              : {s['n']}",
            f"applied axial offset : {self._axial_offset * 1000:.3f} mm",
            f"position MSE  x      : {s['mse_x_mm2']:.4f} mm^2",
            f"position MSE  y      : {s['mse_y_mm2']:.4f} mm^2",
            f"position MSE  z      : {s['mse_z_mm2']:.4f} mm^2",
            f"position RMSE        : {s['pos_rmse_mm']:.3f} mm",
            f"position mean err    : {s['pos_mean_mm']:.3f} mm",
            f"position median err  : {s['pos_median_mm']:.3f} mm",
            f"position max err     : {s['pos_max_mm']:.3f} mm",
            f"perpendicular RMSE   : {s['perp_rmse_mm']:.3f} mm   (radial / honest)",
            f"perpendicular mean   : {s['perp_mean_mm']:.3f} mm",
            f"perpendicular max    : {s['perp_max_mm']:.3f} mm",
            f"along-axis RMSE      : {s['along_rmse_mm']:.3f} mm",
            f"along-axis mean      : {s['along_mean_mm']:.3f} mm   (signed bias)",
            f"along-axis max |.|   : {s['along_max_mm']:.3f} mm",
            f"axis err  mean       : {s['ang_mean_deg']:.3f} deg",
            f"axis err  RMS        : {s['ang_rms_deg']:.3f} deg",
            f"axis err  max        : {s['ang_max_deg']:.3f} deg",
        ]
        summary_path.write_text("\n".join(lines) + "\n")
        self.get_logger().info(
            f"Wrote {s['n']} samples to {self._csv_path} and summary to {summary_path}"
        )
        for ln in lines:
            self.get_logger().info("  " + ln)

        if abs(self._axial_offset) < 1e-9:
            half_h_mm = CYLINDER_HEIGHT / 2.0 * 1000.0
            if abs(abs(s["along_mean_mm"]) - half_h_mm) < 5.0:
                self.get_logger().warn(
                    f"along-axis bias ({s['along_mean_mm']:.2f} mm) is ~half the "
                    f"cylinder height ({half_h_mm:.2f} mm): looks like a GT "
                    f"frame-origin convention offset. Set 'gt_centre_axial_offset' "
                    f"to {np.sign(s['along_mean_mm']) * CYLINDER_HEIGHT / 2.0:+.5f} "
                    f"to null it; perpendicular RMSE ({s['perp_rmse_mm']:.2f} mm) is "
                    f"your true position accuracy."
                )

    def _publish_gt_markers(self, gt: Dict[str, Tuple[np.ndarray, np.ndarray]]):
        arr = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        for i, (_, (pos, R)) in enumerate(gt.items()):
            m = Marker()
            m.header.frame_id = self._world_frame
            m.header.stamp = stamp
            m.ns = "gt_cylinders"
            m.id = i
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            ax = self._gt_axis_world(R)
            centre = pos + self._axial_offset * ax
            m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, centre)
            q = Rotation.from_matrix(R).as_quat()
            (
                m.pose.orientation.x,
                m.pose.orientation.y,
                m.pose.orientation.z,
                m.pose.orientation.w,
            ) = map(float, q)
            m.scale.x = CYLINDER_RADIUS * 2.0
            m.scale.y = CYLINDER_RADIUS * 2.0
            m.scale.z = CYLINDER_HEIGHT
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.0, 0.0, 0.35
            m.lifetime.sec = 2
            arr.markers.append(m)
        self._gt_marker_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = PoseValidationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.write_outputs()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
