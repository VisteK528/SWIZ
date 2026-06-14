import copy
import pathlib
from typing import List, Optional, Tuple

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
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from ultralytics import YOLO
from visualization_msgs.msg import Marker, MarkerArray

image_qos = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


MODEL_POINTS = 2000
CYLINDER_RADIUS = 0.02232
CYLINDER_HEIGHT = 0.06904

# Disk-case detection thresholds
DISK_PLANARITY_THRESHOLD = 0.05
DISK_RATIO_THRESHOLD = 0.4


def create_cylinder_model(
    radius: float,
    height: float,
    n_pts: int = MODEL_POINTS,
    with_caps: bool = True,
) -> o3d.geometry.PointCloud:
    if with_caps:
        area_side = 2.0 * np.pi * radius * height
        area_cap = np.pi * radius * radius
        total = area_side + 2.0 * area_cap
        n_side = int(n_pts * area_side / total)
        n_cap = max(1, int(n_pts * area_cap / total))

        theta = np.random.uniform(0.0, 2.0 * np.pi, n_side)
        side = np.column_stack(
            [
                radius * np.cos(theta),
                radius * np.sin(theta),
                np.random.uniform(-height / 2.0, height / 2.0, n_side),
            ]
        )
        side_n = np.column_stack([np.cos(theta), np.sin(theta), np.zeros(n_side)])

        def cap(z_val, normal_z):
            tc = np.random.uniform(0.0, 2.0 * np.pi, n_cap)
            r = radius * np.sqrt(np.random.uniform(0.0, 1.0, n_cap))
            pts_c = np.column_stack(
                [r * np.cos(tc), r * np.sin(tc), np.full(n_cap, z_val)]
            )
            n_c = np.tile([0.0, 0.0, normal_z], (n_cap, 1))
            return pts_c, n_c

        top_p, top_n = cap(+height / 2.0, +1.0)
        bot_p, bot_n = cap(-height / 2.0, -1.0)

        pts = np.vstack([side, top_p, bot_p])
        normals = np.vstack([side_n, top_n, bot_n])
    else:
        theta = np.random.uniform(0.0, 2.0 * np.pi, n_pts)
        pts = np.column_stack(
            [
                radius * np.cos(theta),
                radius * np.sin(theta),
                np.random.uniform(-height / 2.0, height / 2.0, n_pts),
            ]
        )
        normals = np.column_stack([np.cos(theta), np.sin(theta), np.zeros(n_pts)])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd


def create_cap_only_model(
    radius: float,
    height: float,
    n_pts: int = MODEL_POINTS,
) -> o3d.geometry.PointCloud:
    n_cap = n_pts
    tc = np.random.uniform(0.0, 2.0 * np.pi, n_cap)
    r = radius * np.sqrt(np.random.uniform(0.0, 1.0, n_cap))
    pts_c = np.column_stack(
        [r * np.cos(tc), r * np.sin(tc), np.full(n_cap, +height / 2.0)]
    )
    n_c = np.tile([0.0, 0.0, +1.0], (n_cap, 1))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_c)
    pcd.normals = o3d.utility.Vector3dVector(n_c)
    return pcd


def _orthonormal_basis(axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = axis / np.linalg.norm(axis)
    helper = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = helper - np.dot(helper, a) * a
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    return e1, e2


def _camera_facing_subset(
    model: o3d.geometry.PointCloud,
    init_T: np.ndarray,
    camera_pos: np.ndarray = np.zeros(3),
    cos_threshold: float = 0.0,
) -> o3d.geometry.PointCloud:
    m_pts = np.asarray(model.points)
    m_nrm = np.asarray(model.normals)
    R = init_T[:3, :3]
    t = init_T[:3, 3]
    world_pts = m_pts @ R.T + t
    world_nrm = m_nrm @ R.T
    to_cam = camera_pos - world_pts
    to_cam /= np.linalg.norm(to_cam, axis=1, keepdims=True) + 1e-12
    dot = (world_nrm * to_cam).sum(axis=1)
    keep = np.where(dot > cos_threshold)[0]
    if keep.size < 30:
        keep = np.arange(len(m_pts))
    return model.select_by_index(keep.tolist())


def _rot_z_to(axis: np.ndarray) -> np.ndarray:
    z = np.array([0.0, 0.0, 1.0])
    a = axis / np.linalg.norm(axis)
    c = float(np.dot(z, a))
    if c > 1.0 - 1e-9:
        return np.eye(3)
    if c < -1.0 + 1e-9:
        return np.diag([1.0, -1.0, -1.0])
    v = np.cross(z, a)
    s = np.linalg.norm(v)
    angle = np.arccos(np.clip(c, -1.0, 1.0))
    return Rotation.from_rotvec(angle * v / s).as_matrix()


def _detect_cap(
    pts: np.ndarray,
    normals: np.ndarray,
    radius: float,
    camera_pos: np.ndarray = np.zeros(3),
    distance_threshold: float = 0.0025,
    min_inliers: int = 25,
    min_circularity: float = 0.45,
) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
    n = len(pts)
    if n < min_inliers:
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    try:
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=300,
        )
    except Exception:
        return None

    if len(inliers) < min_inliers:
        return None

    plane_normal = np.array(plane_model[:3], dtype=float)
    nn = np.linalg.norm(plane_normal)
    if nn < 1e-9:
        return None
    plane_normal /= nn

    cap_pts = pts[inliers]
    cap_centre = cap_pts.mean(axis=0)

    centred = cap_pts - cap_centre
    par = centred @ plane_normal
    in_plane = centred - np.outer(par, plane_normal)

    if len(in_plane) < 3:
        return None
    cov = np.cov(in_plane.T)
    ev = np.linalg.eigvalsh(cov)
    ev = np.maximum(ev, 0.0)
    if ev[2] < 1e-9:
        return None
    circularity = ev[1] / ev[2]
    extent = 2.0 * float(np.sqrt(ev[2]))

    if circularity < min_circularity:
        return None
    if extent > 2.4 * radius:
        return None

    # Orient toward camera.
    to_camera = camera_pos - cap_centre
    if np.dot(plane_normal, to_camera) < 0:
        plane_normal = -plane_normal

    return plane_normal, cap_centre, len(inliers)


def _refine_axis(
    pts: np.ndarray,
    axis_init: np.ndarray,
    point_init: np.ndarray,
    radius: float,
    n_iters: int = 8,
) -> Tuple[np.ndarray, np.ndarray, float]:
    axis = axis_init / np.linalg.norm(axis_init)
    axis_pt = point_init.astype(float).copy()

    for _ in range(n_iters):
        e1, e2 = _orthonormal_basis(axis)
        u = (pts - axis_pt) @ e1
        v = (pts - axis_pt) @ e2

        def f_c(c):
            return np.sqrt((u - c[0]) ** 2 + (v - c[1]) ** 2) - radius

        c_opt = least_squares(f_c, np.zeros(2), method="lm").x
        axis_pt = axis_pt + c_opt[0] * e1 + c_opt[1] * e2

        def f_axis(p):
            R1 = Rotation.from_rotvec(p[0] * e1).as_matrix()
            R2 = Rotation.from_rotvec(p[1] * e2).as_matrix()
            a = R2 @ R1 @ axis
            d = pts - axis_pt
            par = d @ a
            perp = d - np.outer(par, a)
            return np.linalg.norm(perp, axis=1) - radius

        ang = least_squares(f_axis, np.zeros(2), method="lm").x
        R1 = Rotation.from_rotvec(ang[0] * e1).as_matrix()
        R2 = Rotation.from_rotvec(ang[1] * e2).as_matrix()
        axis = R2 @ R1 @ axis
        axis /= np.linalg.norm(axis)

    d = pts - axis_pt
    par = d @ axis
    perp = d - np.outer(par, axis)
    res = float(np.mean(np.abs(np.linalg.norm(perp, axis=1) - radius)))
    return axis, axis_pt, res


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

        self.declare_parameter("save_segmented_clouds", False)
        self.declare_parameter("cloud_save_dir", "/tmp/swiz_clouds")
        self.declare_parameter("cylinder_radius", CYLINDER_RADIUS)
        self.declare_parameter("cylinder_height", CYLINDER_HEIGHT)
        self.declare_parameter("model_points", MODEL_POINTS)
        self.declare_parameter("pose_max_rmse", 0.006)
        self.declare_parameter("pose_min_fitness", 0.30)
        self.declare_parameter("publish_low_quality_poses", False)
        self.declare_parameter("marker_lifetime_sec", 10)
        self.declare_parameter("verbose_pose", False)
        self.declare_parameter("disk_planarity_threshold", DISK_PLANARITY_THRESHOLD)
        self.declare_parameter("disk_ratio_threshold", DISK_RATIO_THRESHOLD)

        self._save_clouds = self.get_parameter("save_segmented_clouds").value
        self._save_dir = pathlib.Path(self.get_parameter("cloud_save_dir").value)
        if self._save_clouds:
            self._save_dir.mkdir(parents=True, exist_ok=True)
            self.get_logger().info(f"Saving segmented clouds to {self._save_dir}")

        self.cylinder_radius = float(self.get_parameter("cylinder_radius").value)
        self.cylinder_height = float(self.get_parameter("cylinder_height").value)
        self._model_points = int(self.get_parameter("model_points").value)
        self._pose_max_rmse = float(self.get_parameter("pose_max_rmse").value)
        self._pose_min_fitness = float(self.get_parameter("pose_min_fitness").value)
        self._publish_low_quality = bool(
            self.get_parameter("publish_low_quality_poses").value
        )
        self._marker_lifetime = int(self.get_parameter("marker_lifetime_sec").value)
        self._verbose_pose = bool(self.get_parameter("verbose_pose").value)
        self._disk_planarity_threshold = float(
            self.get_parameter("disk_planarity_threshold").value
        )
        self._disk_ratio_threshold = float(
            self.get_parameter("disk_ratio_threshold").value
        )

        np.random.seed(0)
        self.cylinder_model = create_cylinder_model(
            self.cylinder_radius, self.cylinder_height, self._model_points
        )

        np.random.seed(1)
        self.cap_only_model = create_cap_only_model(
            self.cylinder_radius, self.cylinder_height, self._model_points
        )

        self.get_logger().info(
            f"Cylinder model: r={self.cylinder_radius:.4f} m  "
            f"h={self.cylinder_height:.4f} m  "
            f"{len(self.cylinder_model.points)} points"
        )

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

    def _solve_disk_case(
        self,
        pcd: o3d.geometry.PointCloud,
        pts: np.ndarray,
        centroid: np.ndarray,
        log,
    ) -> Optional[dict]:
        try:
            plane_model, inliers = pcd.segment_plane(
                distance_threshold=0.002, ransac_n=3, num_iterations=200
            )
        except Exception as exc:
            log(f"disk-mode plane fit failed: {exc}")
            return None

        plane_normal = np.array(plane_model[:3], dtype=float)
        n_norm = np.linalg.norm(plane_normal)
        if n_norm < 1e-9:
            log("disk-mode plane normal degenerate")
            return None
        plane_normal /= n_norm

        cap_pts = pts[inliers] if len(inliers) >= 10 else pts
        cap_centre = cap_pts.mean(axis=0)

        to_camera = -cap_centre  # vector from cap to camera (origin)
        if np.dot(plane_normal, to_camera) < 0:
            plane_normal = -plane_normal

        cyl_centre = cap_centre - plane_normal * (self.cylinder_height / 2.0)

        T_init = np.eye(4)
        T_init[:3, :3] = _rot_z_to(plane_normal)
        T_init[:3, 3] = cyl_centre

        cap_model = copy.deepcopy(self.cap_only_model)

        init_eval = o3d.pipelines.registration.evaluate_registration(
            cap_model, pcd, self.cylinder_radius * 0.5, T_init
        )

        icp = o3d.pipelines.registration.registration_icp(
            cap_model,
            pcd,
            max_correspondence_distance=self.cylinder_radius * 0.4,
            init=T_init,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=100,
                relative_fitness=1e-7,
                relative_rmse=1e-7,
            ),
        )

        T_icp = np.asarray(icp.transformation)
        icp_axis = T_icp[:3, :3] @ np.array([0.0, 0.0, 1.0])
        icp_axis /= np.linalg.norm(icp_axis)

        axis_drift_deg = np.degrees(
            np.arccos(np.clip(abs(np.dot(icp_axis, plane_normal)), -1.0, 1.0))
        )
        centre_drift = np.linalg.norm(T_icp[:3, 3] - cyl_centre)

        if axis_drift_deg <= 10.0 and centre_drift <= 0.010:
            T_final = T_icp
            fitness = float(icp.fitness)
            rmse = float(icp.inlier_rmse)
            log(
                f"disk-mode ICP accepted: drift={axis_drift_deg:.1f}°, "
                f"centre_drift={centre_drift * 1000:.2f} mm"
            )
        else:
            T_final = T_init
            fitness = float(init_eval.fitness)
            rmse = float(init_eval.inlier_rmse)
            log(
                f"disk-mode ICP rejected (drift={axis_drift_deg:.1f}°, "
                f"centre_drift={centre_drift * 1000:.2f} mm); using plane-normal seed"
            )

        final_axis = T_final[:3, :3] @ np.array([0.0, 0.0, 1.0])
        final_axis /= np.linalg.norm(final_axis)

        return dict(
            score=rmse - 0.001 * fitness,
            hyp="DISK-cap",
            transform=T_final,
            fitness=fitness,
            rmse=rmse,
            radial_res=0.0,
            centre=T_final[:3, 3].copy(),
            axis=final_axis,
        )

    def estimate_pose(
        self,
        scene_pcd: o3d.geometry.PointCloud,
        object_id: int = 0,
    ) -> Optional[dict]:
        lg = self.get_logger()
        log = (
            (lambda m: lg.info(f"[id={object_id}] {m}"))
            if self._verbose_pose
            else (lambda m: lg.debug(f"[id={object_id}] {m}"))
        )

        if len(scene_pcd.points) < 30:
            log(f"too few input points: {len(scene_pcd.points)}")
            return None

        pcd = copy.deepcopy(scene_pcd)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        if len(pcd.points) >= 50:
            labels = np.array(
                pcd.cluster_dbscan(eps=0.015, min_points=8, print_progress=False)
            )
            if labels.size and labels.max() >= 0:
                valid = labels[labels >= 0]
                largest = int(np.argmax(np.bincount(valid)))
                pcd = pcd.select_by_index(np.where(labels == largest)[0])

        pts = np.asarray(pcd.points)
        log(f"after cleanup: {len(pts)} pts")
        if len(pts) < 30:
            log("too few points after cleanup")
            return None

        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.cylinder_radius * 1.5, max_nn=30
            )
        )
        pcd.orient_normals_towards_camera_location(np.zeros(3))

        centroid = pts.mean(axis=0)

        eigvals, eigvecs = np.linalg.eigh(np.cov((pts - centroid).T))
        eigvals = np.maximum(eigvals, 0.0)

        planarity = eigvals[0] / (eigvals[1] + 1e-12)
        disk_ratio = eigvals[1] / (eigvals[2] + 1e-12)
        is_disk_like = (
            planarity < self._disk_planarity_threshold
            and disk_ratio > self._disk_ratio_threshold
        )

        log(
            f"shape: planarity={planarity:.3f}  disk_ratio={disk_ratio:.3f}  "
            f"-> {'DISK' if is_disk_like else 'NORMAL'}"
        )

        if is_disk_like:
            best = self._solve_disk_case(pcd, pts, centroid, log)
            if best is None:
                log("disk-mode solver failed")
                return None

            log(
                f"best: hyp={best['hyp']:9s}  fitness={best['fitness']:.3f}  "
                f"rmse={best['rmse'] * 1000:5.2f} mm  (disk mode)"
            )

            return dict(
                transform=best["transform"],
                centre=best["centre"],
                axis=best["axis"],
                fitness=best["fitness"],
                rmse=best["rmse"],
                radial_res=best["radial_res"],
                clean_pcd=pcd,
                n_points=len(pts),
                hyp=best["hyp"],
                ok=(best["rmse"] <= self._pose_max_rmse),
            )

        candidates: List[Tuple[str, np.ndarray, np.ndarray]] = [
            ("PCA-major", eigvecs[:, 2], centroid),
            ("PCA-minor", eigvecs[:, 0], centroid),
        ]

        cap_info = None
        try:
            normals_arr = np.asarray(pcd.normals)
            cap_info = _detect_cap(
                pts, normals_arr, self.cylinder_radius, camera_pos=np.zeros(3)
            )
        except Exception as exc:
            log(f"cap detection failed: {exc}")

        cap_axis_skip_refine = False
        if cap_info is not None:
            cap_normal, cap_centre, n_inliers = cap_info
            seed_axis_pt = cap_centre - cap_normal * (self.cylinder_height / 2.0)
            candidates.append(("CAP-normal", cap_normal, seed_axis_pt))
            cap_axis_skip_refine = True
            log(
                f"cap detected: {n_inliers} inliers, "
                f"normal=({cap_normal[0]:+.2f},{cap_normal[1]:+.2f},{cap_normal[2]:+.2f})"
            )

        try:
            rc, ra, rr, inliers = pyr.Cylinder().fit(
                pts, thresh=0.0025, maxIteration=4000
            )
            ra_arr = np.asarray(ra, dtype=float)
            rc_arr = np.asarray(rc, dtype=float)
            n_ra = np.linalg.norm(ra_arr)
            if (
                n_ra > 1e-9
                and 0.4 * self.cylinder_radius < rr < 2.5 * self.cylinder_radius
                and len(inliers) >= 20
            ):
                candidates.append(("RANSAC", ra_arr / n_ra, rc_arr))
                log(f"RANSAC seed: r={rr:.4f}  inliers={len(inliers)}/{len(pts)}")
            else:
                log(f"RANSAC seed rejected (r={rr:.4f}, inliers={len(inliers)})")
        except Exception as exc:
            log(f"RANSAC seed failed: {exc}")

        refined: List[Tuple[str, np.ndarray, np.ndarray, float]] = []
        for name, ax0, p0 in candidates:
            if name == "CAP-normal":
                d = pts - p0
                par = d @ ax0
                perp = d - np.outer(par, ax0)
                res = float(
                    np.mean(np.abs(np.linalg.norm(perp, axis=1) - self.cylinder_radius))
                )
                refined.append((name, ax0, p0, res))
                log(
                    f"hyp {name:10s}: radial residual = {res * 1000:5.2f} mm (no refine)"
                )
                continue
            try:
                ax, p, res = _refine_axis(pts, ax0, p0, self.cylinder_radius)
                refined.append((name, ax, p, res))
                log(f"hyp {name:10s}: radial residual = {res * 1000:5.2f} mm")
            except Exception as exc:
                log(f"hyp {name}: refinement failed ({exc})")

        if not refined:
            log("all axis hypotheses failed")
            return None

        refined.sort(key=lambda t: t[3])
        best_res = refined[0][3]
        keep_threshold = max(2.0 * best_res + 0.5e-3, 3.0e-3)
        kept = [t for t in refined if t[3] <= keep_threshold]

        if cap_axis_skip_refine:
            cap_hyp = next((t for t in refined if t[0] == "CAP-normal"), None)
            if cap_hyp is not None and cap_hyp not in kept:
                kept.append(cap_hyp)
                log("cap-normal hypothesis force-kept despite higher residual")

        top = kept[: min(3, len(kept))]
        log(
            f"keeping {len(top)} hypothesis/es "
            f"(threshold = {keep_threshold * 1000:.2f} mm)"
        )

        best = None

        for name, axis, axis_pt, res in top:
            is_cap_hyp = name == "CAP-normal"

            if is_cap_hyp:
                axial_offsets = [0.0]
                signs = [+1.0]
            else:
                proj = (pts - axis_pt) @ axis
                t_lo, t_hi = float(proj.min()), float(proj.max())
                visible = t_hi - t_lo

                if visible >= 0.85 * self.cylinder_height:
                    axial_offsets = [0.5 * (t_lo + t_hi)]
                else:
                    mid = 0.5 * (t_lo + t_hi)
                    axial_offsets = [
                        mid,
                        t_lo + self.cylinder_height / 2.0,
                        t_hi - self.cylinder_height / 2.0,
                    ]
                signs = [+1.0, -1.0]

            def consider(transform, fitness, rmse):
                nonlocal best
                score = rmse - 0.001 * fitness
                if best is None or score < best["score"]:
                    best = dict(
                        score=score,
                        hyp=name,
                        transform=transform,
                        fitness=float(fitness),
                        rmse=float(rmse),
                        radial_res=res,
                    )

            for ac in axial_offsets:
                centre = axis_pt + ac * axis
                for sign in signs:
                    signed_axis = sign * axis
                    T = np.eye(4)
                    T[:3, :3] = _rot_z_to(signed_axis)
                    T[:3, 3] = centre

                    model_view = _camera_facing_subset(self.cylinder_model, T)

                    init_eval = o3d.pipelines.registration.evaluate_registration(
                        model_view,
                        pcd,
                        self.cylinder_radius * 0.5,
                        T,
                    )
                    consider(T, init_eval.fitness, init_eval.inlier_rmse)

                    icp = o3d.pipelines.registration.registration_icp(
                        model_view,
                        pcd,
                        max_correspondence_distance=self.cylinder_radius * 0.5,
                        init=T,
                        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                            max_iteration=200,
                            relative_fitness=1e-7,
                            relative_rmse=1e-7,
                        ),
                    )

                    T_icp = np.asarray(icp.transformation)
                    icp_axis = T_icp[:3, :3] @ np.array([0.0, 0.0, 1.0])
                    icp_axis /= np.linalg.norm(icp_axis)
                    axis_drift = np.degrees(
                        np.arccos(
                            np.clip(abs(np.dot(icp_axis, signed_axis)), -1.0, 1.0)
                        )
                    )
                    centre_drift = np.linalg.norm(T_icp[:3, 3] - centre)
                    drift_limit_deg = 15.0 if is_cap_hyp else 25.0
                    drift_limit_m = 0.012 if is_cap_hyp else self.cylinder_radius
                    if axis_drift <= drift_limit_deg and centre_drift <= drift_limit_m:
                        consider(T_icp, icp.fitness, icp.inlier_rmse)

        T_final = best["transform"]
        centre = T_final[:3, 3].copy()
        axis = T_final[:3, :3] @ np.array([0.0, 0.0, 1.0])
        axis /= np.linalg.norm(axis)

        if cap_info is not None and best["hyp"] != "CAP-normal":
            cap_normal, cap_centre, _ = cap_info
            axis_vs_cap_deg = np.degrees(
                np.arccos(np.clip(abs(np.dot(axis, cap_normal)), -1.0, 1.0))
            )
            if axis_vs_cap_deg > 30.0:
                log(
                    f"WARNING: best hyp ({best['hyp']}) disagrees with "
                    f"cap normal by {axis_vs_cap_deg:.1f}° — overriding "
                    "with cap-derived pose"
                )
                axis = cap_normal
                centre = cap_centre - cap_normal * (self.cylinder_height / 2.0)
                T_final = np.eye(4)
                T_final[:3, :3] = _rot_z_to(axis)
                T_final[:3, 3] = centre

                cap_model = copy.deepcopy(self.cap_only_model)
                ov_eval = o3d.pipelines.registration.evaluate_registration(
                    cap_model, pcd, self.cylinder_radius * 0.5, T_final
                )
                best = dict(
                    score=ov_eval.inlier_rmse - 0.001 * ov_eval.fitness,
                    hyp="CAP-override",
                    transform=T_final,
                    fitness=float(ov_eval.fitness),
                    rmse=float(ov_eval.inlier_rmse),
                    radial_res=best["radial_res"],
                )

        data_thickness = float(np.sqrt(eigvals[0])) * 2.0
        proj_along_axis = (pts - centre) @ axis
        axial_extent = float(proj_along_axis.max() - proj_along_axis.min())

        if data_thickness < 0.005 and axial_extent > 0.5 * self.cylinder_height:
            log(
                "WARNING: chosen pose inconsistent with disk-like input "
                f"(thickness={data_thickness * 1000:.2f} mm, "
                f"axial_extent={axial_extent * 1000:.2f} mm) — "
                "falling back to disk-case solver"
            )
            fallback = self._solve_disk_case(pcd, pts, centroid, log)
            if fallback is not None:
                return dict(
                    transform=fallback["transform"],
                    centre=fallback["centre"],
                    axis=fallback["axis"],
                    fitness=fallback["fitness"],
                    rmse=fallback["rmse"],
                    radial_res=fallback["radial_res"],
                    clean_pcd=pcd,
                    n_points=len(pts),
                    hyp=fallback["hyp"] + "-fallback",
                    ok=(fallback["rmse"] <= self._pose_max_rmse),
                )

        log(
            f"best: hyp={best['hyp']:9s}  fitness={best['fitness']:.3f}  "
            f"rmse={best['rmse'] * 1000:5.2f} mm  "
            f"radial_res={best['radial_res'] * 1000:5.2f} mm"
        )

        cap_derived = best["hyp"] in ("CAP-normal", "CAP-override")
        if cap_derived:
            is_ok = best["rmse"] <= self._pose_max_rmse
        else:
            is_ok = (
                best["rmse"] <= self._pose_max_rmse
                and best["fitness"] >= self._pose_min_fitness
            )

        return dict(
            transform=T_final,
            centre=centre,
            axis=axis,
            fitness=best["fitness"],
            rmse=best["rmse"],
            radial_res=best["radial_res"],
            clean_pcd=pcd,
            n_points=len(pts),
            hyp=best["hyp"],
            ok=is_ok,
        )

    def run_ransac_cylinder(
        self,
        scene_pcd: o3d.geometry.PointCloud,
        stamp,
        frame_id: str,
        object_id: int,
        color_rgb: Optional[Tuple[int, int, int]] = None,
    ) -> Optional[Marker]:
        result = self.estimate_pose(scene_pcd, object_id=object_id)
        if result is None:
            return None

        if not result["ok"]:
            self.get_logger().warn(
                f"[id={object_id}] pose quality low — "
                f"fitness={result['fitness']:.3f}, rmse={result['rmse'] * 1000:.2f} mm"
                f" (>{self._pose_max_rmse * 1000:.1f} mm)"
            )
            if not self._publish_low_quality:
                return None

        self.get_logger().info(
            f"[id={object_id}] pose ✓ hyp={result['hyp']} "
            f"fitness={result['fitness']:.3f} "
            f"rmse={result['rmse'] * 1000:.2f} mm  "
            f"centre={np.round(result['centre'], 4).tolist()}"
        )

        matrix_slice = result["transform"][:3, :3].copy()
        quat = Rotation.from_matrix(matrix_slice).as_quat()
        return self._create_marker(
            object_id,
            frame_id,
            result["centre"],
            quat,
            stamp,
            color_rgb=color_rgb,
            low_quality=not result["ok"],
        )

    def _make_xyzrgb_cloud(self, header, point_chunks, colors):
        chunks = []
        for pts, (r, g, b) in zip(point_chunks, colors):
            if len(pts) == 0:
                continue
            rgb_int = np.uint32((int(r) << 16) | (int(g) << 8) | int(b))
            rgb_float = rgb_int.view(np.float32)
            rgb_col = np.full((len(pts), 1), rgb_float, dtype=np.float32)
            chunks.append(np.hstack([pts.astype(np.float32), rgb_col]))

        combined = np.vstack(chunks) if chunks else np.empty((0, 4), dtype=np.float32)

        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = combined.shape[0]
        msg.is_dense = False
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = msg.point_step * msg.width
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg.data = combined.tobytes()
        return msg

    def _create_marker(
        self,
        object_id: int,
        frame_id: str,
        position: np.ndarray,
        quat: np.ndarray,
        stamp,
        color_rgb: Optional[Tuple[int, int, int]] = None,
        low_quality: bool = False,
    ) -> Marker:
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = frame_id
        marker.ns = "cylinders"
        marker.id = int(object_id)
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD

        marker.pose.position.x = float(position[0])
        marker.pose.position.y = float(position[1])
        marker.pose.position.z = float(position[2])
        marker.pose.orientation.x = float(quat[0])
        marker.pose.orientation.y = float(quat[1])
        marker.pose.orientation.z = float(quat[2])
        marker.pose.orientation.w = float(quat[3])

        marker.scale.x = self.cylinder_radius * 2.0
        marker.scale.y = self.cylinder_radius * 2.0
        marker.scale.z = self.cylinder_height

        if color_rgb is not None:
            marker.color.r = color_rgb[0] / 255.0
            marker.color.g = color_rgb[1] / 255.0
            marker.color.b = color_rgb[2] / 255.0
        else:
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
        marker.color.a = 0.4 if low_quality else 0.75

        marker.lifetime.sec = self._marker_lifetime

        return marker

    def _delete_all_markers(self, stamp, frame_id: str) -> Marker:
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = frame_id
        marker.ns = "cylinders"
        marker.action = Marker.DELETEALL
        return marker

    def _sync_callback(self, image_msg, pointcloud_msg):
        colored_instance_mask = None
        result = None
        track_ids = []

        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")

            results = self.model.track(
                cv_image, persist=True, retina_masks=True, verbose=False, conf=0.2
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

            annotated = result.plot()
            ros_annotated_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            ros_annotated_msg.header = image_msg.header
            self._pub.publish(ros_annotated_msg)

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

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

        downsampled_pcd = o3d_pcd.voxel_down_sample(voxel_size=0.005)

        downsampled_points = np.asarray(downsampled_pcd.points, dtype=np.float32)
        out_msg = pc2.create_cloud_xyz32(pointcloud_msg.header, downsampled_points)
        self._downsampled_cloud_pub.publish(out_msg)

        if self.fx is not None and result is not None and result.masks is not None:
            masks_array = result.masks.data.cpu().numpy()
            self.get_logger().info(f"Detected cylinders: {len(track_ids)}")

            marker_array = MarkerArray()
            marker_array.markers.append(
                self._delete_all_markers(
                    pointcloud_msg.header.stamp, pointcloud_msg.header.frame_id
                )
            )

            seg_point_chunks = []
            seg_colors = []

            for i in range(len(track_ids)):
                track_id = track_ids[i]
                binary_mask = (masks_array[i] > 0.5).astype(np.uint8) * 255

                segmented_pcd = self.mask_pointcloud_with_2d(
                    downsampled_pcd, binary_mask, self.fx, self.fy, self.cx, self.cy
                )
                self.get_logger().info(
                    f"Segmented point cloud has {len(segmented_pcd.points)} points after masking."
                )

                segmented_points = np.asarray(segmented_pcd.points, dtype=np.float32)

                if self._save_clouds and len(segmented_points) > 0:
                    stamp = pointcloud_msg.header.stamp
                    fname = (
                        self._save_dir
                        / f"{stamp.sec}_{stamp.nanosec:09d}_id{track_id}.pcd"
                    )
                    o3d.io.write_point_cloud(str(fname), segmented_pcd)

                np.random.seed(track_id)
                color = tuple(int(x) for x in np.random.randint(50, 255, size=3))
                seg_point_chunks.append(segmented_points)
                seg_colors.append(color)

                marker = self.run_ransac_cylinder(
                    segmented_pcd,
                    pointcloud_msg.header.stamp,
                    pointcloud_msg.header.frame_id,
                    object_id=track_id,
                    color_rgb=color,
                )

                if marker is not None:
                    marker_array.markers.append(marker)

            self._segmented_point_cloud_pub.publish(
                self._make_xyzrgb_cloud(
                    pointcloud_msg.header, seg_point_chunks, seg_colors
                )
            )

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
