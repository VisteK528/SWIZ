import numpy as np
import json
from datetime import datetime
import cv2


class CameraCalibration:
    """Class for storing and managing stereo camera calibration data."""

    def __init__(self):
        self.camera_matrix_left = None
        self.dist_coeffs_left = None
        self.camera_matrix_right = None
        self.dist_coeffs_right = None

        # Stereo calibration parameters
        self.R = None  # Rotation matrix
        self.T = None  # Translation vector
        self.E = None  # Essential matrix
        self.F = None  # Fundamental matrix

        # Rectification parameters
        self.R1 = None
        self.R2 = None
        self.P1 = None
        self.P2 = None
        self.Q = None  # Disparity-to-depth mapping matrix
        self.roi_left = None
        self.roi_right = None

        # Rectification maps
        self.map_left_x = None
        self.map_left_y = None
        self.map_right_x = None
        self.map_right_y = None

        # Image and checkerboard info
        self.image_size = None
        self.checkerboard_size = None
        self.square_size = None

    def load(self, filepath):
        data = np.load(filepath)

        self.camera_matrix_left = data["camera_matrix_left"]
        self.dist_coeffs_left = data["dist_coeffs_left"]
        self.camera_matrix_right = data["camera_matrix_right"]
        self.dist_coeffs_right = data["dist_coeffs_right"]

        self.R = data["R"]
        self.T = data["T"]
        self.E = data["E"]
        self.F = data["F"]

        self.R1 = data["R1"]
        self.R2 = data["R2"]
        self.P1 = data["P1"]
        self.P2 = data["P2"]
        self.Q = data["Q"]
        self.roi_left = data["roi_left"]
        self.roi_right = data["roi_right"]

        self.map_left_x = data["map_left_x"]
        self.map_left_y = data["map_left_y"]
        self.map_right_x = data["map_right_x"]
        self.map_right_y = data["map_right_y"]

        self.image_size = tuple(data["image_size"])
        self.checkerboard_size = tuple(data["checkerboard_size"])
        self.square_size = float(data["square_size"])

        return self

    def save(self, filepath):
        np.savez(
            filepath,
            camera_matrix_left=self.camera_matrix_left,
            dist_coeffs_left=self.dist_coeffs_left,
            camera_matrix_right=self.camera_matrix_right,
            dist_coeffs_right=self.dist_coeffs_right,
            R=self.R,
            T=self.T,
            E=self.E,
            F=self.F,
            R1=self.R1,
            R2=self.R2,
            P1=self.P1,
            P2=self.P2,
            Q=self.Q,
            roi_left=self.roi_left,
            roi_right=self.roi_right,
            map_left_x=self.map_left_x,
            map_left_y=self.map_left_y,
            map_right_x=self.map_right_x,
            map_right_y=self.map_right_y,
            image_size=self.image_size,
            checkerboard_size=self.checkerboard_size,
            square_size=self.square_size,
        )

        metadata_file = filepath.replace(".npz", "_info.json")
        metadata = {
            "calibration_date": datetime.now().isoformat(),
            "checkerboard_size": self.checkerboard_size,
            "square_size": self.square_size,
            "image_size": self.image_size,
            "baseline": float(np.linalg.norm(self.T)),
            "camera_matrix_left": self.camera_matrix_left.tolist(),
            "camera_matrix_right": self.camera_matrix_right.tolist(),
            "dist_coeffs_left": self.dist_coeffs_left.ravel().tolist(),
            "dist_coeffs_right": self.dist_coeffs_right.ravel().tolist(),
            "rotation_matrix": self.R.tolist(),
            "translation_vector": self.T.ravel().tolist(),
        }

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    def get_baseline(self):
        if self.T is not None:
            return np.linalg.norm(self.T)
        return None

    def rectify_images(self, img_left, img_right):
        if self.map_left_x is None or self.map_right_x is None:
            raise ValueError(
                "Rectification maps not available. Load calibration first."
            )

        rect_left = cv2.remap(
            img_left, self.map_left_x, self.map_left_y, cv2.INTER_LINEAR
        )
        rect_right = cv2.remap(
            img_right, self.map_right_x, self.map_right_y, cv2.INTER_LINEAR
        )

        return rect_left, rect_right
