import numpy as np
import cv2
import glob
import os
import argparse
from camera_calibration import CameraCalibration


class StereoCalibrator:
    def __init__(self, checkerboard_size=(7, 10), square_size=25.0):
        self.checkerboard_size = (checkerboard_size[0] - 1, checkerboard_size[1] - 1)
        self.square_size = square_size

        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        self.objp = np.zeros(
            (self.checkerboard_size[0] * self.checkerboard_size[1], 3), np.float32
        )
        self.objp[:, :2] = np.mgrid[
            0 : self.checkerboard_size[0], 0 : self.checkerboard_size[1]
        ].T.reshape(-1, 2)
        self.objp *= square_size

        self.objpoints = []
        self.imgpoints_left = []
        self.imgpoints_right = []

        self.image_size = None

        self.calibration = CameraCalibration()

    def load_image_pairs(self, image_dir, visualize=False):
        # left_images = sorted(glob.glob(os.path.join(image_dir, "left_*.png")))
        # right_images = sorted(glob.glob(os.path.join(image_dir, "right_*.png")))
        left_images = sorted(glob.glob(os.path.join(image_dir, "*_L.png")))
        right_images = sorted(glob.glob(os.path.join(image_dir, "*_R.png")))

        if len(left_images) == 0 or len(right_images) == 0:
            print(f"ERROR: No images found in {image_dir}")
            return False

        print(f"\nFound {len(left_images)} left and {len(right_images)} right images")
        print(
            f"Checkerboard: {self.checkerboard_size[0]}x{self.checkerboard_size[1]} corners"
        )
        print(f"Square size: {self.square_size} units\n")

        successful_pairs = 0

        for idx, (left_path, right_path) in enumerate(zip(left_images, right_images)):
            left_name = os.path.basename(left_path)
            right_name = os.path.basename(right_path)

            print(f"[{idx+1}/{len(left_images)}] Processing: {left_name}, {right_name}")

            img_left = cv2.imread(left_path)
            img_right = cv2.imread(right_path)

            if img_left is None or img_right is None:
                print("Failed to load images")
                continue

            gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

            if self.image_size is None:
                self.image_size = gray_left.shape[::-1]
                print(f"  Image size: {self.image_size[0]}x{self.image_size[1]}")

            ret_left, corners_left = cv2.findChessboardCorners(
                gray_left,
                self.checkerboard_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            ret_right, corners_right = cv2.findChessboardCorners(
                gray_right,
                self.checkerboard_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
            )

            if ret_left and ret_right:
                corners_left = cv2.cornerSubPix(
                    gray_left, corners_left, (11, 11), (-1, -1), self.criteria
                )
                corners_right = cv2.cornerSubPix(
                    gray_right, corners_right, (11, 11), (-1, -1), self.criteria
                )

                self.objpoints.append(self.objp)
                self.imgpoints_left.append(corners_left)
                self.imgpoints_right.append(corners_right)
                successful_pairs += 1

                if visualize:
                    img_left_corners = cv2.drawChessboardCorners(
                        img_left.copy(), self.checkerboard_size, corners_left, ret_left
                    )
                    img_right_corners = cv2.drawChessboardCorners(
                        img_right.copy(),
                        self.checkerboard_size,
                        corners_right,
                        ret_right,
                    )
                    cv2.imwrite(f"{left_name}_corners.png", img_left_corners)
                    cv2.imwrite(f"{right_name}_corners.png", img_right_corners)
            else:
                print("  Failed to find corners")

        print(f"\n{'='*60}")
        print(f"Successfully processed {successful_pairs}/{len(left_images)} pairs")
        print(f"{'='*60}\n")

        return successful_pairs > 0

    def calibrate_cameras(self):
        print("=" * 60)
        print("CALIBRATING LEFT CAMERA")
        print("=" * 60)

        flags = cv2.CALIB_FIX_K3

        (
            ret_left,
            self.calibration.camera_matrix_left,
            self.calibration.dist_coeffs_left,
            rvecs_left,
            tvecs_left,
        ) = cv2.calibrateCamera(
            self.objpoints,
            self.imgpoints_left,
            self.image_size,
            None,
            None,
            flags=flags,
        )

        print(f"RMS reprojection error: {ret_left:.4f} pixels")
        print(f"\nCamera Matrix:\n{self.calibration.camera_matrix_left}")
        print(
            f"\nDistortion Coefficients:\n{self.calibration.dist_coeffs_left.ravel()}"
        )

        print("\n" + "=" * 60)
        print("CALIBRATING RIGHT CAMERA")
        print("=" * 60)

        (
            ret_right,
            self.calibration.camera_matrix_right,
            self.calibration.dist_coeffs_right,
            rvecs_right,
            tvecs_right,
        ) = cv2.calibrateCamera(
            self.objpoints,
            self.imgpoints_right,
            self.image_size,
            None,
            None,
            flags=flags,
        )

        print(f"RMS reprojection error: {ret_right:.4f} pixels")
        print(f"\nCamera Matrix:\n{self.calibration.camera_matrix_right}")
        print(
            f"\nDistortion Coefficients:\n{self.calibration.dist_coeffs_right.ravel()}"
        )
        print("\n")
        return ret_left, ret_right

    def stereo_calibrate(self):
        print("=" * 60)
        print("PERFORMING STEREO CALIBRATION")
        print("=" * 60)

        flags = cv2.CALIB_FIX_INTRINSIC

        (
            ret,
            _,
            _,
            _,
            _,
            self.calibration.R,
            self.calibration.T,
            self.calibration.E,
            self.calibration.F,
        ) = cv2.stereoCalibrate(
            self.objpoints,
            self.imgpoints_left,
            self.imgpoints_right,
            self.calibration.camera_matrix_left,
            self.calibration.dist_coeffs_left,
            self.calibration.camera_matrix_right,
            self.calibration.dist_coeffs_right,
            self.image_size,
            criteria=self.criteria,
            flags=flags,
        )

        baseline = np.linalg.norm(self.calibration.T)
        print(f"RMS reprojection error: {ret:.4f} pixels")
        print(f"\nRotation matrix:\n{self.calibration.R}")
        print(f"\nTranslation vector:\n{self.calibration.T.ravel()}")
        print(f"\nBaseline: {baseline:.4f} units")

        rotation_angle = (
            np.arccos(np.clip((np.trace(self.calibration.R) - 1) / 2, -1, 1))
            * 180
            / np.pi
        )
        print(f"Rotation angle: {rotation_angle:.2f} degrees\n")
        return ret

    def stereo_rectify(self):
        """Compute rectification with ROI-aware scaling to maximize resolution."""
        print("=" * 60)
        print("COMPUTING STEREO RECTIFICATION WITH ROI SCALING")
        print("=" * 60)

        # First get ROI with alpha=0
        (R1, R2, P1, P2, Q, roi_left, roi_right) = cv2.stereoRectify(
            self.calibration.camera_matrix_left,
            self.calibration.dist_coeffs_left,
            self.calibration.camera_matrix_right,
            self.calibration.dist_coeffs_right,
            self.image_size,
            self.calibration.R,
            self.calibration.T,
            alpha=0,
            newImageSize=self.image_size,
        )

        print("Initial ROI:")
        print(
            f"  Left:  x={roi_left[0]}, y={roi_left[1]}, w={roi_left[2]}, h={roi_left[3]}"
        )
        print(
            f"  Right: x={roi_right[0]}, y={roi_right[1]}, w={roi_right[2]}, h={roi_right[3]}"
        )

        # Use the common ROI (intersection)
        x = max(roi_left[0], roi_right[0])
        y = max(roi_left[1], roi_right[1])
        w = min(roi_left[0] + roi_left[2], roi_right[0] + roi_right[2]) - x
        h = min(roi_left[1] + roi_left[3], roi_right[1] + roi_right[3]) - y

        if w <= 0 or h <= 0:
            print("WARNING: Invalid ROI, using alpha=0 without scaling")
            self.calibration.R1 = R1
            self.calibration.R2 = R2
            self.calibration.P1 = P1
            self.calibration.P2 = P2
            self.calibration.Q = Q
            self.calibration.roi_left = roi_left
            self.calibration.roi_right = roi_right
        else:
            print(f"\nCommon ROI: x={x}, y={y}, w={w}, h={h}")
            print(f"Original resolution: {self.image_size[0]}x{self.image_size[1]}")
            print(f"ROI resolution: {w}x{h}")

            scale_x = self.image_size[0] / w
            scale_y = self.image_size[1] / h
            scale = min(scale_x, scale_y)

            print(f"Scale factor: {scale:.2f}x")
            print(f"Effective resolution increase: {scale:.1f}x")

            P1_scaled = P1.copy()
            P2_scaled = P2.copy()

            P1_scaled[0, 0] *= scale  # fx
            P1_scaled[1, 1] *= scale  # fy

            P2_scaled[0, 0] *= scale
            P2_scaled[1, 1] *= scale

            P1_scaled[0, 2] = (P1[0, 2] - x) * scale
            P1_scaled[1, 2] = (P1[1, 2] - y) * scale

            P2_scaled[0, 2] = (P2[0, 2] - x) * scale
            P2_scaled[1, 2] = (P2[1, 2] - y) * scale

            P2_scaled[0, 3] *= scale

            Q_scaled = Q.copy()
            Q_scaled[0, 3] = (Q[0, 3] - x) * scale  # -cx
            Q_scaled[1, 3] = (Q[1, 3] - y) * scale  # -cy
            Q_scaled[2, 3] *= scale
            Q_scaled[3, 2] /= scale

            self.calibration.R1 = R1
            self.calibration.R2 = R2
            self.calibration.P1 = P1_scaled
            self.calibration.P2 = P2_scaled
            self.calibration.Q = Q_scaled
            self.calibration.roi_left = (
                0,
                0,
                self.image_size[0],
                self.image_size[1],
            )
            self.calibration.roi_right = (0, 0, self.image_size[0], self.image_size[1])

        print(f"\nFinal P1:\n{self.calibration.P1}")
        print(f"Final P2:\n{self.calibration.P2}")

        self.calibration.map_left_x, self.calibration.map_left_y = (
            cv2.initUndistortRectifyMap(
                self.calibration.camera_matrix_left,
                self.calibration.dist_coeffs_left,
                self.calibration.R1,
                self.calibration.P1,
                self.image_size,
                cv2.CV_32FC1,
            )
        )

        self.calibration.map_right_x, self.calibration.map_right_y = (
            cv2.initUndistortRectifyMap(
                self.calibration.camera_matrix_right,
                self.calibration.dist_coeffs_right,
                self.calibration.R2,
                self.calibration.P2,
                self.image_size,
                cv2.CV_32FC1,
            )
        )

        print("\nMap validation:")
        print(
            f"  Left X: [{self.calibration.map_left_x.min():.1f}, {self.calibration.map_left_x.max():.1f}]"
        )
        print(
            f"  Left Y: [{self.calibration.map_left_y.min():.1f}, {self.calibration.map_left_y.max():.1f}]"
        )
        print(f"\nQ matrix:\n{self.calibration.Q}\n")


def main():
    parser = argparse.ArgumentParser(description="Stereo camera calibration")

    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--width", type=int, default=7)
    parser.add_argument("--height", type=int, default=9)
    parser.add_argument("--square-size", type=float, default=20.0)
    parser.add_argument("--output", type=str, default="stereo_calibration.npz")
    parser.add_argument("--visualize", action="store_true")

    args = parser.parse_args()

    if not os.path.isdir(args.image_dir):
        print(f"ERROR: Directory not found: {args.image_dir}")
        return 1

    print("\n" + "=" * 60)
    print("STEREO CAMERA CALIBRATION")
    print("=" * 60)
    print(f"Image directory: {args.image_dir}")
    print(f"Checkerboard: {args.width}x{args.height} internal corners")
    print(f"Output: {args.output}")
    print("=" * 60 + "\n")

    calibrator = StereoCalibrator(
        checkerboard_size=(args.width, args.height), square_size=args.square_size
    )

    if not calibrator.load_image_pairs(args.image_dir, visualize=args.visualize):
        print("\nERROR: Failed to load images!")
        return 1

    try:
        calibrator.calibrate_cameras()
        calibrator.stereo_calibrate()
        calibrator.stereo_rectify()

        calibrator.calibration.image_size = calibrator.image_size
        calibrator.calibration.checkerboard_size = calibrator.checkerboard_size
        calibrator.calibration.square_size = calibrator.square_size

        calibrator.calibration.save(args.output)

        print("=" * 60)
        print("CALIBRATION COMPLETED!")
        print("=" * 60)
        print(
            f"\nRectified images will now fill the full {calibrator.image_size[0]}x{calibrator.image_size[1]} frame"
        )
        print(f"Use '{args.output}' with depth_map_creation.py\n")

        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
