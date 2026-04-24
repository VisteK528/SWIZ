import numpy as np
import cv2
import glob
import os
import argparse
from camera_calibration import CameraCalibration


class DepthMapGenerator:
    def __init__(self, calibration_file):
        self.calibration = CameraCalibration()
        self.calibration.load(calibration_file)

        self.stereo_params = {
            "minDisparity": 0,
            "numDisparities": 16 * 10,  # Must be divisible by 16
            "blockSize": 5,
            "P1": 8 * 3 * 5**2,
            "P2": 32 * 3 * 5**2,
            "disp12MaxDiff": 1,
            "uniquenessRatio": 5,
            "speckleWindowSize": 50,
            "speckleRange": 2,
            "mode": cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        }

        self.stereo = cv2.StereoSGBM.create(**self.stereo_params)

        self.use_wls_filter = False
        self.wls_lambda = 8000.0
        self.wls_sigma = 1.5
        self.right_matcher = None
        self.wls_filter = None

    def enable_wls_filter(self, lambda_value=8000.0, sigma=1.5):
        self.use_wls_filter = True
        self.wls_lambda = lambda_value
        self.wls_sigma = sigma

        self.right_matcher = cv2.ximgproc.createRightMatcher(self.stereo)

        self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.stereo)
        self.wls_filter.setLambda(self.wls_lambda)
        self.wls_filter.setSigmaColor(self.wls_sigma)

    def update_stereo_params(self, **kwargs):
        self.stereo_params.update(kwargs)
        if "blockSize" in kwargs:
            bs = kwargs["blockSize"]
            self.stereo_params["P1"] = 8 * 3 * bs**2
            self.stereo_params["P2"] = 32 * 3 * bs**2
        self.stereo = cv2.StereoSGBM.create(**self.stereo_params)

        if self.use_wls_filter:
            self.right_matcher = cv2.ximgproc.createRightMatcher(self.stereo)
            self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.stereo)
            self.wls_filter.setLambda(self.wls_lambda)
            self.wls_filter.setSigmaColor(self.wls_sigma)

    def compute_disparity(self, img_left, img_right):
        if len(img_left.shape) == 3:
            gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        else:
            gray_left = img_left

        if len(img_right.shape) == 3:
            gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        else:
            gray_right = img_right

        if self.use_wls_filter:
            disparity_left = self.stereo.compute(gray_left, gray_right)
            disparity_right = self.right_matcher.compute(gray_right, gray_left)

            disparity = self.wls_filter.filter(
                disparity_left, gray_left, None, disparity_right
            )
            disparity = disparity.astype(np.float32) / 16.0
        else:
            disparity = (
                self.stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
            )

        return disparity

    def disparity_to_depth(self, disparity):
        points_3d = cv2.reprojectImageTo3D(disparity, self.calibration.Q)

        depth = points_3d[:, :, 2]

        depth[depth <= 0] = 0
        depth[depth > 3000] = 0

        return depth

    def process_image_pair(
        self,
        left_path,
        right_path,
        output_dir=None,
        depth_scale=1.0,
        save_visualization=False,
        save_debug=False,
    ):
        img_left = cv2.imread(left_path)
        img_right = cv2.imread(right_path)

        if img_left is None or img_right is None:
            print(f"ERROR: Failed to load images: {left_path}, {right_path}")
            return None, None

        rect_left = cv2.remap(
            img_left,
            self.calibration.map_left_x,
            self.calibration.map_left_y,
            cv2.INTER_LINEAR,
        )
        rect_right = cv2.remap(
            img_right,
            self.calibration.map_right_x,
            self.calibration.map_right_y,
            cv2.INTER_LINEAR,
        )

        disparity = self.compute_disparity(rect_left, rect_right)

        depth = self.disparity_to_depth(disparity)

        if output_dir:
            base_name = (
                os.path.basename(left_path)
                .replace("left_", "")
                .replace("_L", "")
                .replace(".png", "")
            )

            depth_scaled = (depth * depth_scale).astype(np.float32)
            depth_scaled = np.clip(depth_scaled, 0, 65535)
            depth_uint16 = depth_scaled.astype(np.uint16)
            cv2.imwrite(
                os.path.join(output_dir, f"{base_name}_depth.png"), depth_uint16
            )

            np.save(os.path.join(output_dir, f"{base_name}_depth.npy"), depth)

            if save_debug:
                cv2.imwrite(
                    os.path.join(output_dir, f"{base_name}_rectified_left.png"),
                    rect_left,
                )
                cv2.imwrite(
                    os.path.join(output_dir, f"{base_name}_rectified_right.png"),
                    rect_right,
                )

                disp_normalized = cv2.normalize(
                    disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
                )
                cv2.imwrite(
                    os.path.join(output_dir, f"{base_name}_disparity.png"),
                    disp_normalized,
                )

        return disparity, depth

    def process_directory(
        self,
        image_dir,
        output_dir=None,
        depth_scale=1.0,
        save_visualization=False,
        save_debug=False,
    ):
        left_images = sorted(glob.glob(os.path.join(image_dir, "*_L.png")))
        right_images = sorted(glob.glob(os.path.join(image_dir, "*_R.png")))

        if len(left_images) == 0 or len(right_images) == 0:
            print(f"ERROR: No images found in {image_dir}")
            return []

        if len(left_images) != len(right_images):
            print("WARNING: Mismatched number of images!")
            print(f"  Left: {len(left_images)}, Right: {len(right_images)}")

        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"\nProcessing {len(left_images)} stereo pairs...")
        if self.use_wls_filter:
            print(
                f"WLS Filter enabled: lambda={self.wls_lambda}, sigma={self.wls_sigma}"
            )

        results = []

        for idx, (left_path, right_path) in enumerate(zip(left_images, right_images)):
            left_name = os.path.basename(left_path)
            right_name = os.path.basename(right_path)

            print(f"[{idx+1}/{len(left_images)}] Processing: {left_name}, {right_name}")

            disparity, depth = self.process_image_pair(
                left_path,
                right_path,
                output_dir=output_dir,
                depth_scale=depth_scale,
                save_visualization=save_visualization,
                save_debug=save_debug,
            )

            if disparity is not None:
                results.append(
                    {
                        "left_path": left_path,
                        "right_path": right_path,
                        "disparity": disparity,
                        "depth": depth,
                    }
                )

                valid_depth = depth[depth > 0]
                if len(valid_depth) > 0:
                    print(
                        f"  Depth range: {valid_depth.min():.2f} - {valid_depth.max():.2f} units"
                    )
                    print(f"  Mean depth: {valid_depth.mean():.2f} units")

        print(f"\n{'='*60}")
        print(f"Successfully processed {len(results)}/{len(left_images)} image pairs")
        print(f"{'='*60}\n")

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate depth maps from stereo images using calibration data"
    )

    parser.add_argument(
        "--calibration",
        type=str,
        required=True,
        help="Path to calibration file (.npz)",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        required=True,
        help="Directory containing *_L.png and *_R.png images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="depth_output",
        help="Output directory for depth maps (default: depth_output)",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1.0,
        help="Scaling factor for depth values (default: 1.0)",
    )
    parser.add_argument(
        "--num-disparities",
        type=int,
        default=160,
        help="Maximum disparity (must be divisible by 16, default: 160)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=5,
        help="Block size for matching (odd number, default: 5)",
    )
    parser.add_argument(
        "--min-disparity",
        type=int,
        default=0,
        help="Minimum disparity (default: 0)",
    )
    parser.add_argument(
        "--uniqueness",
        type=int,
        default=5,
        help="Uniqueness ratio (default: 5, higher = more strict)",
    )
    parser.add_argument(
        "--speckle-window",
        type=int,
        default=50,
        help="Speckle window size (default: 50)",
    )
    parser.add_argument(
        "--speckle-range",
        type=int,
        default=2,
        help="Speckle range (default: 2)",
    )
    parser.add_argument(
        "--use-wls-filter",
        action="store_true",
        help="Enable WLS filter for better depth map quality",
    )
    parser.add_argument(
        "--wls-lambda",
        type=float,
        default=8000.0,
        help="WLS filter lambda parameter (default: 8000.0, higher = smoother)",
    )
    parser.add_argument(
        "--wls-sigma",
        type=float,
        default=1.5,
        help="WLS filter sigma parameter (default: 1.5, lower = sharper edges)",
    )
    parser.add_argument(
        "--save-visualization",
        action="store_true",
        help="Save color-coded depth visualization images",
    )
    parser.add_argument(
        "--save-debug",
        action="store_true",
        help="Save debug outputs (numpy arrays, disparity, rectified images, etc.)",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.calibration):
        print(f"ERROR: Calibration file not found: {args.calibration}")
        return 1

    if not os.path.isdir(args.image_dir):
        print(f"ERROR: Image directory not found: {args.image_dir}")
        return 1

    if args.num_disparities % 16 != 0:
        print(
            f"ERROR: num-disparities must be divisible by 16 (got {args.num_disparities})"
        )
        return 1

    if args.block_size % 2 == 0:
        print(f"ERROR: block-size must be odd (got {args.block_size})")
        return 1

    print("\n" + "=" * 60)
    print("DEPTH MAP GENERATION")
    print("=" * 60)
    print(f"Calibration file: {args.calibration}")
    print(f"Image directory: {args.image_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Depth scale: {args.depth_scale}")
    if args.use_wls_filter:
        print(f"WLS Filter: ENABLED (lambda={args.wls_lambda}, sigma={args.wls_sigma})")
    else:
        print("WLS Filter: DISABLED")
    print("=" * 60 + "\n")

    try:
        generator = DepthMapGenerator(args.calibration)

        if args.use_wls_filter:
            generator.enable_wls_filter(args.wls_lambda, args.wls_sigma)

        generator.update_stereo_params(
            numDisparities=args.num_disparities,
            blockSize=args.block_size,
            minDisparity=args.min_disparity,
            uniquenessRatio=args.uniqueness,
            speckleWindowSize=args.speckle_window,
            speckleRange=args.speckle_range,
        )

        results = generator.process_directory(
            args.image_dir,
            output_dir=args.output_dir,
            depth_scale=args.depth_scale,
            save_visualization=args.save_visualization,
            save_debug=args.save_debug,
        )

        if len(results) > 0:
            print("=" * 60)
            print("DEPTH MAP GENERATION COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"\nOutputs saved to: {args.output_dir}")
            return 0
        else:
            print("ERROR: No depth maps were generated!")
            return 1

    except Exception as e:
        print(f"\nERROR during depth map generation: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
