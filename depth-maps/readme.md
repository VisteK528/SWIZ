# Stereo Depth Map Generator

Stereo camera calibration and depth map generation for bin picking.

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

### 1. Calibrate

```bash
python calibrate.py --images "calibration images/" --width 7 --height 10
```

Outputs a `.npz` calibration file.

### 2. Generate depth maps

```bash
python depth_map_generator.py \
  --calibration calibration.npz \
  --image-dir "bin images/" \
  --output-dir output/
```

Input images must be named `*_L.png` / `*_R.png`.

#### Key options

| Flag | Default | Description |
|---|---|---|
| `--depth-scale` | 1.0 | Scale factor for output depth values |
| `--num-disparities` | 160 | Max disparity, must be divisible by 16 |
| `--block-size` | 5 | Matching block size (odd number) |
| `--use-wls-filter` | off | Enable WLS filter for smoother depth |
| `--save-debug` | off | Save rectified images and disparity maps |

### Outputs

Each stereo pair produces:
- `*_depth.png` — 16-bit depth image
- `*_depth.npy` — raw float32 depth array
