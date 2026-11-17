# Ferris Lab — Monocular Human–Object Metric 3D Distance Estimation

CPU-first research prototype for detecting a person and a second object (initially a
bottle), estimating metric depth from a **single RGB webcam**, back-projecting
segmented visible surfaces to 3D, and estimating their relative XYZ displacement
and separation. Includes **post-run 3D point-cloud MP4 visualizations** with 12
camera viewpoints.

> **Research prototype — not safety-rated.** Predicted depth is not measured
> depth, and the nearest surface distances can be badly distorted by prediction
> errors, segmentation boundaries, and occlusion. Do **not** use for protective
> robot stopping distances, collision avoidance, or other safety decisions.

## How it works

1. YOLO11n-seg (`yolo11n-seg.pt`) segments COCO `person` (class 0) and `bottle` (class 39).
2. Ultralytics ByteTrack associates detections through time using `persist=True`.
3. Depth Anything V2 **Metric Indoor Small** predicts optical-axis depth (meters).
4. The per-instance segmentation masks select depth pixels; 3D back-projection
   uses undistorted rays from the calibrated webcam: `P=(x_normalized*Z, y_normalized*Z, Z)`.
5. Component-wise medians summarize visible 3D points for each object. The
   person-to-bottle representative distance is the Euclidean distance between
   these summaries. A KD-tree also measures sampled visible-surface proximity.
6. Optional compressed `.npz` files record per-frame visible points and RGB
   colors; a separate renderer creates the 12-view 3D MP4s **after recording**.

**Important:** Component-wise median 3D points are not body centers. The nearest
sampled surface pair is not necessarily the true closest physical pair. This
pipeline does not reconstruct hidden surfaces, produce a persistent SLAM map,
or measure ground-truth distance automatically.

## Repository layout

```text
Ferris_Lab_GitHub_Ready/
├── README.md
├── requirements.txt
├── LICENSE                 # AGPL-3.0
├── .gitignore
├── .github/workflows/syntax.yml
├── calibration/
│   └── README.md           # Each user calibrates their OWN camera
├── models/
│   └── README.md           # Weights downloaded on first run
├── src/
│   ├── __init__.py
│   ├── calibrate_webcam.py
│   ├── live_metric_depth.py
│   ├── checkerboard_depth_calibration.py
│   ├── live_two_object_depth.py
│   ├── live_two_object_depth_record3d.py
│   └── render_person_bottle_3d.py
└── tests/
    └── test_pipeline_io.py
```

Runtime folders (for example `results/<session>/`, camera `.npz`, checkpoints
and MP4 files) are **not** tracked by Git.

## Installation

Python **3.10 or 3.11** is recommended. These are recommended setup versions,
not a claim that the exact combination is validated on every operating system.
Examples assume macOS/Linux and a terminal opened in the repository root.
Windows users can use the equivalent `python` and virtual-environment commands.

**For new users (new, isolated environment):**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For macOS ARM64 or another specialized platform, install an appropriate CPU
PyTorch build using [official PyTorch instructions](https://pytorch.org/get-started/locally/)
if the generic `requirements.txt` install does not work. This project defaults
to `--device cpu`; MPS is **not needed** and should not be assumed stable.

**For contributors who already have a working environment:** keep it. Do not
blindly reinstall PyTorch, Ultralytics, or Transformers. Install only missing
packages, such as `python -m pip install imageio-ffmpeg` for MP4 rendering.

Pretrained checkpoints are fetched automatically on the first model run;
internet access is needed then. Model weights are not bundled in this repo.

## 1. Calibrate YOUR webcam

For 3D coordinates, use the **same camera and 1280x720 capture mode** for
calibration and inference. You need a printed checkerboard with **10x7 inner
corners** (11x8 squares) and known **25 mm squares**. If your printed board
has a different layout or square size, update the CLI values.

```bash
python src/calibrate_webcam.py \
  --camera 0 --width 1280 --height 720 \
  --inner-cols 10 --inner-rows 7 --square-size-mm 25 \
  --min-samples 20
```

Press **SPACE** to capture each clearly detected board pose and **Q** to
calibrate and quit. Collect at least 20 *different* positions, tilts and
camera distances, with board corners visible across the field of view.
Output: `calibration/webcam_calibration.npz`.

You can also pass an existing valid calibration matching the webcam and mode
with `--calibration /path/to/your_calibration.npz`.

Camera intrinsics alone do **not** calibrate the metric-depth model. The model's
absolute-depth bias needs an independently measured validation dataset.

## 2. Test person segmentation and metric depth

```bash
python src/live_metric_depth.py \
  --mode both --device cpu \
  --width 1280 --height 720 \
  --depth-width 336 --depth-height 252 --limit 120
```

Useful modes: `--mode yolo` (segmentation only), `--mode depth` (depth only),
`--mode both` (both). This initial script visualizes person masks and model
metric depth only; it does not estimate person–bottle separation.

## 3. Track person and bottle and measure 3D separation

```bash
python src/live_two_object_depth.py \
  --mode both --device cpu --limit 120 \
  --calibration calibration/webcam_calibration.npz
```

Output: `results/<timestamp>/person_bottle_tracking.csv`.

**Current implementation constraints:** exactly one selected person and one
selected bottle are measured per frame, even when other instances are visible.
Selected IDs may reset after missing detections, and distance estimates are
unfiltered. Default max 1,800 sampled 3D points per object.

## 4. Record 3D point clouds for later animations

This is a **separate version of the tracker**, so the original pipeline stays
available unchanged:

```bash
python src/live_two_object_depth_record3d.py \
  --mode both --device cpu --limit 120 \
  --calibration calibration/webcam_calibration.npz
```

The terminal prints the new `results/<timestamp_microseconds>` directory.
It contains the normal CSV plus `pointcloud_frames/frame_XXXXXX.npz` and
`recording_metadata.json`. Each NPZ stores sampled visible XYZ points (meters),
matching RGB colors, representative points, tracking IDs, and distance values.
The experiment may run more slowly due to extra disk writes.

You may reduce saved point-cloud frequency with `--record-every 2`, for example.
The CSV still logs every processed frame.

## 5. Render 3D MP4s AFTER the experiment

Replace `YOUR_SESSION` with the exact recording directory printed by step 4.
A CSV from `live_two_object_depth.py` **cannot** render per-frame point clouds
because it does not contain them.

```bash
# The two recommended viewpoints (upright isometric + from camera)
python src/render_person_bottle_3d.py \
  "results/YOUR_SESSION" --only 03,07

# All 12 viewpoints
python src/render_person_bottle_3d.py \
  "results/YOUR_SESSION" --all-views

# Comparison PNG of all 12 viewpoints, without rendering MP4s
python src/render_person_bottle_3d.py \
  "results/YOUR_SESSION" --preview-only

# Render the original RGB colors instead of class colors
python src/render_person_bottle_3d.py \
  "results/YOUR_SESSION" --only 03,07 --color-mode rgb
```

Outputs: `results/YOUR_SESSION/3d_visualizations/`, including
`view_comparison.png`, `view_guide.txt`, and selected `*.mp4` files.

Views **01–02** use native OpenCV camera axes (X right, Y down, Z forward).
Views **03–12** use upright plotting axes (X right, forward depth, height up).
The geometry and numeric distances remain in the original camera frame;
only the visualization axes are permuted/flipped. Plot limits stay fixed for
the session, sampled trajectories break on detection loss or tracking-ID change,
and video timing uses recorded timestamps with sample-and-hold (no fabricated
intermediate 3D poses).

### Render problems

- `ModuleNotFoundError: No module named 'imageio_ffmpeg'`:
  `python -m pip install imageio-ffmpeg` in the environment running the script.
- `This session has no recorded 3D frames`: use `live_two_object_depth_record3d.py`,
  not the older logger.
- `Missing calibration file`: run step 1 or pass your exact calibration path.
- `Actual camera frame differs from calibration`: use the **same physical camera**,
  resolution, crop, and mode used to calibrate.
- `MPS` crashes/instability: run explicitly with `--device cpu`.

## 6. Checkerboard depth accuracy experiment (optional)

This is a separate depth-validation/calibration *experiment* and is **not**
automatically applied to person/bottle depth:

```bash
python src/checkerboard_depth_calibration.py \
  --device cpu --limit 120 \
  --calibration calibration/webcam_calibration.npz
```

Output includes `checkerboard_depth_frames.csv` and `board_scale_summary.json`.
The rolling offsets and scales are exploratory. Do **not** transfer a fitted
correction into human or bottle estimates without **independent** measurements.

## Experimental observations (single setup; not universal performance)

On a MacBook Pro M2 Pro using CPU and 336x252 depth inference, the earlier
unmodified prototype reached roughly **7.1 processed FPS** without 3D
recording, and approximately **5.4–5.8 reported processed FPS** with 3D
tracking. Timestamp-derived throughput in one 120-frame stationary trial was
about **4.7 FPS**, indicating some overhead omitted by the displayed FPS.

In that single stationary person-and-bottle run, representative separation was
approximately **1.18 m**, with a **5 cm standard deviation** despite no
intentional movement. Visible nearest-point separation was much smaller. The
true physical separation was **not independently measured**, so these are
*stability observations*, **not** accuracy claims or universal benchmarks.

## Run basic tests

The tests use synthetic geometry and no camera or model downloads:

```bash
python -m unittest discover -s tests -v
python -m compileall -q src
```

These do **not** replace a real-camera, end-to-end validation.

## Limitations and next steps

- Monocular metric depth has domain-dependent bias and error; adjacent objects
  may receive inconsistent depth even while stationary.
- Bicubic depth upsampling may blend foreground/background at mask boundaries.
- Representative XYZ medians are convenient point summaries, not object centroids.
- Raw nearest-neighbor distances are sensitive to outliers and sparse sampling.
- Tracking may lose/reassign IDs during occlusion, and missing positions are
  not extrapolated.
- Frame-rate logging does not guarantee end-to-end capture-to-display latency.
- Object class 39 is **bottle** in COCO, not a generic robot/cobot class.
- Do not use for robot safety/control without appropriate independent sensing,
  validation, and a separately certified safety architecture.

Future improvements: depth edge filtering, robust closest-distance quantiles,
uncertainty reporting, independent ground truth, and latency profiling.

## Dependencies, credits, and licensing

This repo is provided under the **GNU Affero General Public License v3.0**
(`LICENSE`). Ultralytics YOLO is also offered under AGPL-3.0 or a commercial
Enterprise license; review [Ultralytics' terms](https://docs.ultralytics.com/help/contributing/)
for obligations when distributing or deploying applications built with YOLO.

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) — pretrained
  YOLO11 instance segmentation and ByteTrack integration.
- [Depth Anything V2 metric indoor model](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf)
  — model authors and checkpoint usage terms apply separately.
- [OpenCV](https://opencv.org/) — camera calibration and geometric back-projection.
- [SciPy](https://scipy.org/) — KD-tree nearest neighbor queries.
- [Matplotlib](https://matplotlib.org/) and
  [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg) — post-run MP4s.

**No model checkpoints, personalized calibration files, or experiment videos
are redistributed in this ZIP.** Review third-party model licenses before
uploading weights or publishing derivatives.
