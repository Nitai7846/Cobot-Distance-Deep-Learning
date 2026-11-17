#!/usr/bin/env python3
# Copyright (C) 2026 Nitai Shah
# SPDX-License-Identifier: AGPL-3.0-only
"""Checkerboard-referenced metric depth experiment (not safety-rated).

Use one real camera calibrated with webcam_calibration.npz (keys K, dist,
image_size). The 11x8-square board has 10x7 inner corners, 25 mm squares.

For every frame:
  1. YOLO11n-seg finds person pixels.
  2. Depth Anything V2 Metric Indoor Small predicts optical-axis depth Z (m).
  3. OpenCV detects a 10x7 checkerboard and solvePnP recovers its pose.
  4. Compute actual Z (m) at EACH sampled board pixel using the fitted plane,
     taking camera distortion into account.
  5. Compare depths at the same board pixels; derive rolling additive bias
     and multiplicative scale factors.
  6. Apply both exploratory corrections to person-mask pixels and log everything.

A board held in front of the torso is NOT at the same physical depth as the torso.
The correction transfers a *model scale estimate*, not board ground truth, to
human pixels. Single-board scaling can still be wrong for another object/depth.

Run with --device cpu on macOS Sonoma (14.6.1); existing newer PyTorch crashes
on MPS. No package changes required.
"""

import argparse
import csv
import json
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from live_metric_depth import (
    PROJECT_ROOT,
    DEPTH_MODEL_ID,
    get_people_mask,
    depth_to_colormap,
    overlay_person_mask,
    text_line,
)

PATTERN = (10, 7)  # inner corners: 11x8 squares
SQUARE_SIZE_M = 0.025


def args_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--calibration", type=Path, required=True,
                   help="webcam_calibration.npz from the SAME physical camera and video mode")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1280, help="Must match camera calibration's width")
    p.add_argument("--height", type=int, default=720, help="Must match camera calibration's height")
    p.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    p.add_argument("--depth-width", type=int, default=336)
    p.add_argument("--depth-height", type=int, default=252)
    p.add_argument("--yolo-size", type=int, default=416)
    p.add_argument("--max-display-depth", type=float, default=5.0)
    p.add_argument("--max-reproj-px", type=float, default=3.0)
    p.add_argument("--board-sample-step", type=int, default=4,
                   help="Use every Nth pixel inside board ROI for 3D comparison")
    p.add_argument("--scale-window", type=int, default=50,
                   help="Accepted checkerboard frames used for rolling median scale")
    p.add_argument("--limit", type=int, default=0, help="0 runs until q or Ctrl-C")
    p.add_argument("--no-display", action="store_true")
    return p.parse_args()


def load_calibration(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Calibration not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        if {"K", "dist", "image_size"}.issubset(keys):
            K, dist = data["K"], data["dist"]
        elif {"camera_matrix", "dist_coeffs", "image_size"}.issubset(keys):
            K, dist = data["camera_matrix"], data["dist_coeffs"]
        else:
            raise ValueError(
                f"Unexpected calibration keys {sorted(keys)}. Need K/dist/image_size "
                "(or camera_matrix/dist_coeffs/image_size)."
            )
        size = tuple(int(x) for x in np.asarray(data["image_size"]).ravel())
        rms = float(np.asarray(data["rms"]).ravel()[0]) if "rms" in keys else None
    K = np.asarray(K, dtype=np.float64)
    dist = np.asarray(dist, dtype=np.float64).reshape(-1, 1)
    if K.shape != (3, 3) or len(size) != 2:
        raise ValueError("Invalid K or image_size in calibration file.")
    return K, dist, size, rms


def object_corners_m():
    xyz = np.zeros((PATTERN[0] * PATTERN[1], 3), dtype=np.float64)
    xyz[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQUARE_SIZE_M
    return xyz


def find_board(gray, K, dist, obj):
    """Return corners, rvec, tvec, reprojection RMSE; None if unreliable."""
    ok = False
    corners = None
    if hasattr(cv2, "findChessboardCornersSB"):
        ok, corners = cv2.findChessboardCornersSB(gray, PATTERN, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        ok, corners = cv2.findChessboardCorners(gray, PATTERN, flags=flags)
        if ok:
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
            cv2.cornerSubPix(gray, corners, (9, 9), (-1, -1), criteria)
    if not ok or corners is None or len(corners) != len(obj):
        return None

    corners = corners.reshape(-1, 1, 2).astype(np.float32)
    solved, rvec, tvec = cv2.solvePnP(obj, corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not solved or not np.all(np.isfinite(tvec)) or float(tvec[2, 0]) <= 0:
        return None
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    rmse = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - corners.reshape(-1, 2)) ** 2, axis=1))))
    return corners, rvec, tvec, rmse


def evaluate_board_depth(depth, corners, rvec, tvec, K, dist, step):
    """Compare depth map to exact plane-intersection Z at visible interior board pixels."""
    H, W = depth.shape
    hull = cv2.convexHull(np.round(corners.reshape(-1, 2)).astype(np.int32))
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    # Avoid boundary mixing with hands, clothing, and background.
    mask = cv2.erode(mask, np.ones((7, 7), dtype=np.uint8), iterations=1)
    ys_sub, xs_sub = np.where(mask[::step, ::step] > 0)
    xs, ys = xs_sub * step, ys_sub * step
    if len(xs) < 25:
        return None
    pred = depth[ys, xs].astype(np.float64)
    uv = np.stack((xs, ys), axis=1).astype(np.float64).reshape(-1, 1, 2)
    undistorted = cv2.undistortPoints(uv, K, dist).reshape(-1, 2)
    rays = np.column_stack((undistorted, np.ones(len(xs), dtype=np.float64)))
    R, _ = cv2.Rodrigues(rvec)
    normal = R[:, 2].reshape(3)
    point_on_board = tvec.reshape(3)
    numerator = float(normal @ point_on_board)
    denominator = rays @ normal
    with np.errstate(divide="ignore", invalid="ignore"):
        true_z = numerator / denominator
    valid = np.isfinite(pred) & (pred > 0.05) & (pred < 20) & np.isfinite(true_z) & (true_z > 0)
    if np.count_nonzero(valid) < 25:
        return None
    pred, true_z = pred[valid], true_z[valid]
    ratios = true_z / pred
    # Remove physically implausible ratios and gross outliers before estimating scale.
    in_range = (ratios > 0.25) & (ratios < 4.0)
    if np.count_nonzero(in_range) < 25:
        return None
    pred, true_z, ratios = pred[in_range], true_z[in_range], ratios[in_range]
    center_m = R @ np.array([(PATTERN[0] - 1) * SQUARE_SIZE_M / 2,
                              (PATTERN[1] - 1) * SQUARE_SIZE_M / 2, 0.0]) + point_on_board
    return {
        "roi_mask": mask.astype(bool),
        "n_points": int(len(pred)),
        "board_gt_z_m": float(np.median(true_z)),
        "board_gt_center_z_m": float(center_m[2]),
        "board_gt_center_range_m": float(np.linalg.norm(center_m)),
        "board_depth_pred_m": float(np.median(pred)),
        "board_depth_error_m": float(np.median(pred - true_z)),
        "board_depth_mae_m": float(np.median(np.abs(pred - true_z))),
        "scale_ratio": float(np.median(ratios)),
    }


def median_valid(depth, mask):
    valid = mask & np.isfinite(depth) & (depth > 0.05)
    if not np.any(valid):
        return float("nan"), float("nan"), 0
    values = depth[valid]
    return float(np.median(values)), float(np.percentile(values, 10)), int(values.size)


def fmt(value, precision=2):
    return f"{value:.{precision}f}" if np.isfinite(value) else "n/a"


def main():
    a = args_parser()
    if a.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available; use --device cpu")
    if min(a.depth_width, a.depth_height, a.width, a.height, a.board_sample_step) < 1:
        raise ValueError("Image dimensions and board sampling step must be positive")
    if a.scale_window < 1:
        raise ValueError("--scale-window must be at least 1")

    K, dist, calibration_size, rms = load_calibration(a.calibration)
    if (a.width, a.height) != calibration_size:
        raise ValueError(
            f"Requested camera size {(a.width, a.height)} != calibration size {calibration_size}. "
            "Use the calibrated camera and pass matching --width/--height. "
            "Changing camera, aspect ratio or cropping invalidates the reference."
        )
    print(f"Calibration: {a.calibration} | image_size={calibration_size} | RMS={rms}", flush=True)
    print(f"Board: {PATTERN[0]}x{PATTERN[1]} inner corners, {SQUARE_SIZE_M * 1000:g} mm squares", flush=True)
    print(f"Device: {a.device}; torch={torch.__version__}", flush=True)
    if rms is not None and rms > 2.5:
        print("WARNING: calibration RMS > 2.5 px. Board pose accuracy may be limited.", flush=True)

    from ultralytics import YOLO
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    yolo_file = PROJECT_ROOT / "models" / "yolo11n-seg.pt"
    yolo = YOLO(str(yolo_file if yolo_file.exists() else "yolo11n-seg.pt"))
    processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
    depth_net = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID).to(a.device).eval()
    print("Depth model type:", getattr(depth_net.config, "depth_estimation_type", "unknown"), flush=True)

    cap = cv2.VideoCapture(a.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Camera index {a.camera} not available")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    ok, check_frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Camera opened but could not read a frame")
    H, W = check_frame.shape[:2]
    if (W, H) != calibration_size:
        cap.release()
        raise ValueError(
            f"Actual webcam output {(W, H)} != calibrated {calibration_size}. "
            "Try selecting your calibrated webcam using --camera, or request its calibrated resolution."
        )

    run_dir = PROJECT_ROOT / "results" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "checkerboard_depth_frames.csv"
    summary_path = run_dir / "board_scale_summary.json"
    fields = [
        "timestamp", "frame", "fps_rolling", "yolo_ms", "depth_ms", "board_detect_ms", "total_ms", "people",
        "board_detected", "pose_rmse_px", "board_points", "board_gt_z_m", "board_gt_center_z_m",
        "board_gt_center_range_m", "board_depth_pred_m", "board_depth_error_m", "board_depth_mae_m",
        "scale_frame", "scale_rolling", "offset_frame_m", "offset_rolling_m", "scale_age_frames", "scale_n_samples", "person_pred_median_m",
        "person_pred_p10_m", "person_scaled_median_m", "person_scaled_p10_m", "person_offset_median_m", "person_offset_p10_m", "person_pixels",
    ]
    print(f"Saving per-frame log: {csv_path}", flush=True)
    print("Controls: q=quit, ESC=quit | A scale from the board is a calibration estimate, NOT true torso depth.", flush=True)

    obj = object_corners_m()
    scale_history = deque(maxlen=a.scale_window)
    offset_history = deque(maxlen=a.scale_window)
    fps_history = deque(maxlen=30)
    accepted_ratios = []
    accepted_offsets = []
    accepted_pairs = []
    last_good_board_frame = None
    frame_idx = 0
    frame_buffer = check_frame

    try:
        with csv_path.open("w", newline="", buffering=1) as out:
            writer = csv.DictWriter(out, fieldnames=fields)
            writer.writeheader()
            while True:
                start_total = time.perf_counter()
                if frame_buffer is not None:
                    frame = frame_buffer
                    frame_buffer = None
                else:
                    ok, frame = cap.read()
                    if not ok:
                        print("Camera stopped delivering frames", file=sys.stderr)
                        break
                frame_idx += 1

                start = time.perf_counter()
                result = yolo.predict(source=frame, imgsz=a.yolo_size, device=a.device,
                                      classes=[0], conf=0.3, retina_masks=True, verbose=False)[0]
                if a.device == "mps":
                    torch.mps.synchronize()
                person_mask, people = get_people_mask(result, frame.shape)
                yolo_ms = (time.perf_counter() - start) * 1000

                start = time.perf_counter()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                batch = processor(images=Image.fromarray(rgb),
                                  size={"height": a.depth_height, "width": a.depth_width},
                                  return_tensors="pt")
                batch = {k: v.to(a.device) for k, v in batch.items()}
                with torch.inference_mode():
                    pred = depth_net(**batch).predicted_depth
                    expanded = torch.nn.functional.interpolate(
                        pred.unsqueeze(1), size=(H, W), mode="bicubic", align_corners=False
                    ).squeeze(1).squeeze(0)
                depth = expanded.float().cpu().numpy()
                depth_ms = (time.perf_counter() - start) * 1000

                start = time.perf_counter()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                pose = find_board(gray, K, dist, obj)
                pose_rmse = float("nan")
                board = None
                if pose is not None:
                    corners, rvec, tvec, pose_rmse = pose
                    if pose_rmse <= a.max_reproj_px:
                        board = evaluate_board_depth(depth, corners, rvec, tvec, K, dist, a.board_sample_step)
                        if board is not None:
                            scale_history.append(board["scale_ratio"])
                            offset = -board["board_depth_error_m"]  # GT - predicted (same pixels)
                            offset_history.append(offset)
                            accepted_ratios.append(board["scale_ratio"])
                            accepted_offsets.append(offset)
                            accepted_pairs.append((board["board_depth_pred_m"], board["board_gt_z_m"]))
                            last_good_board_frame = frame_idx
                board_ms = (time.perf_counter() - start) * 1000

                scale = float(np.median(scale_history)) if scale_history else float("nan")
                offset = float(np.median(offset_history)) if offset_history else float("nan")
                age = (frame_idx - last_good_board_frame) if last_good_board_frame is not None else -1
                # Exclude the checkerboard region from the person segmentation mask when visible.
                person_only = person_mask & ~(board["roi_mask"] if board is not None else np.zeros((H, W), dtype=bool))
                human_med, human_p10, n_person_pixels = median_valid(depth, person_only)
                human_corrected = human_med * scale
                human_corrected_p10 = human_p10 * scale
                human_offset = human_med + offset
                human_offset_p10 = human_p10 + offset

                total_ms = (time.perf_counter() - start_total) * 1000
                fps_history.append(total_ms)
                fps = 1000.0 / float(np.mean(fps_history))
                row = {
                    "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                    "frame": frame_idx, "fps_rolling": round(fps, 2),
                    "yolo_ms": round(yolo_ms, 1), "depth_ms": round(depth_ms, 1),
                    "board_detect_ms": round(board_ms, 1), "total_ms": round(total_ms, 1),
                    "people": people, "board_detected": int(board is not None),
                    "pose_rmse_px": round(pose_rmse, 3) if np.isfinite(pose_rmse) else "",
                    "board_points": board["n_points"] if board is not None else "",
                    "board_gt_z_m": round(board["board_gt_z_m"], 4) if board is not None else "",
                    "board_gt_center_z_m": round(board["board_gt_center_z_m"], 4) if board is not None else "",
                    "board_gt_center_range_m": round(board["board_gt_center_range_m"], 4) if board is not None else "",
                    "board_depth_pred_m": round(board["board_depth_pred_m"], 4) if board is not None else "",
                    "board_depth_error_m": round(board["board_depth_error_m"], 4) if board is not None else "",
                    "board_depth_mae_m": round(board["board_depth_mae_m"], 4) if board is not None else "",
                    "scale_frame": round(board["scale_ratio"], 5) if board is not None else "",
                    "scale_rolling": round(scale, 5) if np.isfinite(scale) else "",
                    "offset_frame_m": round(-board["board_depth_error_m"], 4) if board is not None else "",
                    "offset_rolling_m": round(offset, 4) if np.isfinite(offset) else "",
                    "scale_age_frames": age if age >= 0 else "",
                    "scale_n_samples": len(scale_history),
                    "person_pred_median_m": round(human_med, 4) if np.isfinite(human_med) else "",
                    "person_pred_p10_m": round(human_p10, 4) if np.isfinite(human_p10) else "",
                    "person_scaled_median_m": round(human_corrected, 4) if np.isfinite(human_corrected) else "",
                    "person_scaled_p10_m": round(human_corrected_p10, 4) if np.isfinite(human_corrected_p10) else "",
                    "person_offset_median_m": round(human_offset, 4) if np.isfinite(human_offset) else "",
                    "person_offset_p10_m": round(human_offset_p10, 4) if np.isfinite(human_offset_p10) else "",
                    "person_pixels": n_person_pixels,
                }
                writer.writerow(row)

                if not a.no_display:
                    left = overlay_person_mask(frame, person_mask)
                    if board is not None:
                        cv2.drawChessboardCorners(left, PATTERN, corners, True)
                        contour = cv2.convexHull(np.round(corners.reshape(-1, 2)).astype(np.int32))
                        cv2.polylines(left, [contour], True, (0, 255, 255), 2)
                    left_y = 28
                    text_line(left, f"{a.device}+YOLO+depth+checkerboard | FPS {fps:.1f}", left_y)
                    text_line(left, f"Human Z: raw {fmt(human_med)} | +offset {fmt(human_offset)} | xscale {fmt(human_corrected)} m", left_y + 29)
                    if board is not None:
                        text_line(left, f"Board GT Z: {fmt(board['board_gt_z_m'])}m | model Z: {fmt(board['board_depth_pred_m'])}m", left_y + 58)
                        text_line(left, f"Rolling offset: {fmt(offset,2)}m | scale: {fmt(scale,3)} | reproj: {fmt(pose_rmse,2)}px", left_y + 87)
                    elif np.isfinite(scale):
                        text_line(left, f"Board missing | LAST scale {fmt(scale,3)} (age {age} frames)", left_y + 58,
                                  (0, 200, 255))
                    else:
                        text_line(left, "Board not yet detected: show full 10x7 inner corners", left_y + 58,
                                  (0, 200, 255))
                    right = depth_to_colormap(depth, a.max_display_depth)
                    text_line(right, f"Predicted metric Z (m) | depth {depth_ms:.0f} ms", 28)
                    display = np.hstack([left, right])
                    # Resize only display; all pose/depth computations use original camera pixels.
                    if display.shape[1] > 1700:
                        display = cv2.resize(display, (display.shape[1] // 2, display.shape[0] // 2))
                    cv2.imshow("Checkerboard GT vs Metric Depth - q to quit", display)
                    if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                        break

                if frame_idx % 30 == 0:
                    print(
                        f"frames={frame_idx} FPS={fps:.1f} YOLO={yolo_ms:.0f}ms depth={depth_ms:.0f}ms "
                        f"board={board is not None} GT_Z={fmt(board['board_gt_z_m'] if board else float('nan'))}m "
                        f"model_Z={fmt(board['board_depth_pred_m'] if board else float('nan'))}m "
                        f"scale={fmt(scale,3)} offset={fmt(offset)}m human_raw={fmt(human_med)}m "
                        f"human_offset={fmt(human_offset)}m human_scaled={fmt(human_corrected)}m",
                        flush=True,
                    )
                if a.limit and frame_idx >= a.limit:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        summary = {
            "calibration_file": str(a.calibration),
            "camera": a.camera,
            "camera_size": [W, H],
            "board_inner_corners": list(PATTERN),
            "board_square_size_m": SQUARE_SIZE_M,
            "accepted_board_frames": len(accepted_ratios),
            "median_scale_all_accepted_frames": float(np.median(accepted_ratios)) if accepted_ratios else None,
            "median_additive_offset_m_all_accepted_frames": float(np.median(accepted_offsets)) if accepted_offsets else None,
            "scale_note": "Exploratory global scale calibration derived from board pixels, not validation of human depth.",
            "depth_model": DEPTH_MODEL_ID,
        }
        # Optional affine fit needs measurements from meaningfully different distances.
        if len(accepted_pairs) >= 30:
            pairs = np.asarray(accepted_pairs, dtype=np.float64)
            ground_truth_span = float(np.percentile(pairs[:, 1], 90) - np.percentile(pairs[:, 1], 10))
            summary["ground_truth_depth_p90_minus_p10_m"] = ground_truth_span
            if ground_truth_span >= 0.4 and float(np.ptp(pairs[:, 0])) >= 0.2:
                slope, intercept = np.polyfit(pairs[:, 0], pairs[:, 1], 1)
                summary["exploratory_affine_fit_ground_truth_equals_a_times_model_plus_b"] = {
                    "a": float(slope), "b_m": float(intercept),
                    "warning": "Fit on these same frames: must validate on separately recorded distances.",
                }
            else:
                summary["affine_fit_note"] = "Not enough distinct distances to fit a reliable slope/intercept. Move the checkerboard among different depths."
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Finished. CSV: {csv_path}", flush=True)
        print(f"Scale summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
