#!/usr/bin/env python3
# Copyright (C) 2026 Nitai Shah
# SPDX-License-Identifier: AGPL-3.0-only
"""Create a camera calibration .npz using printed checkerboard observations.

The target is the SAME camera and video mode used for metric depth recording.
This is intrinsic calibration, distinct from validating the depth model's scale.

Default board: 10x7 INNER corners (11x8 squares), 25 mm square side.
Press SPACE to add a valid, varied viewpoint; Q or ESC to finish.
"""

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--inner-cols", type=int, default=10)
    p.add_argument("--inner-rows", type=int, default=7)
    p.add_argument("--square-size-mm", type=float, default=25.0)
    p.add_argument("--min-samples", type=int, default=15)
    p.add_argument("--output", type=Path, default=ROOT / "calibration" / "webcam_calibration.npz")
    return p.parse_args()


def board_object_points(cols, rows, square_size_mm):
    xyz = np.zeros((cols * rows, 3), dtype=np.float32)
    xyz[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * (square_size_mm / 1000.0)
    return xyz


def detect_board(gray, pattern):
    found = False
    corners = None
    if hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(
            gray, pattern, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        found, corners = cv2.findChessboardCorners(
            gray, pattern,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if found:
            cv2.cornerSubPix(gray, corners, (9, 9), (-1, -1),
                             (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
    if not found or corners is None or len(corners) != pattern[0] * pattern[1]:
        return None
    return corners.astype(np.float32).reshape(-1, 1, 2)


def calibrate_samples(image_points, image_size, object_points):
    if len(image_points) < 3:
        raise ValueError("At least 3 distinct checkerboard views are required")
    objects = [object_points.copy() for _ in image_points]
    rms, K, dist, rotations, translations = cv2.calibrateCamera(
        objects, image_points, image_size, None, None)
    if not np.isfinite(rms) or not np.isfinite(K).all() or not np.isfinite(dist).all():
        raise RuntimeError("Camera calibration returned non-finite parameters")
    return rms, K, dist


def main():
    args = parse_args()
    if min(args.width, args.height, args.inner_cols, args.inner_rows, args.min_samples) < 2:
        raise ValueError("Image dimensions, corner counts and min-samples must be >= 2")
    if args.square_size_mm <= 0:
        raise ValueError("Square size must be positive")
    pattern = (args.inner_cols, args.inner_rows)
    points_3d = board_object_points(*pattern, args.square_size_mm)

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError("Could not open webcam; check --camera and OS camera permissions")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    measurements = []
    try:
        print("Use the same physical webcam and resolution as inference.")
        print("Show the WHOLE board. Vary position, orientation, depth and image corners.")
        print("SPACE: capture detected corners | Q or ESC: finish")
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Webcam did not deliver a frame")
            h, w = frame.shape[:2]
            if (w, h) != (args.width, args.height):
                raise RuntimeError(
                    f"Camera delivers {w}x{h}, but {args.width}x{args.height} was requested. "
                    "Re-run with the actual camera resolution and use it in the tracker.")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners = detect_board(gray, pattern)
            view = frame.copy()
            if corners is not None:
                cv2.drawChessboardCorners(view, pattern, corners, True)
            text = (f"Captured {len(measurements)} / {args.min_samples}  | "
                    f"board {'found' if corners is not None else 'NOT found'}")
            cv2.putText(view, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 220, 0) if corners is not None else (0, 0, 255), 2)
            cv2.imshow("Webcam intrinsic calibration - SPACE capture / Q finish", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                if corners is None:
                    print("Skipped: full board corners not detected", flush=True)
                else:
                    measurements.append(corners.copy())
                    print(f"Saved view {len(measurements)}", flush=True)
            elif key in (ord("q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    if len(measurements) < args.min_samples:
        raise SystemExit(
            f"Only {len(measurements)} valid views; need at least {args.min_samples}. "
            "Nothing saved. Re-run and capture varied full-board views.")
    rms, K, dist = calibrate_samples(measurements, (args.width, args.height), points_3d)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, K=K, dist=dist,
                        image_size=np.array((args.width, args.height), dtype=np.int32),
                        rms=np.array(rms), num_views=np.array(len(measurements)),
                        square_size_m=np.array(args.square_size_mm / 1000.0))
    print(f"Camera intrinsics saved: {args.output}")
    print(f"Views: {len(measurements)} | RMS reprojection error: {rms:.3f} px")
    print("K:\n", K)
    print("Distortion coefficients:", dist.ravel())
    if rms > 2.5:
        print("WARNING: high reprojection RMS; recapture more varied, sharp board views.", file=sys.stderr)
    print("NOTE: This calibrates camera geometry, NOT learned metric-depth accuracy.")


if __name__ == "__main__":
    main()
