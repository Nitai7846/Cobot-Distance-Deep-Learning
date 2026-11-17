#!/usr/bin/env python3
# Copyright (C) 2026 Nitai Shah
# SPDX-License-Identifier: AGPL-3.0-only
"""YOLO11n instance segmentation + metric-depth 3D person-to-bottle tracking.

Research prototype only. Depth and distances are estimates from one monocular
camera, not ground truth or safety-rated separation measurements.

Requires the existing human_robot_depth environment and a camera calibration
matching the physical camera and EXACT requested capture resolution.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
PERSON_CLASS, BOTTLE_CLASS = 0, 39  # YOLO11 COCO labels
COLORS = {PERSON_CLASS: (70, 205, 70), BOTTLE_CLASS: (30, 160, 255)}  # BGR


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calibration", type=Path, default=PROJECT_ROOT / "calibration" / "webcam_calibration.npz")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--mode", choices=["yolo", "both"], default="both",
                   help="Start with yolo to verify person/bottle detection without loading depth")
    p.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    p.add_argument("--yolo-size", type=int, default=416)
    p.add_argument("--depth-width", type=int, default=336)
    p.add_argument("--depth-height", type=int, default=252)
    p.add_argument("--confidence", type=float, default=0.25)
    p.add_argument("--mask-erode", type=int, default=3, help="Mask erosion kernel pixels; 0 disables")
    p.add_argument("--sample-step", type=int, default=4, help="Pixel sampling stride for 3D points")
    p.add_argument("--max-points", type=int, default=1800, help="Maximum 3D points per object")
    p.add_argument("--min-depth", type=float, default=0.2)
    p.add_argument("--max-depth", type=float, default=8.0)
    p.add_argument("--max-display-depth", type=float, default=5.0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--no-display", action="store_true")
    return p.parse_args()


def read_calibration(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing calibration file: {path}")
    with np.load(path, allow_pickle=False) as d:
        if {"K", "dist", "image_size"}.issubset(d.files):
            K, dist = d["K"], d["dist"]
        elif {"camera_matrix", "dist_coeffs", "image_size"}.issubset(d.files):
            K, dist = d["camera_matrix"], d["dist_coeffs"]
        else:
            raise ValueError(f"Unsupported calibration keys: {d.files}")
        size = tuple(int(x) for x in np.asarray(d["image_size"]).ravel())
    K = np.asarray(K, dtype=np.float64)
    dist = np.asarray(dist, dtype=np.float64).reshape(-1)
    if K.shape != (3, 3) or len(size) != 2:
        raise ValueError("Invalid camera calibration K/image_size")
    return K, dist, size


def get_instances(result, shape):
    """Keep individual person/bottle masks and their (possibly absent) track IDs."""
    h, w = shape[:2]
    if result.boxes is None or result.masks is None:
        return []
    labels = result.boxes.cls.int().cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    masks = result.masks.data.cpu().numpy()
    ids = (result.boxes.id.int().cpu().numpy()
           if result.boxes.id is not None else [None] * len(labels))
    output = []
    for class_id, confidence, tracking_id, raw_mask in zip(labels, confs, ids, masks):
        class_id = int(class_id)
        if class_id not in (PERSON_CLASS, BOTTLE_CLASS):
            continue
        if raw_mask.shape != (h, w):
            raw_mask = cv2.resize(raw_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        mask = raw_mask > 0.5
        output.append({"cls": class_id, "conf": float(confidence),
                       "id": None if tracking_id is None else int(tracking_id), "mask": mask})
    return output


def pick_instance(instances, class_id, previous_id):
    matches = [obj for obj in instances if obj["cls"] == class_id]
    if not matches:
        return None
    if previous_id is not None:
        kept = [obj for obj in matches if obj["id"] == previous_id]
        if kept:
            return kept[0]
    # Without an established track, take the most confident matching object.
    return max(matches, key=lambda obj: obj["conf"])


def instance_3d_points(obj, depth, K, dist, args):
    """Back-project sampled instance pixels using undistorted camera rays and metric Z."""
    mask = obj["mask"].astype(np.uint8)
    if args.mask_erode >= 2:
        kernel = np.ones((args.mask_erode, args.mask_erode), dtype=np.uint8)
        eroded = cv2.erode(mask, kernel, iterations=1)
        if np.count_nonzero(eroded) >= 20:
            mask = eroded
    valid = (mask > 0) & np.isfinite(depth) & (depth >= args.min_depth) & (depth <= args.max_depth)
    step = args.sample_step
    rows, cols = np.where(valid[::step, ::step])
    if len(rows) < 8 and step > 1:
        rows, cols = np.where(valid)
        y, x = rows, cols
    else:
        y, x = rows * step, cols * step
    if len(x) == 0:
        return None
    if len(x) > args.max_points:
        chosen = np.linspace(0, len(x) - 1, args.max_points).astype(np.int32)
        x, y = x[chosen], y[chosen]
    Z = depth[y, x].astype(np.float64)
    uv = np.column_stack([x, y]).astype(np.float64).reshape(-1, 1, 2)
    normalized = cv2.undistortPoints(uv, K, dist).reshape(-1, 2)
    points = np.column_stack([normalized[:, 0] * Z, normalized[:, 1] * Z, Z])
    # Median provides a stable representative visible-surface location, not a rigid-body center.
    median_xyz = np.median(points, axis=0)
    median_z = float(np.median(Z))
    med_uv = np.median(np.column_stack([x, y]), axis=0).astype(int)
    return {"points": points, "median_xyz": median_xyz, "median_z": median_z,
            "pixel_center": (int(med_uv[0]), int(med_uv[1])), "count": len(points)}


def surface_separation(human_xyz, bottle_xyz):
    """Visible nearest-pair estimate and a less outlier-sensitive low percentile."""
    tree = cKDTree(bottle_xyz["points"])
    distances, index = tree.query(human_xyz["points"], k=1)
    min_index = int(np.argmin(distances))
    closest_h = human_xyz["points"][min_index]
    closest_b = bottle_xyz["points"][index[min_index]]
    return float(distances[min_index]), float(np.percentile(distances, 5)), closest_h, closest_b


def depth_colormap(depth, max_depth):
    normalized = np.clip(depth / max_depth, 0, 1)
    vis = cv2.applyColorMap((255 * (1 - normalized)).astype(np.uint8), cv2.COLORMAP_TURBO)
    vis[~np.isfinite(depth) | (depth <= 0)] = (0, 0, 0)
    return vis


def display_line(frame, label, row, color=(255, 255, 255)):
    cv2.putText(frame, label, (12, row), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, label, (12, row), cv2.FONT_HERSHEY_SIMPLEX, 0.66, color, 2, cv2.LINE_AA)


def run():
    a = parse_args()
    if a.sample_step < 1 or a.max_points < 50 or a.max_depth <= a.min_depth:
        raise ValueError("Invalid sampling/depth parameters")
    if a.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS not available; use --device cpu")
    K, dist, calibrated_size = read_calibration(a.calibration)
    if calibrated_size != (a.width, a.height):
        raise ValueError(f"Calibration is for {calibrated_size}, but requested {(a.width, a.height)}. "
                         "Use the same calibrated camera mode; do not silently scale intrinsics across aspect ratios.")

    from ultralytics import YOLO
    model_path = PROJECT_ROOT / "models" / "yolo11n-seg.pt"
    yolo = YOLO(str(model_path if model_path.is_file() else "yolo11n-seg.pt"))
    processor = depth_net = None
    if a.mode == "both":
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        print("Loading metric indoor depth model...", flush=True)
        processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
        depth_net = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID).to(a.device).eval()
        print("Depth model type:", getattr(depth_net.config, "depth_estimation_type", "unknown"))
    print(f"Device: {a.device} | Camera: {a.width}x{a.height} | Mode: {a.mode}")

    cap = cv2.VideoCapture(a.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    path = PROJECT_ROOT / "results" / datetime.now().strftime("%Y%m%d_%H%M%S")
    path.mkdir(parents=True, exist_ok=True)
    csv_path = path / "person_bottle_tracking.csv"
    print("Every-frame CSV:", csv_path, flush=True)
    fields = ["timestamp", "frame", "mode", "fps_rolling", "yolo_ms", "depth_ms",
              "person_track_id", "bottle_track_id", "person_conf", "bottle_conf",
              "person_points", "bottle_points", "person_depth_m", "bottle_depth_m",
              "person_x_m", "person_y_m", "person_z_m",
              "bottle_x_m", "bottle_y_m", "bottle_z_m",
              "relative_x_m", "relative_y_m", "relative_z_m",
              "representative_distance_m", "visible_min_m", "visible_p05_m"]
    recent = deque(maxlen=30)
    chosen_ids = {PERSON_CLASS: None, BOTTLE_CLASS: None}
    frames = 0
    try:
        with csv_path.open("w", newline="", buffering=1) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            while True:
                tick = time.perf_counter()
                ok, frame = cap.read()
                if not ok:
                    print("Camera returned no frame", file=sys.stderr)
                    break
                h, w = frame.shape[:2]
                if (w, h) != calibrated_size:
                    raise RuntimeError(f"Actual camera frame {(w, h)} differs from calibration {calibrated_size}")
                frames += 1
                t = time.perf_counter()
                result = yolo.track(source=frame, imgsz=a.yolo_size, device=a.device,
                                    classes=[PERSON_CLASS, BOTTLE_CLASS], conf=a.confidence,
                                    tracker="bytetrack.yaml", persist=True,
                                    retina_masks=True, verbose=False)[0]
                if a.device == "mps":
                    torch.mps.synchronize()
                yolo_ms = (time.perf_counter() - t) * 1000
                instances = get_instances(result, frame.shape)
                selected = {}
                for class_id in (PERSON_CLASS, BOTTLE_CLASS):
                    obj = pick_instance(instances, class_id, chosen_ids[class_id])
                    selected[class_id] = obj
                    chosen_ids[class_id] = obj["id"] if obj is not None else None

                depth, depth_ms = None, float("nan")
                if a.mode == "both":
                    t = time.perf_counter()
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    batch = processor(images=Image.fromarray(rgb),
                                      size={"height": a.depth_height, "width": a.depth_width},
                                      return_tensors="pt")
                    batch = {key: value.to(a.device) for key, value in batch.items()}
                    with torch.inference_mode():
                        pred = depth_net(**batch).predicted_depth
                        resized = torch.nn.functional.interpolate(
                            pred.unsqueeze(1), size=(h, w), mode="bicubic",
                            align_corners=False).squeeze(0).squeeze(0)
                    depth = resized.float().cpu().numpy()
                    depth_ms = (time.perf_counter() - t) * 1000

                object_data = {}
                if depth is not None:
                    for class_id in (PERSON_CLASS, BOTTLE_CLASS):
                        obj = selected[class_id]
                        if obj is not None:
                            object_data[class_id] = instance_3d_points(obj, depth, K, dist, a)

                human = object_data.get(PERSON_CLASS)
                bottle = object_data.get(BOTTLE_CLASS)
                relative = None
                representative_distance = float("nan")
                visible_min, visible_p05 = float("nan"), float("nan")
                if human is not None and bottle is not None:
                    relative = bottle["median_xyz"] - human["median_xyz"]
                    representative_distance = float(np.linalg.norm(relative))
                    visible_min, visible_p05, _, _ = surface_separation(human, bottle)

                delta_ms = (time.perf_counter() - tick) * 1000
                recent.append(delta_ms)
                fps = 1000.0 / float(np.mean(recent))
                record = {"timestamp": datetime.now().isoformat(timespec="milliseconds"),
                          "frame": frames, "mode": a.mode, "fps_rolling": round(fps, 2),
                          "yolo_ms": round(yolo_ms, 1),
                          "depth_ms": round(depth_ms, 1) if np.isfinite(depth_ms) else ""}
                for cls, prefix in [(PERSON_CLASS, "person"), (BOTTLE_CLASS, "bottle")]:
                    obj, points = selected[cls], object_data.get(cls)
                    record[f"{prefix}_track_id"] = obj["id"] if obj is not None else ""
                    record[f"{prefix}_conf"] = round(obj["conf"], 3) if obj is not None else ""
                    record[f"{prefix}_points"] = points["count"] if points is not None else 0
                    record[f"{prefix}_depth_m"] = round(points["median_z"], 3) if points is not None else ""
                    for k, axis in enumerate(("x", "y", "z")):
                        record[f"{prefix}_{axis}_m"] = (round(float(points["median_xyz"][k]), 3)
                                                           if points is not None else "")
                for k, axis in enumerate(("x", "y", "z")):
                    record[f"relative_{axis}_m"] = round(float(relative[k]), 3) if relative is not None else ""
                record["representative_distance_m"] = (round(representative_distance, 3)
                                                         if np.isfinite(representative_distance) else "")
                record["visible_min_m"] = round(visible_min, 3) if np.isfinite(visible_min) else ""
                record["visible_p05_m"] = round(visible_p05, 3) if np.isfinite(visible_p05) else ""
                writer.writerow(record)

                if not a.no_display:
                    vis = frame.copy()
                    for cls, label in [(PERSON_CLASS, "Person"), (BOTTLE_CLASS, "Bottle")]:
                        obj = selected[cls]
                        if obj is None:
                            continue
                        color = COLORS[cls]
                        tint = np.zeros_like(vis)
                        tint[:] = color
                        vis[obj["mask"]] = cv2.addWeighted(vis, 0.7, tint, 0.3, 0)[obj["mask"]]
                        points = object_data.get(cls)
                        if points is not None:
                            px, py = points["pixel_center"]
                            cv2.circle(vis, (px, py), 6, color, -1)
                            cv2.putText(vis, f"{label} Z={points['median_z']:.2f}m",
                                        (min(px + 12, w - 300), max(40, py - 8)),
                                        cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2, cv2.LINE_AA)
                    if human is not None and bottle is not None:
                        cv2.line(vis, human["pixel_center"], bottle["pixel_center"], (0, 255, 255), 2)
                    display_line(vis, f"Person + bottle | {fps:.1f} FPS | CPU/MPS: {a.device}", 28)
                    display_line(vis, f"YOLO: {yolo_ms:.0f} ms | Depth: {depth_ms:.0f} ms", 56)
                    if np.isfinite(representative_distance):
                        display_line(vis, f"3D representative distance: {representative_distance:.2f} m", 84,
                                     (0, 255, 255))
                        display_line(vis, f"Visible min: {visible_min:.2f} m | P05: {visible_p05:.2f} m", 112,
                                     (0, 255, 255))
                        display_line(vis, f"Bottle - person XYZ: {relative[0]:+.2f}, {relative[1]:+.2f}, {relative[2]:+.2f} m", 140)
                    else:
                        display_line(vis, "Show one person and one bottle to measure separation", 84,
                                     (0, 255, 255))
                    if depth is not None:
                        view = np.hstack([vis, depth_colormap(depth, a.max_display_depth)])
                    else:
                        view = vis
                    # Keep the window usable on a MacBook display; coordinates above remain native.
                    if view.shape[1] > 1500:
                        factor = 1500 / view.shape[1]
                        view = cv2.resize(view, (1500, round(view.shape[0] * factor)))
                    cv2.imshow("Person + Bottle | Metric 3D separation (q to quit)", view)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                if frames % 30 == 0:
                    val = f"{representative_distance:.2f}m" if np.isfinite(representative_distance) else "n/a"
                    print(f"frames={frames} FPS={fps:.1f} YOLO={yolo_ms:.0f}ms "
                          f"depth={depth_ms:.0f}ms person={human is not None} "
                          f"bottle={bottle is not None} separation={val}", flush=True)
                if a.limit and frames >= a.limit:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Finished. CSV:", csv_path, flush=True)


if __name__ == "__main__":
    run()
