#!/usr/bin/env python3
# Copyright (C) 2026 Nitai Shah
# SPDX-License-Identifier: AGPL-3.0-only
"""Live person instance segmentation + metric depth for Apple Silicon.

Research-only prototype. This script intentionally does NOT estimate the
human-to-robot distance: stock COCO weights segment people but do not provide
a reliable dedicated cobot mask. Visualized depth values are metric model
estimates, not safety measurements.
"""

import argparse
import csv
from collections import deque
from datetime import datetime
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["yolo", "depth", "both"], default="both")
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="cpu",
                        help="CPU is the portable default; MPS must be selected explicitly")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--yolo-size", type=int, default=416)
    parser.add_argument("--depth-height", type=int, default=336)
    parser.add_argument("--depth-width", type=int, default=448)
    parser.add_argument("--max-display-depth", type=float, default=5.0)
    parser.add_argument("--no-display", action="store_true", help="Log without opening OpenCV windows")
    parser.add_argument("--limit", type=int, default=0, help="Stop after N frames (0 = unlimited)")
    return parser.parse_args()


def select_device(requested):
    available = torch.backends.mps.is_available()
    if requested == "mps" and not available:
        raise RuntimeError("MPS unavailable. Check native arm64 Python/PyTorch or use --device cpu.")
    if requested == "auto":
        return "mps" if available else "cpu"
    return requested


def sync_if_needed(device):
    if device == "mps":
        torch.mps.synchronize()


def get_people_mask(result, image_shape):
    """Return a union of COCO 'person' instance masks at input resolution."""
    height, width = image_shape[:2]
    union = np.zeros((height, width), dtype=bool)
    count = 0
    if result.masks is None or result.boxes is None:
        return union, count
    labels = result.boxes.cls.detach().cpu().numpy().astype(int)
    mask_data = result.masks.data.detach().cpu().numpy()
    for cls_id, mask in zip(labels, mask_data):
        if cls_id != 0:  # COCO person
            continue
        count += 1
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
        union |= mask > 0.5
    return union, count


def depth_to_colormap(depth, max_depth_m):
    """Fixed color scale to prevent visually misleading per-frame normalization."""
    scaled = np.clip(depth / max_depth_m, 0, 1)
    gray = (255 * (1 - scaled)).astype(np.uint8)
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    colored[(~np.isfinite(depth)) | (depth <= 0)] = (0, 0, 0)
    return colored


def overlay_person_mask(image, person_mask):
    result = image.copy()
    if not person_mask.any():
        return result
    green = np.zeros_like(result)
    green[:, :, 1] = 230
    result[person_mask] = cv2.addWeighted(result, 0.65, green, 0.35, 0)[person_mask]
    return result


def text_line(img, text, y, color=(255, 255, 255)):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1, cv2.LINE_AA)


def main():
    args = parse_args()
    if min(args.depth_height, args.depth_width) <= 0 or args.max_display_depth <= 0:
        raise ValueError("Depth resolution and display range must be positive")
    device = select_device(args.device)
    print(f"Device: {device}; PyTorch: {torch.__version__}", flush=True)

    yolo = None
    processor = None
    depth_net = None

    if args.mode in {"yolo", "both"}:
        from ultralytics import YOLO
        model_file = PROJECT_ROOT / "models" / "yolo11n-seg.pt"
        # Ultralytics downloads known pretrained model names when not cached.
        if not model_file.exists():
            print("Downloading YOLO11n segmentation checkpoint on first run...", flush=True)
            model_file = "yolo11n-seg.pt"
        yolo = YOLO(str(model_file))

    if args.mode in {"depth", "both"}:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        print("Loading metric indoor depth checkpoint (first run downloads weights)...", flush=True)
        processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
        depth_net = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID)
        depth_net.to(device)
        depth_net.eval()
        # Important: this is a metric checkpoint, NOT the relative-depth checkpoint.
        print("Depth model type:", getattr(depth_net.config, "depth_estimation_type", "unknown"), flush=True)

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError("Cannot open camera. Check Terminal camera permissions in macOS Settings.")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # may be ignored on macOS

    run_dir = PROJECT_ROOT / "results" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_file = run_dir / "per_frame.csv"
    print(f"Writing every-frame metrics to {csv_file}", flush=True)
    fields = ["timestamp", "frame", "mode", "device", "yolo_ms", "depth_ms", "total_ms", "fps_rolling", "people", "person_depth_median_m", "person_depth_p10_m"]
    frame_times = deque(maxlen=30)

    try:
        with csv_file.open("w", newline="", buffering=1) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            frame_index = 0
            while True:
                cycle_start = time.perf_counter()
                ok, frame = capture.read()
                if not ok:
                    print("Camera stopped returning frames.", file=sys.stderr)
                    break
                frame_index += 1
                H, W = frame.shape[:2]

                mask = np.zeros((H, W), dtype=bool)
                people = 0
                yolo_ms = float("nan")
                depth_ms = float("nan")
                depth = None
                median = float("nan")
                p10 = float("nan")

                if yolo is not None:
                    start = time.perf_counter()
                    result = yolo.predict(source=frame, imgsz=args.yolo_size, device=device,
                                          classes=[0], conf=0.3, retina_masks=True,
                                          verbose=False)[0]
                    sync_if_needed(device)
                    mask, people = get_people_mask(result, frame.shape)
                    yolo_ms = (time.perf_counter() - start) * 1000

                if depth_net is not None:
                    start = time.perf_counter()
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb)
                    batch = processor(images=pil_image,
                                      size={"height": args.depth_height, "width": args.depth_width},
                                      return_tensors="pt")
                    batch = {key: val.to(device) for key, val in batch.items()}
                    with torch.inference_mode():
                        predicted = depth_net(**batch).predicted_depth
                        restored = torch.nn.functional.interpolate(
                            predicted.unsqueeze(1), size=(H, W), mode="bicubic",
                            align_corners=False,
                        ).squeeze(1).squeeze(0)
                    depth = restored.float().cpu().numpy()
                    depth_ms = (time.perf_counter() - start) * 1000
                    if mask.any():
                        valid = mask & np.isfinite(depth) & (depth > 0)
                        if valid.any():
                            median = float(np.median(depth[valid]))
                            p10 = float(np.percentile(depth[valid], 10))

                total_ms = (time.perf_counter() - cycle_start) * 1000
                frame_times.append(total_ms)
                fps = 1000.0 / np.mean(frame_times)
                writer.writerow({
                    "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                    "frame": frame_index, "mode": args.mode, "device": device,
                    "yolo_ms": f"{yolo_ms:.1f}" if np.isfinite(yolo_ms) else "",
                    "depth_ms": f"{depth_ms:.1f}" if np.isfinite(depth_ms) else "",
                    "total_ms": f"{total_ms:.1f}", "fps_rolling": f"{fps:.2f}",
                    "people": people,
                    "person_depth_median_m": f"{median:.3f}" if np.isfinite(median) else "",
                    "person_depth_p10_m": f"{p10:.3f}" if np.isfinite(p10) else "",
                })

                if not args.no_display:
                    left = overlay_person_mask(frame, mask)
                    text_line(left, f"Mode: {args.mode} | Device: {device} | FPS: {fps:.1f}", 26)
                    if yolo is not None:
                        text_line(left, f"People: {people} | YOLO: {yolo_ms:.0f} ms", 51)
                    if np.isfinite(median):
                        text_line(left, f"Person median camera depth: {median:.2f} m", 76)
                        text_line(left, f"Person 10th pct depth: {p10:.2f} m", 101)
                    elif yolo is not None and depth is not None:
                        text_line(left, "No valid person depth", 76, (0, 200, 255))
                    if depth is not None:
                        right = depth_to_colormap(depth, args.max_display_depth)
                        text_line(right, f"Metric depth | 0-{args.max_display_depth:g} m | {depth_ms:.0f} ms", 26)
                        view = np.hstack([left, right])
                    else:
                        view = left
                    cv2.imshow("Ferris Lab - YOLO Seg + Metric Depth (q to quit)", view)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        break

                if frame_index % 30 == 0:
                    print(f"frames={frame_index} | FPS={fps:.1f} | YOLO={yolo_ms:.0f}ms | depth={depth_ms:.0f}ms | person_depth={median:.2f}m", flush=True)
                if args.limit and frame_index >= args.limit:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        cv2.destroyAllWindows()
        print(f"Finished. CSV: {csv_file}")


if __name__ == "__main__":
    main()
