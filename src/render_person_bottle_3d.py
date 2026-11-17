#!/usr/bin/env python3
# Copyright (C) 2026 Nitai Shah
# SPDX-License-Identifier: AGPL-3.0-only
"""Render post-run 3D person/bottle point-cloud MP4s from a recorded session.

Run AFTER collecting frames with live_two_object_depth_record3d.py.
Uses actual recorded frame timestamps (sample-and-hold; no pose interpolation).
Preserves the 12 original checkerboard animation views, including upright axes.
The geometry is camera-frame visible-surface depth estimates, NOT a world map.

Example:
  python src/render_person_bottle_3d.py results/20261111_123000_123456 --only 03,07
  python src/render_person_bottle_3d.py results/20261111_123000_123456 --all-views
"""
from __future__ import annotations

import argparse
import csv
import json
from functools import lru_cache
from pathlib import Path

try:
    import imageio_ffmpeg
except ImportError as exc:
    raise SystemExit(
        "Missing video dependency. Install with: python -m pip install imageio-ffmpeg"
    ) from exc
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np

# Same view names, axis transforms, elevations, and azimuths as the user's
# dual_checkerboard_3d_animation.py. Native coordinates never get modified.
VIEWS = [
    ("01_original_z_vertical", "native", 30, -60, "Original: Z vertical"),
    ("02_original_from_opposite", "native", 30, 120, "Z vertical: opposite"),
    ("03_upright_isometric_a", "upright", 25, -65, "Upright diagonal A"),
    ("04_upright_isometric_b", "upright", 25, 25, "Upright diagonal B"),
    ("05_upright_isometric_c", "upright", 25, 115, "Upright diagonal C"),
    ("06_upright_isometric_d", "upright", 25, 205, "Upright diagonal D"),
    ("07_upright_from_camera", "upright", 0, -90, "Looking from camera"),
    ("08_upright_from_behind", "upright", 0, 90, "Looking toward camera"),
    ("09_upright_from_left", "upright", 0, 180, "Left profile"),
    ("10_upright_from_right", "upright", 0, 0, "Right profile"),
    ("11_upright_birds_eye", "upright", 65, -65, "High-angle view"),
    ("12_upright_top_down", "upright", 90, -90, "Directly overhead"),
]
AXIS_LABELS = {
    "native": ("X right (m)", "Y down (m)", "Z forward (m)"),
    "upright": ("X right (m)", "Depth Z forward (m)", "Height -Y up (m)"),
}
OBJECT_COLORS = {"person": "tab:blue", "bottle": "tab:orange"}


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("session_dir", type=Path, help="Result directory from live_two_object_depth_record3d.py")
    p.add_argument("--only", default="03,07", help="View numbers, e.g. 03,07,11")
    p.add_argument("--all-views", action="store_true", help="Render all 12 MP4 viewpoints (slow)")
    p.add_argument("--fps", type=int, default=6, help="Playback frames/sec; original measurement times are retained")
    p.add_argument("--max-render-points", type=int, default=1200, help="Maximum displayed points per object; does not discard saved data")
    p.add_argument("--color-mode", choices=["class", "rgb"], default="class")
    p.add_argument("--trail-seconds", type=float, default=20.0,
                   help="How far back the trajectory extends (0 hides trajectory)")
    p.add_argument("--show-min-line", action="store_true",
                   help="Also show sensitive minimum nearest-surface pair as a dotted line")
    p.add_argument("--preview-only", action="store_true", help="Generate the 12-view comparison PNG, no video")
    return p.parse_args()


def transform(pts, mode):
    """Only adjust PLOT axes; never change measured OpenCV-camera coordinates."""
    pts = np.asarray(pts, dtype=float)
    if mode == "native":
        return pts
    if mode == "upright":
        return pts[..., [0, 2, 1]] * np.array([1.0, 1.0, -1.0])
    raise ValueError(mode)


def load_session(session):
    csv_path = session / "person_bottle_tracking.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing {csv_path}")
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "recorded_3d_frame" not in rows[0]:
        raise ValueError("This session has no recorded 3D frames. The old CSV alone stores only median XYZ; "
                         "run live_two_object_depth_record3d.py to collect per-pixel 3D points.")

    records = []
    for row in rows:
        filename = row.get("recorded_3d_frame", "")
        if not filename:
            continue
        file = session / filename
        if not file.is_file():
            print(f"Warning: referenced 3D frame missing: {file}")
            continue
        records.append({"path": file, "elapsed": float(row["elapsed_s"]),
                        "frame": int(row["frame"])})
    records.sort(key=lambda x: (x["elapsed"], x["frame"]))
    if not records:
        raise ValueError("No recorded .npz point-cloud frames in this session")
    meta = session / "recording_metadata.json"
    if meta.is_file():
        print("Recording metadata:", json.loads(meta.read_text()).get("camera_frame", "not specified"))
    return records


@lru_cache(maxsize=3)
def load_frame(path):
    with np.load(path, allow_pickle=False) as dat:
        return {key: dat[key].copy() for key in dat.files}


def as_xyz(p):
    p = np.asarray(p, dtype=float)
    if p.shape == (3,) and np.isfinite(p).all():
        return p
    return None


def get_bounds(records, mode):
    """Fixed plot limits for the whole run, robust against rare extreme depth spikes."""
    lows = [np.zeros(3)]
    highs = [np.zeros(3)]
    for r in records:
        dat = load_frame(str(r["path"]))
        for k in ("person", "bottle"):
            pts = np.asarray(dat[k + "_points"], dtype=float)
            pts = pts[np.isfinite(pts).all(axis=1)]
            if len(pts):
                displayed = transform(pts, mode)
                lows.append(np.percentile(displayed, 1, axis=0))
                highs.append(np.percentile(displayed, 99, axis=0))
                center = as_xyz(dat[k + "_center_xyz"])
                if center is not None:
                    center = transform(center, mode)
                    lows.append(center)
                    highs.append(center)
    lo = np.min(np.stack(lows), axis=0)
    hi = np.max(np.stack(highs), axis=0)
    margin = np.maximum((hi - lo) * 0.10, 0.2)
    return lo - margin, hi + margin


def trace_segments(records, idx, name, seconds):
    """Separate trajectories on missing detections, unknown IDs, or ID changes."""
    if seconds <= 0:
        return []
    current_t = records[idx]["elapsed"]
    start = idx
    while start > 0 and records[start - 1]["elapsed"] >= current_t - seconds:
        start -= 1
    segments, segment, previous_id = [], [], None
    for j in range(start, idx + 1):
        dat = load_frame(str(records[j]["path"]))
        center = as_xyz(dat[name + "_center_xyz"])
        track = int(dat[name + "_track_id"])
        # Do not link paths through an unmatched or unknown identity.
        if center is None or track < 0:
            if segment:
                segments.append(np.stack(segment))
            segment, previous_id = [], None
            continue
        if previous_id is not None and track != previous_id:
            if segment:
                segments.append(np.stack(segment))
            segment = []
        segment.append(center)
        previous_id = track
    if segment:
        segments.append(np.stack(segment))
    return segments


def draw(ax, idx, view, records, bounds, args, preview=False):
    _, mode, elev, azim, label = view
    dat = load_frame(str(records[idx]["path"]))
    lo, hi = bounds
    ax.cla()
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(np.maximum(hi - lo, 0.1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel(AXIS_LABELS[mode][0], fontsize=8)
    ax.set_ylabel(AXIS_LABELS[mode][1], fontsize=8)
    ax.set_zlabel(AXIS_LABELS[mode][2], fontsize=8)
    ax.tick_params(labelsize=7)
    ax.scatter([0], [0], [0], c="black", marker="^", s=40,
               depthshade=False, label="Camera")

    seen = {}
    for key, name in (("person", "Person"), ("bottle", "Bottle")):
        pts = np.asarray(dat[key + "_points"], dtype=float)
        rgb = dat[key + "_rgb"]
        valid = np.isfinite(pts).all(axis=1)
        pts, rgb = pts[valid], rgb[valid]
        center = as_xyz(dat[key + "_center_xyz"])
        seen[key] = center

        if len(pts):
            if len(pts) > args.max_render_points:
                choices = np.linspace(0, len(pts) - 1, args.max_render_points, dtype=int)
                pts, rgb = pts[choices], rgb[choices]
            xyz = transform(pts, mode)
            c = (rgb.astype(float) / 255.0) if args.color_mode == "rgb" else OBJECT_COLORS[key]
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=c, s=2.4,
                       alpha=0.72, depthshade=False, rasterized=True,
                       label=f"{name} visible points")
        if not preview and args.trail_seconds > 0:
            for trace in trace_segments(records, idx, key, args.trail_seconds):
                if len(trace) >= 2:
                    xyz = transform(trace, mode)
                    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                            color=OBJECT_COLORS[key], linewidth=1.5, alpha=0.9)
        if center is not None:
            p = transform(center, mode)
            ax.scatter([p[0]], [p[1]], [p[2]], color=OBJECT_COLORS[key],
                       marker="o", s=48, edgecolors="black", linewidths=0.5,
                       depthshade=False)
            ax.text(p[0], p[1], p[2], " " + name, fontsize=8,
                    color=OBJECT_COLORS[key], weight="bold")

    center_distance = float(dat["representative_distance_m"])
    surface_min = float(dat["visible_min_m"])
    surface_p05 = float(dat["visible_p05_m"])
    if seen["person"] is not None and seen["bottle"] is not None:
        a, b = transform(seen["person"], mode), transform(seen["bottle"], mode)
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                "--", color="tab:green", linewidth=2.1, label="Representative separation")
        mid = (a + b) / 2
        if np.isfinite(center_distance):
            ax.text(mid[0], mid[1], mid[2], f"  {center_distance:.2f} m",
                    fontsize=9, color="darkgreen", weight="bold")

    if args.show_min_line and np.isfinite(surface_min):
        close_h = as_xyz(dat["closest_h_xyz"])
        close_b = as_xyz(dat["closest_b_xyz"])
        if close_h is not None and close_b is not None:
            a, b = transform(close_h, mode), transform(close_b, mode)
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                    ":", color="purple", linewidth=2, label="Raw nearest visible points")

    t = records[idx]["elapsed"] - records[0]["elapsed"]
    s_center = f"{center_distance:.2f} m" if np.isfinite(center_distance) else "unavailable"
    s_surface = (f"min={surface_min:.2f}, P05={surface_p05:.2f} m"
                 if np.isfinite(surface_min) and np.isfinite(surface_p05) else "unavailable")
    ax.set_title(f"{label}  |  recorded t={t:.1f}s  |  frame={records[idx]['frame']}\n"
                 f"Representatives: {s_center}  |  visible surfaces: {s_surface}", fontsize=10)
    if not preview:
        ax.legend(loc="upper left", fontsize=7)


def export_preview(out, records, bounds, args):
    both = []
    for i, r in enumerate(records):
        dat = load_frame(str(r["path"]))
        if as_xyz(dat["person_center_xyz"]) is not None and as_xyz(dat["bottle_center_xyz"]) is not None:
            both.append(i)
    candidate = both if both else list(range(len(records)))
    idx = candidate[len(candidate) // 2]
    fig = plt.figure(figsize=(18, 13))
    for j, view in enumerate(VIEWS, start=1):
        ax = fig.add_subplot(3, 4, j, projection="3d")
        draw(ax, idx, view, records, bounds[view[1]], args, preview=True)
        ax.set_title(view[0].replace("_", " "), fontsize=8)
        ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
        ax.tick_params(labelsize=5)
    fig.suptitle("Person–bottle: 12 camera viewpoints (same recorded 3D frame)", fontsize=15)
    fig.savefig(out / "view_comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("Saved view comparison:", out / "view_comparison.png", flush=True)


def render_video(out, view, records, bounds, args):
    name, mode, _, _, _ = view
    t = np.array([r["elapsed"] for r in records])
    duration_s = max(0.0, t[-1] - t[0])
    n_output_frames = max(1, int(np.ceil(duration_s * args.fps)) + 1)
    # Hold the most recently RECORDED frame; do not interpolate unmeasured poses.
    target_times = t[0] + np.arange(n_output_frames, dtype=float) / args.fps
    indices = np.searchsorted(t, target_times, side="right") - 1
    indices = np.clip(indices, 0, len(records) - 1)
    target = out / f"{name}.mp4"
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    try:
        with FFMpegWriter(fps=args.fps, bitrate=1800).saving(fig, str(target), dpi=110) as writer:
            for n, i in enumerate(indices):
                draw(ax, int(i), view, records, bounds[mode], args)
                writer.grab_frame()
                if (n + 1) % 25 == 0 or n == n_output_frames - 1:
                    print(f"  {name}: rendered {n+1}/{n_output_frames} video frames", flush=True)
    finally:
        plt.close(fig)
    print("Saved MP4:", target, flush=True)


def main():
    a = arguments()
    if a.fps < 1 or a.max_render_points < 10 or a.trail_seconds < 0:
        raise ValueError("fps, max-render-points, and trail-seconds must be valid positive values")
    selected = {x.strip().zfill(2) for x in a.only.split(",") if x.strip()}
    if not a.all_views and any(x not in {v[0][:2] for v in VIEWS} for x in selected):
        raise ValueError("Unknown --only view; choose 01 through 12")
    chosen = VIEWS if a.all_views else [v for v in VIEWS if v[0][:2] in selected]
    if not chosen and not a.preview_only:
        raise ValueError("Choose at least one view")

    session = a.session_dir.expanduser().resolve()
    records = load_session(session)
    out = session / "3d_visualizations"
    out.mkdir(exist_ok=True)
    bounds = {mode: get_bounds(records, mode) for mode in ("native", "upright")}
    print(f"Loaded {len(records)} point-cloud frames from {session}", flush=True)
    print("Geometry: monocular visible surfaces in METERS; origin = camera.", flush=True)

    guides = [
        "Derived from YOLO segmentation + Depth Anything V2 Metric (unvalidated ground-truth scale).",
        "Fixed plot bounds use per-frame 1st/99th percentile to avoid outlier-driven axes.",
        "Native: X right, Y down, Z forward; upright: X right, depth Z forward, height -Y up.",
        "Representative points are component-wise XYZ medians; not complete-object centers.",
        "Nearest-visible-point distances can be corrupted by depth outliers.",
        "Videos use recorded elapsed time with sample-and-hold; missing poses not interpolated.",
        "Tracking trajectories are interrupted on absent/unknown IDs or ID changes.", "",
    ] + [f"{v[0]}: {v[4]} (elev={v[2]}, azim={v[3]})" for v in VIEWS]
    (out / "view_guide.txt").write_text("\n".join(guides) + "\n")
    export_preview(out, records, bounds, a)
    if not a.preview_only:
        for j, view in enumerate(chosen, start=1):
            print(f"[{j}/{len(chosen)}] Rendering {view[0]}", flush=True)
            render_video(out, view, records, bounds, a)
    print("All visualization output:", out, flush=True)


if __name__ == "__main__":
    main()
