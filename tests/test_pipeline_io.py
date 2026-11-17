# Copyright (C) 2026 Nitai Shah
# SPDX-License-Identifier: AGPL-3.0-only
"""Camera-free sanity tests for point-cloud recording and plotted coordinates."""
import csv
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calibrate_webcam import board_object_points, calibrate_samples
from live_two_object_depth_record3d import (
    PERSON_CLASS, BOTTLE_CLASS, save_pointcloud_frame,
)
from render_person_bottle_3d import load_frame, load_session, transform, get_bounds


class CameraGeometryTest(unittest.TestCase):
    def test_board_shape_and_units(self):
        points = board_object_points(10, 7, 25.0)
        self.assertEqual(points.shape, (70, 3))
        np.testing.assert_allclose(points[0], [0, 0, 0])
        self.assertAlmostEqual(float(points[:, 0].max()), 0.225)
        self.assertAlmostEqual(float(points[:, 1].max()), 0.150)

    def test_upright_is_plot_only(self):
        camera_xyz = np.array([0.5, 0.4, 1.6])
        np.testing.assert_allclose(transform(camera_xyz, "upright"), [0.5, 1.6, -0.4])
        np.testing.assert_allclose(camera_xyz, [0.5, 0.4, 1.6])
        self.assertAlmostEqual(np.linalg.norm(camera_xyz),
                               np.linalg.norm(transform(camera_xyz, "upright")))


class RecordingIOTest(unittest.TestCase):
    def test_npz_roundtrip_without_pickle(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            session = folder / "results" / "example"
            (session / "pointcloud_frames").mkdir(parents=True)
            target = session / "pointcloud_frames" / "frame_000001.npz"
            frame = np.zeros((6, 8, 3), dtype=np.uint8)
            frame[2, 3, :] = (10, 20, 30)  # BGR
            person = {
                "points": np.array([[0.1, -0.2, 2.0], [0.2, 0.0, 2.1]]),
                "median_xyz": np.array([0.15, -0.1, 2.05]),
                "pixel_xy": np.array([[3, 2], [4, 2]]),
            }
            bottle = {
                "points": np.array([[0.5, 0.0, 1.8]]),
                "median_xyz": np.array([0.5, 0.0, 1.8]),
                "pixel_xy": np.array([[5, 2]]),
            }
            selected = {PERSON_CLASS: {"id": 5}, BOTTLE_CLASS: {"id": 9}}
            save_pointcloud_frame(target, frame, 0.25, selected,
                                  {PERSON_CLASS: person, BOTTLE_CLASS: bottle},
                                  0.4, 0.25, 0.3,
                                  (person["points"][0], bottle["points"][0]))
            self.assertTrue(target.exists())
            self.assertFalse((target.with_name(target.name + ".tmp")).exists())
            data = load_frame(str(target))
            self.assertEqual(data["person_points"].shape, (2, 3))
            self.assertEqual(data["bottle_points"].shape, (1, 3))
            np.testing.assert_array_equal(data["person_rgb"][0], [30, 20, 10])
            self.assertEqual(int(data["person_track_id"]), 5)
            with (session / "person_bottle_tracking.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["elapsed_s", "frame", "recorded_3d_frame"])
                writer.writeheader()
                writer.writerow({"elapsed_s": "0.25", "frame": "1",
                                 "recorded_3d_frame": "pointcloud_frames/frame_000001.npz"})
            records = load_session(session)
            self.assertEqual(len(records), 1)
            lo, hi = get_bounds(records, "upright")
            self.assertTrue(np.all(lo < hi))


if __name__ == "__main__":
    unittest.main()
