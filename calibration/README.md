# Camera calibration

`webcam_calibration.npz` is **not included** because it must be measured for each
physical camera, capture resolution, focus/crop mode, and lens configuration.

Generate it from a planar **checkerboard with 10 x 7 inner corners** (11 x 8
squares) and **25 mm squares** (or pass the actual board parameters):

```bash
python src/calibrate_webcam.py --camera 0 --width 1280 --height 720 \
  --inner-cols 10 --inner-rows 7 --square-size-mm 25 --min-samples 20
```

Press SPACE at 20+ **different** camera-to-board poses (vary yaw/pitch, depth,
position across the frame, including near edges). Press `q` to finish; ensure
all inner corners are visible. The file `calibration/webcam_calibration.npz`
will contain `K`, `dist`, `image_size=[width,height]`, and `rms`.

A reprojection RMS is only an internal calibration fit metric; it does not
prove accurate camera-to-object distance or verify the learned depth scale.

Do not publish your personal calibration file as a generally reusable
calibration or run inference at a mismatched resolution.
