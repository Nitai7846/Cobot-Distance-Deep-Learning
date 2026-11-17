# Model checkpoints

No model checkpoints are committed to GitHub.

- `yolo11n-seg.pt` is loaded locally if present, otherwise Ultralytics can
  download the pretrained model on its first run.
- `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf` is retrieved
  through the Hugging Face Transformers cache on first run.

An internet connection is needed for the initial downloads. Check the
respective model licenses and usage terms before redistribution.
