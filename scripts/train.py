"""
Phase 3b — Train the hand-sign detector via transfer learning.

This script does NOT hand-write a training loop. Ultralytics wraps data
loading, augmentation, the forward/backward pass, the optimizer, validation,
logging, and checkpointing behind `model.train(...)`. Our job here is to
declare the configuration and press go — the intelligence is in the choices,
not the line count.

Transfer learning, in one breath:
  `yolov8n.pt` was pretrained on COCO (80 everyday object classes, hundreds of
  thousands of images). COCO has no hand signs, but its early conv layers
  already learned generic visual primitives — edges, corners, textures — that
  transfer to recognizing *anything*. We keep those and only adapt the head to
  our 3 classes. That's why ~880 training images suffices instead of ~880,000.

Run it (GPU env — NOT the MediaPipe .venv):
  .venv-rocm\\Scripts\\python.exe scripts\\train.py

Outputs land in runs/detect/train/:
  weights/best.pt   — best checkpoint by val mAP (the Phase 4 artifact)
  weights/last.pt   — most recent epoch (for resuming)
  results.png       — loss + mAP curves over epochs
  confusion_matrix.png, val_batch*_pred.jpg — diagnose by eye, not just metrics

Windows note: the `if __name__ == '__main__':` guard is REQUIRED, not stylistic.
Ultralytics' dataloader spawns worker processes; on Windows `spawn` re-imports
this module in each worker. Without the guard, that re-import re-runs train()
recursively and the run crashes.
"""

from __future__ import annotations

from ultralytics import YOLO

# COCO-pretrained nano checkpoint. Smallest/fastest YOLOv8 (~3M params).
# Auto-downloads on first run. Escalate to 'yolov8s.pt' only if nano underfits
# AFTER data fixes — don't reach for a bigger model to paper over a data problem.
MODEL = "yolov8n.pt"

# The data contract: tells YOLO where train/val/test live and the class names
# (nc: 3 -> other_hand, sign_nine, sign_ysl). The model reads THIS, not folders.
DATA = "data/dataset/data.yaml"

# --- Hyperparameters (the knobs we're actually deciding) ---
EPOCHS = 100      # upper bound on full passes; early stopping usually ends sooner
IMGSZ = 640       # input resolution — MUST match the 640x640 Roboflow export
BATCH = 16        # images per gradient update; conservative vs. the 15.9 GB VRAM
DEVICE = 0        # the AMD RX 9060 XT (ROCm exposes it through the CUDA API)
PATIENCE = 20     # stop if val mAP hasn't improved for this many epochs


def main() -> None:
    model = YOLO(MODEL)
    model.train(
        data=DATA,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        patience=PATIENCE,
    )


if __name__ == "__main__":
    main()
