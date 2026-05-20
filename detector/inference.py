"""
Phase 4 — Perception layer.

`HandSignDetector` wraps the trained YOLO checkpoint (`best.pt`) and turns a
single BGR frame into a list of `Detection`s. It is deliberately STATELESS:
it knows nothing about previous frames, triggers, or cooldowns. That logic
lives one layer up in `state_machine.py`. Keeping perception dumb makes it
trivial to reason about and test frame-by-frame.

Two things worth knowing:
  * BGR vs RGB: OpenCV hands us frames in BGR order. Ultralytics' `predict()`
    expects exactly that for a NumPy array (it assumes OpenCV convention and
    converts internally), so we pass the cv2 frame straight through — no manual
    cvtColor. Get this wrong with a model that wants RGB and colors invert.
  * NMS (Non-Maximum Suppression): when the model emits several overlapping
    boxes for one hand, it keeps the highest-confidence one and suppresses the
    rest. Ultralytics runs NMS inside `predict()`, so each Detection we return
    is already de-duplicated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO

# Repo root = two levels up from this file (detector/inference.py -> repo/).
# Resolving relative to the file (not the cwd) means inference works no matter
# which directory the caller runs from.
_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = _REPO_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"

# Detector-side confidence floor. Kept LOW so the HUD shows everything the model
# sees; the *trigger* threshold (much stricter) is the state machine's job.
DEFAULT_CONF = 0.25


@dataclass(frozen=True)
class Detection:
    """One detected hand in one frame. bbox is (x1, y1, x2, y2) in pixels."""

    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]


class HandSignDetector:
    def __init__(
        self,
        weights: Path | str = DEFAULT_WEIGHTS,
        conf: float = DEFAULT_CONF,
        imgsz: int = 640,
        device: int | str = 0,
    ) -> None:
        weights = Path(weights)
        if not weights.exists():
            raise FileNotFoundError(
                f"Model weights not found at {weights}. "
                "Train first (scripts/train.py) or pass an explicit path."
            )
        self.model = YOLO(str(weights))
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        # id -> name map baked into the checkpoint (e.g. {0:'other_hand', ...})
        self.class_names: dict[int, str] = self.model.names

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run the model on one BGR frame; return de-duplicated detections."""
        results = self.model.predict(
            frame,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,  # silence per-frame console spam
        )
        result = results[0]  # one frame in -> one result out
        detections: list[Detection] = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            detections.append(
                Detection(
                    class_name=self.class_names[cls_id],
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                )
            )
        return detections
