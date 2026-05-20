"""
Phase 4/5 — Live webcam inference, trigger logic, and YouTube actions.

The full detection stack: webcam frame -> detector -> state machine -> action.
When a sign clears the vote + cooldown gates, the ActionDispatcher opens that
sign's configured YouTube music video in the browser (configs/actions.yaml).
Until you fill in real URLs there, triggers are logged but open nothing — so
this still runs as a pure detection demo out of the box.

Run (GPU env — has both ultralytics and opencv):
  .venv-rocm\\Scripts\\python.exe scripts\\run.py

Controls: hold a sign in front of the camera; press 'q' to quit.

Note: the FIRST inference call pauses briefly while ROCm compiles GPU kernels.
After that it runs in real time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

# Make the repo root importable so `import detector` works regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from actions.dispatcher import ActionDispatcher  # noqa: E402
from detector.inference import HandSignDetector  # noqa: E402
from detector.state_machine import SignStateMachine, TriggerEvent  # noqa: E402

# Per-class box colors (BGR). other_hand is muted grey since it never fires.
CLASS_COLORS = {
    "other_hand": (150, 150, 150),
    "sign_nine": (0, 220, 0),
    "sign_ysl": (255, 140, 0),
}
_DEFAULT_COLOR = (255, 255, 255)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def on_trigger(event: TriggerEvent, dispatcher: ActionDispatcher) -> None:
    """A sign cleared the vote + cooldown gates -> run its configured action."""
    print(f"FIRE: {event.class_name}  (t={event.timestamp:.1f}s)")
    dispatcher.fire(event.class_name)


def draw_detections(frame, detections) -> None:
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        color = CLASS_COLORS.get(det.class_name, _DEFAULT_COLOR)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        cv2.putText(frame, label, (x1, max(y1 - 8, 12)), _FONT, 0.6, color, 2)


def draw_status(frame, state: SignStateMachine) -> None:
    if state.in_cooldown():
        text = f"COOLDOWN {state.cooldown_remaining():.1f}s"
        color = (0, 165, 255)  # orange
    else:
        leading, votes = state.progress()
        if leading:
            text = f"Tracking {leading}: {votes}/{state.window}"
            color = (0, 255, 255)  # yellow
        else:
            text = "Waiting for a sign..."
            color = (200, 200, 200)
    cv2.putText(frame, text, (10, 30), _FONT, 0.7, color, 2)


def main() -> None:
    detector = HandSignDetector()
    state = SignStateMachine()
    dispatcher = ActionDispatcher()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam at index 0.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame grab failed; stopping.")
                break

            frame = cv2.flip(frame, 1)  # mirror for natural interaction

            detections = detector.detect(frame)
            event = state.update(detections)
            if event is not None:
                on_trigger(event, dispatcher)

            draw_detections(frame, detections)
            draw_status(frame, state)
            cv2.imshow("9-vicious-detector", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
