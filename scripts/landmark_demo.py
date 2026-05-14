"""
Phase 0 demo: open the webcam, run MediaPipe Hand Landmarker on every frame,
draw the 21-keypoint skeleton overlay, exit on 'q'.

Uses the modern MediaPipe Tasks API (mp.tasks.*). The legacy mp.solutions
namespace was removed in MediaPipe 0.10.35+.

Goal of this script:
  1. Prove the dev environment works end-to-end (Python + OpenCV + MediaPipe + webcam).
  2. First real-time CV inference experience before training our own model.
"""

import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = "models/hand_landmarker.task"

# The 21 hand landmarks form a known skeleton. Each tuple is an edge between
# two landmark indices. Drawing these connects fingertips through knuckles to
# the wrist. (MediaPipe documents the index layout in its Hand Landmarker guide.)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index finger
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring finger
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),  # pinky + palm base
]


def draw_hand(frame, landmarks) -> None:
    """Draw the skeleton for one hand onto `frame` (BGR, in-place)."""
    h, w = frame.shape[:2]
    # Landmark coords are normalized (0..1); multiply by image size to get pixels.
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, points[a], points[b], (255, 255, 255), 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)


def main() -> None:
    # BaseOptions tells the Tasks API which model file to load.
    # The .task file is a self-contained bundle of weights + metadata.
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)

    # HandLandmarkerOptions configures task-specific behavior.
    #   running_mode=VIDEO -> we're processing a continuous stream (vs IMAGE for one-shot)
    #   num_hands         -> track up to N hands per frame
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Is it connected? Is another app using it?")

    # `with` ensures the model unloads cleanly on exit.
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        # VIDEO mode requires STRICTLY increasing ms timestamps (t_new > t_prev).
        # Integer-flooring monotonic ns to ms can produce duplicates when two
        # frames arrive less than 1 ms apart, so we also track the last value
        # and bump by 1 ms whenever the wall-clock derivation would tie or regress.
        start_ns = time.monotonic_ns()
        last_ts_ms = -1

        while True:
            ok, frame = cap.read()
            if not ok:
                continue  # transient read failure — try again

            # Mirror for selfie-style UX so on-screen movement matches the user.
            frame = cv2.flip(frame, 1)

            # OpenCV captures BGR; MediaPipe Tasks expects RGB.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Wrap the numpy array in MediaPipe's Image container.
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            ts_ms = (time.monotonic_ns() - start_ns) // 1_000_000
            if ts_ms <= last_ts_ms:
                ts_ms = last_ts_ms + 1
            last_ts_ms = ts_ms

            result = landmarker.detect_for_video(mp_image, ts_ms)

            # result.hand_landmarks is a list (one entry per detected hand),
            # each entry is a list of 21 landmark objects with .x .y .z.
            for hand in result.hand_landmarks:
                draw_hand(frame, hand)

            cv2.imshow("Phase 0 - Hand Landmarker (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
