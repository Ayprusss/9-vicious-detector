"""
Phase 1 — Data collection.

Opens the webcam, lets you tag frames with one of three classes
(sign_ysl, sign_nine, other_hand) using number keys, and saves them on SPACE.

Keybindings:
  1 / 2 / 3   switch active class
  SPACE       save current frame to data/raw/<active_class>/<utc-ts>.jpg
  B           start 5-second countdown, then burst-save 10 frames
              (press B again during countdown to cancel)
  BACKSPACE   delete the most recent save (this session only)
  q           quit

Burst mode exists so you can step back from the camera (far-distance
and full-body shots are otherwise awkward to capture solo). The shots
are spaced ~150ms apart so the hand drifts naturally between frames —
a true camera-rate burst would just produce 10 near-identical images.

Saved frames mirror the preview (WYSIWYG). HUD overlays are drawn AFTER
the frame is copied for saving, so saved images contain no annotations.
Annotated saves would let the model learn the overlay as a shortcut
feature ("ACTIVE: sign_ysl" == sign_ysl). This is why we copy first,
draw second.
"""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

import cv2

DATA_ROOT = Path("data/raw")

# Order matters: index 0/1/2 maps to keys 1/2/3.
CLASSES = ("sign_ysl", "sign_nine", "other_hand")

# Short labels for the on-screen count strip (full names are too long).
SHORT_NAMES = {"sign_ysl": "ysl", "sign_nine": "nine", "other_hand": "other"}

# BGR (OpenCV is BGR) — one distinct color per active class.
CLASS_COLORS = {
    "sign_ysl": (255, 255, 0),     # cyan
    "sign_nine": (255, 0, 255),    # magenta
    "other_hand": (200, 200, 200), # near-white
}

# Capture at 1280x720. YOLO trains at 640x640, but extra pixels give
# Roboflow more to work with for crops/resizes in Phase 2.
CAP_WIDTH = 1280
CAP_HEIGHT = 720

# Frames to keep the "SAVED" / "DELETED" flash visible (~0.5s at 30fps).
FLASH_FRAMES = 15

# Imbalance warning kicks in only once at least one class has 20 images
# (early counts are too noisy to be meaningful).
IMBALANCE_MIN_COUNT = 20
IMBALANCE_RATIO = 0.20

# Burst-with-countdown: hit B, get 5s to walk back, then 10 photos auto-save.
BURST_COUNTDOWN_SECONDS = 5
BURST_PHOTO_COUNT = 10
# Spacing between burst shots (seconds). ~150ms gives natural micro-variation
# in hand pose across the 10 frames — a camera-rate burst (~33ms) would yield
# near-duplicates that teach the model nothing new.
BURST_INTERVAL_S = 0.15


def init_counts(data_root: Path) -> dict[str, int]:
    """Scan existing class folders so previous-session counts persist."""
    counts: dict[str, int] = {}
    for c in CLASSES:
        d = data_root / c
        d.mkdir(parents=True, exist_ok=True)
        counts[c] = sum(1 for _ in d.glob("*.jpg"))
    return counts


def save_frame(frame, class_name: str, data_root: Path) -> Path:
    # Microsecond UTC timestamp: no collisions even on rapid bursts, sorts chronologically.
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    out = data_root / class_name / f"{ts}.jpg"
    cv2.imwrite(str(out), frame)
    return out


def check_imbalance(counts: dict[str, int]) -> str | None:
    max_c = max(counts.values())
    if max_c < IMBALANCE_MIN_COUNT:
        return None
    min_c = min(counts.values())
    if (max_c - min_c) / max_c < IMBALANCE_RATIO:
        return None
    leader = max(counts, key=lambda k: counts[k])
    laggard = min(counts, key=lambda k: counts[k])
    pct = round((max_c - min_c) / max_c * 100)
    return f"{SHORT_NAMES[leader]} ahead of {SHORT_NAMES[laggard]} by {pct}%"


def draw_hud(
    frame,
    active_class: str,
    counts: dict[str, int],
    flash_kind: str | None,
    flash_frames_left: int,
    burst_state: str = "idle",
    countdown_remaining: int = 0,
    burst_progress: int = 0,
) -> None:
    h, w = frame.shape[:2]
    color = CLASS_COLORS[active_class]

    # Top strip — active class + counts.
    cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)
    cv2.putText(frame, f"ACTIVE: {active_class}", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    counts_str = "  ".join(f"{SHORT_NAMES[c]}: {counts[c]}" for c in CLASSES)
    cv2.putText(frame, counts_str, (12, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1)

    # Top-right status: burst progress takes precedence over save/delete flash.
    if burst_state == "burst":
        cv2.putText(frame, f"BURST {burst_progress}/{BURST_PHOTO_COUNT}",
                    (w - 290, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
    elif flash_frames_left > 0 and flash_kind:
        if flash_kind == "SAVED":
            flash_color = (0, 255, 0)
        elif flash_kind == "BURST DONE":
            flash_color = (0, 255, 255)
        else:
            flash_color = (0, 165, 255)
        cv2.putText(frame, flash_kind, (w - 290, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, flash_color, 3)

    # Big centered countdown number (only while counting down).
    if burst_state == "countdown" and countdown_remaining > 0:
        text = str(countdown_remaining)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 8.0
        thickness = 15
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x = (w - tw) // 2
        y = (h + th) // 2
        # Black outline so the number stays legible against any background.
        cv2.putText(frame, text, (x, y), font, font_scale, (0, 0, 0), thickness + 8)
        cv2.putText(frame, text, (x, y), font, font_scale, (0, 255, 255), thickness)
        sub = "GET INTO POSITION"
        (sw, _), _ = cv2.getTextSize(sub, font, 0.9, 2)
        cv2.putText(frame, sub, ((w - sw) // 2, y + 60), font, 0.9, (0, 0, 0), 5)
        cv2.putText(frame, sub, ((w - sw) // 2, y + 60), font, 0.9, (0, 255, 255), 2)

    # Imbalance warning (only shown when triggered).
    warn = check_imbalance(counts)
    if warn:
        cv2.rectangle(frame, (0, h - 70), (w, h - 35), (0, 0, 60), -1)
        cv2.putText(frame, f"BALANCE: {warn}", (12, h - 47),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    # Bottom keybind strip.
    cv2.rectangle(frame, (0, h - 35), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, "1/2/3 class   SPACE save   B burst(5s)   BACKSPACE undo   q quit",
                (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)


def main() -> None:
    counts = init_counts(DATA_ROOT)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Is it connected? Is another app using it?")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)

    active_class = CLASSES[0]
    last_saved: Path | None = None  # session-only undo target
    flash_kind: str | None = None
    flash_frames_left = 0

    # Burst state machine: "idle" -> "countdown" -> "burst" -> "idle".
    # Driven by monotonic wall-clock time so frame-rate jitter can't desync it.
    burst_state = "idle"
    burst_start_time = 0.0       # monotonic anchor for the countdown
    burst_saves_done = 0
    burst_last_save_time = 0.0   # monotonic anchor for inter-shot spacing

    # Some webcams silently ignore the resolution request — verify and warn.
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (actual_w, actual_h) == (CAP_WIDTH, CAP_HEIGHT):
        res_note = ""
    else:
        res_note = f"  (requested {CAP_WIDTH}x{CAP_HEIGHT}, webcam ignored)"
    print(f"Capture res:     {actual_w}x{actual_h}{res_note}")
    print(f"Starting counts: {counts}")
    print(f"Active class:    {active_class}")
    print(f"Saving under:    {DATA_ROOT.resolve()}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            # Mirror for natural selfie-style UX; saved frame inherits the mirror.
            frame = cv2.flip(frame, 1)

            # Snapshot the clean frame BEFORE drawing the HUD.
            save_candidate = frame.copy()

            # Burst state machine — runs every frame, driven by wall-clock time.
            now = time.monotonic()
            countdown_remaining = 0
            if burst_state == "countdown":
                elapsed = now - burst_start_time
                if elapsed >= BURST_COUNTDOWN_SECONDS:
                    # Transition: countdown -> burst. Init save_time to 0.0
                    # so the very first burst shot fires this same iteration.
                    burst_state = "burst"
                    burst_saves_done = 0
                    burst_last_save_time = 0.0
                else:
                    # int(elapsed) rolls over once per whole second, so this
                    # displays 5 for [0,1), 4 for [1,2), ..., 1 for [4,5).
                    countdown_remaining = BURST_COUNTDOWN_SECONDS - int(elapsed)

            if burst_state == "burst":
                if burst_saves_done >= BURST_PHOTO_COUNT:
                    burst_state = "idle"
                    flash_kind = "BURST DONE"
                    flash_frames_left = FLASH_FRAMES
                elif (now - burst_last_save_time) >= BURST_INTERVAL_S:
                    path = save_frame(save_candidate, active_class, DATA_ROOT)
                    counts[active_class] += 1
                    last_saved = path
                    burst_saves_done += 1
                    burst_last_save_time = now

            if flash_frames_left > 0:
                flash_frames_left -= 1
            draw_hud(frame, active_class, counts, flash_kind, flash_frames_left,
                     burst_state, countdown_remaining, burst_saves_done)

            cv2.imshow("Phase 1 - Collect (q to quit)", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("1"):
                active_class = CLASSES[0]
            elif key == ord("2"):
                active_class = CLASSES[1]
            elif key == ord("3"):
                active_class = CLASSES[2]
            elif key == ord("b"):
                # Start a countdown from idle; cancel one already in progress.
                # Pressing B mid-burst is ignored — let the burst finish.
                if burst_state == "idle":
                    burst_state = "countdown"
                    burst_start_time = now
                elif burst_state == "countdown":
                    burst_state = "idle"
            elif key == ord(" ") and burst_state == "idle":
                path = save_frame(save_candidate, active_class, DATA_ROOT)
                counts[active_class] += 1
                last_saved = path
                flash_kind = "SAVED"
                flash_frames_left = FLASH_FRAMES
            elif key == 8 and burst_state == "idle":  # BACKSPACE
                if last_saved and last_saved.exists():
                    cls = last_saved.parent.name
                    last_saved.unlink()
                    counts[cls] = max(0, counts[cls] - 1)
                    last_saved = None
                    flash_kind = "DELETED"
                    flash_frames_left = FLASH_FRAMES
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Final counts:    {counts}")


if __name__ == "__main__":
    main()
