"""
Phase 1 — Quick grid viewer for captured images.

Browse images saved by scripts/collect.py so systematic issues
(hand cut off at the edge, focus locked on background, autoexposure
crushing details, sign form drifting) get caught at 30 frames instead
of at 300.

Keybindings:
  1 / 2 / 3      switch class (sign_ysl / sign_nine / other_hand)
  n / SPACE      next page
  p              previous page
  q / ESC        quit

Usage:
  .venv\\Scripts\\python.exe scripts\\browse.py            # start on sign_ysl
  .venv\\Scripts\\python.exe scripts\\browse.py sign_nine  # start on a specific class
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

DATA_ROOT = Path("data/raw")
CLASSES = ("sign_ysl", "sign_nine", "other_hand")

GRID_COLS = 4
GRID_ROWS = 3
PAGE_SIZE = GRID_COLS * GRID_ROWS

THUMB_W = 320
THUMB_H = 180         # 16:9 — matches the 1280x720 capture aspect ratio
HEADER_H = 40
LABEL_H = 22

CANVAS_W = GRID_COLS * THUMB_W
CANVAS_H = HEADER_H + GRID_ROWS * (THUMB_H + LABEL_H)


def list_images(class_name: str) -> list[Path]:
    d = DATA_ROOT / class_name
    if not d.exists():
        return []
    return sorted(d.glob("*.jpg"))


def render_page(class_name: str, images: list[Path], page: int) -> np.ndarray:
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

    if not images:
        cv2.putText(canvas, f"{class_name}: no images yet", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2)
        cv2.putText(canvas, "(run scripts/collect.py to start capturing)",
                    (12, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 140, 140), 1)
        return canvas

    total_pages = max(1, (len(images) + PAGE_SIZE - 1) // PAGE_SIZE)
    start = page * PAGE_SIZE
    end = min(start + PAGE_SIZE, len(images))

    header = f"{class_name}   |   page {page + 1}/{total_pages}   |   showing {start + 1}-{end} of {len(images)}"
    cv2.putText(canvas, header, (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    for i in range(start, end):
        path = images[i]
        img = cv2.imread(str(path))
        if img is None:
            continue
        thumb = cv2.resize(img, (THUMB_W, THUMB_H))

        cell = i - start
        row = cell // GRID_COLS
        col = cell % GRID_COLS
        x = col * THUMB_W
        y = HEADER_H + row * (THUMB_H + LABEL_H)

        canvas[y:y + THUMB_H, x:x + THUMB_W] = thumb

        # Label strip directly below each thumbnail.
        label_y = y + THUMB_H
        cv2.rectangle(canvas, (x, label_y), (x + THUMB_W, label_y + LABEL_H),
                      (30, 30, 30), -1)
        # Last 6 chars of the timestamp filename are microseconds — short + distinctive.
        label = f"#{i + 1}   ...{path.stem[-6:]}"
        cv2.putText(canvas, label, (x + 6, label_y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return canvas


def main() -> None:
    start_class = CLASSES[0]
    if len(sys.argv) > 1:
        if sys.argv[1] in CLASSES:
            start_class = sys.argv[1]
        else:
            print(f"Unknown class '{sys.argv[1]}'. Valid: {', '.join(CLASSES)}. Defaulting to {start_class}.")

    class_idx = CLASSES.index(start_class)
    page = 0

    while True:
        class_name = CLASSES[class_idx]
        images = list_images(class_name)
        total_pages = max(1, (len(images) + PAGE_SIZE - 1) // PAGE_SIZE)
        page = max(0, min(page, total_pages - 1))

        canvas = render_page(class_name, images, page)
        cv2.imshow("Browse captures (q to quit)", canvas)

        # waitKey(0) blocks — no CPU spin between key presses.
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), 27):  # q or ESC
            break
        elif key == ord("1"):
            class_idx = 0
            page = 0
        elif key == ord("2"):
            class_idx = 1
            page = 0
        elif key == ord("3"):
            class_idx = 2
            page = 0
        elif key in (ord("n"), ord(" ")):
            if total_pages > 0:
                page = (page + 1) % total_pages
        elif key == ord("p"):
            if total_pages > 0:
                page = (page - 1) % total_pages

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
