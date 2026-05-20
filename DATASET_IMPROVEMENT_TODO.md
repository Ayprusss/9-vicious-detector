# Dataset Improvement TODO

**Status:** Deferred / backlog. The current model (`runs/detect/train/weights/best.pt`)
is **good enough for this personal "gag" project** and Phase 5 proceeds with it.
This file captures how to make the detector more robust *if* it's ever revisited —
it is **not** blocking anything.

> Context: the v1 dataset was 416 images, **one person, mostly one room**, with
> burst-captured near-duplicate frames. Validation mAP@0.5 was 0.992, but that was
> optimistic because the val set looked almost identical to the train set (same
> face, same wall, leaked near-duplicates). Live webcam testing exposed the gaps
> below — exactly the "data distribution = model distribution" lesson in practice.

## Observed failure modes (from live testing, 2026-05-19)

1. **Face detected as `sign_nine` (false positive).**
   The negative class (`other_hand`) didn't contain enough face/no-hand examples,
   so the model never learned that "a face is not a sign." It fills the gap by
   guessing `sign_nine`.
2. **Some angles/orientations of `sign_nine` and `sign_ysl` are missed (recall gap).**
   Poses, rotations, and distances that were underrepresented (or absent) in the
   416 training images don't get detected — the model only confidently recognizes
   the poses it actually saw.

## Improvement backlog (roughly highest-impact first)

- [ ] **Add hard negatives to fix the face false-positive.** Collect more
      `other_hand` / no-hand frames: bare face at many angles, face + non-sign hand
      poses, empty scene, other body parts, objects. Hard negatives are the direct
      cure for "X gets misread as a sign." This is the single highest-leverage fix.
- [ ] **Broaden pose coverage for the two real signs.** Re-capture `sign_nine` and
      `sign_ysl` deliberately across the axes that are currently thin: rotation
      (hand tilted/rolled), distance (close + far), camera angle (above/below/side),
      partial occlusion, both hands. Target the *specific* angles observed to fail.
- [ ] **Kill near-duplicate leakage so metrics stop lying.** The burst-capture
      frames (~10 shots / 150 ms) are near-identical; random Roboflow splitting
      scatters them across train/val/test, inflating scores. Either (a) dedupe
      near-duplicates before splitting, or (b) split by *capture session* so all
      frames of one continuous pose stay in the same split. Then the next mAP number
      is trustworthy.
- [ ] **Add lighting + background variety.** Daylight, lamp-only, overhead, side-lit;
      cluttered vs. clean backgrounds; different rooms. Defeats shortcut learning
      (model latching onto "the wall" instead of the hand).
- [ ] **(Optional, deprioritized — personal project)** Multi-person data. Capturing a
      friend's hand would fix single-person overfit, but is intentionally skipped
      since this build is for the user alone.
- [ ] **Re-train and compare apples-to-apples.** After the above, regenerate the
      Roboflow version, retrain, and compare on a *clean* (leak-free) test split.
      Also eyeball live behavior — the honest test, not just the metric.
- [ ] **(Optional) Try a larger backbone** (`yolov8s`) only *after* the data is
      fixed. A bigger model can't fix a data-coverage problem; data comes first.

## Quick reference

- Collect more raw frames: `.venv\Scripts\python.exe scripts\collect.py`
- QC the captures: `.venv\Scripts\python.exe scripts\browse.py`
- Re-annotate/version/export in Roboflow → `data/dataset/` (fix `data.yaml` paths)
- Retrain: `.venv-rocm\Scripts\python.exe scripts\train.py`
