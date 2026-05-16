# HANDOFF.md

**Generated:** 2026-05-13 (updated mid-Phase 1)
**Last completed:** Phase 0 (Environment + MediaPipe webcam demo) — verified working.
**Currently in:** Phase 1 (Data collection) — collection + browse scripts written, capture not yet started.
**Up next:** Capture reference frames, then first real collection session.

## Session goal

Design and build the Phase 1 data-collection workflow:
- `scripts/collect.py` — webcam loop, hotkey class switching, SPACE-to-save, BACKSPACE undo, HUD with per-class counts and balance warning.
- `scripts/browse.py` — grid viewer for mid-session QC of captured images.
- `DATA_COLLECTION_NOTES.md` — high-impact gaps the user must stay mindful of during capture.

No images captured yet. Capture begins next session.

## Current state of the code

| File | Status | Purpose |
|---|---|---|
| `CLAUDE.md` | unchanged | Project config — mentor instructions, domain framing, full phase plan, status checklist, session log, handoff protocol |
| `.gitignore` | unchanged | Ignores `.venv/`, secrets, `data/`, training artifacts, model files, logs, IDE folders |
| `requirements.txt` | unchanged | `opencv-python==4.13.0.92`, `mediapipe==0.10.35` |
| `scripts/landmark_demo.py` | unchanged | Phase 0 MediaPipe webcam demo |
| `scripts/collect.py` | **NEW this session** | Phase 1 collection script. Webcam loop, hotkeys `1/2/3` switch class, SPACE saves, BACKSPACE undoes last save (session-only), `q` quits. WYSIWYG mirrored saves. HUD drawn AFTER frame copy → saves are pristine (no overlay contamination). Balance warning at 20+ frames + >20% gap between max/min class. Prints actual capture resolution on startup. |
| `scripts/browse.py` | **NEW this session** | Grid viewer for QC. 4×3 thumbnails per page, `1/2/3` switches class, `n`/SPACE next page, `p` previous, `q`/ESC quit. Optional CLI arg: starting class. |
| `DATA_COLLECTION_NOTES.md` | **NEW this session** | High-impact gaps + pre-capture checklist + dataset composition reminders. Read before every capture session. |
| `models/hand_landmarker.task` | unchanged, gitignored | MediaPipe Hand Landmarker model |
| `.venv/` | unchanged, gitignored | Python 3.12.10 virtualenv |

## Files actively being edited

None — `scripts/collect.py` and `scripts/browse.py` are complete. No images captured yet; `data/raw/` doesn't exist until `collect.py` is run once.

## What we tried and what failed

Nothing failed this session. All design choices held up under discussion and were confirmed via AskUserQuestion prompts. Carried-over pitfalls from Phase 0 remain documented in `CLAUDE.md`'s Session Log.

## What worked / decisions locked in this session

### `scripts/collect.py` design choices (all explicitly confirmed)

- **Save the mirrored (displayed) frame**, not raw camera frame. WYSIWYG — composition in the preview matches what gets saved. Roboflow's horizontal-flip augmentation in Phase 2 produces the un-mirrored version for free.
- **BACKSPACE = undo last save** (session-only; only saves made this session can be undone). Re-arms to `None` after one use to prevent double-deletes.
- **Imbalance warning in HUD** — when any class has ≥20 frames AND (max − min) / max > 0.20, show "BALANCE: X ahead of Y by N%" in a red strip near the bottom. Stays silent until thresholds are met to avoid noisy warnings on tiny counts.
- **HUD drawn on a `.copy()` taken AFTER frame capture, BEFORE annotation.** Critical: saving an annotated frame would let the model learn "ACTIVE: sign_ysl" text in the corner as a shortcut feature. Worth one extra numpy copy per frame.
- **UTC microsecond timestamps** for filenames (`YYYYMMDDTHHMMSSffffff.jpg`) — no collisions on rapid bursts, sorts chronologically, embeds capture time.
- **Capture resolution: 1280×720** requested explicitly. Script reads back actual resolution and prints "requested X, webcam ignored" if the webcam silently downscaled.
- **Counts initialize by scanning `data/raw/<class>/`** so the per-class display is correct across sessions.

### `scripts/browse.py` design choices

- 4×3 grid of 320×180 thumbnails per page (12 per page, matches 16:9 capture aspect).
- Header: `<class>  |  page X/Y  |  showing A-B of N`.
- Each thumbnail labeled `#index   ...<last-6-chars-of-timestamp>` for findability.
- Blocking `cv2.waitKey(0)` so the viewer doesn't peg CPU between key presses.
- "No images yet" placeholder for empty class folders (handles fresh-clone case).

### Conceptual decisions reached in conversation (worth re-stating)

- **`other_hand` should be heavy on hard negatives.** Target mix: ~50% similar/competing hand signs (peace, OK, rock-on, other rap signs), ~20% transitional/mid-motion poses, ~15% random gestures, ~15% mundane (typing/resting). Easy negatives alone teach the model nothing — it needs to see the cases that look like the positives but aren't.
- **Distribution of context (face/body presence, framing) must match across classes.** Otherwise the model learns shortcuts like "face visible → sign_ysl." Plan: pick ~3 framing styles (close-up hand-only / mid-shot with face / wide off-center) and apply each in similar proportions to all three classes.
- **Pose variation within a class is the goal, not a problem.** Different angles/tilts/orientations teach pose invariance. Filter rule: if a stranger viewing the photo couldn't tell which sign it is, it's label noise — skip.
- **In-plane rotation gets some help from Phase 2 augmentation (±10°). Out-of-plane rotation (palm vs. back of hand) must be captured explicitly** — no augmentation can synthesize 3D viewpoint changes from a single 2D image.
- **Horizontal flip is free via Phase 2 augmentation** — don't waste captures shooting mirror-image versions of poses you already have.

## High-impact gaps documented in `DATA_COLLECTION_NOTES.md`

1. Capture 3–5 reference frames per sign **first** as canonical anchors against form drift.
2. Spread captures across **3+ sessions on different days** for free distribution variation.
3. Reserve a **held-out test set** (~50–80 images, ~15–25 per class) under deliberately different conditions; manually assign in Roboflow.
4. **Back up `data/raw/`** after every session — it's gitignored.
5. **QC pause** after ~30 frames per class — run `scripts/browse.py` and skim.
6. **Hand-size floor:** hand should occupy ≥5% of frame (≈200×200 at 1280×720).
7. **Friend's-hand commitment:** ~40 of 400 per positive class from a friend's hand.

## Next steps (exact order if a fresh session picked this up)

1. **Capture reference frames first.** Create `data/reference/sign_ysl/` and `data/reference/sign_nine/`. With no other goal in mind, capture 3–5 of the cleanest-possible execution of each sign. Either capture via `scripts/collect.py` and manually move the files, or use any image tool. These anchors prevent form drift across multi-day capture.
2. **First real collection session.** `.venv\Scripts\python.exe scripts\collect.py`. Aim for ~30 per class. Rotate through `1` / `2` / `3` rather than batching by class. Keep the variation matrix in mind.
3. **QC pause.** `.venv\Scripts\python.exe scripts\browse.py`. Skim each class. Identify and stop any systematic issue before scaling up.
4. **Back up `data/raw/`** to external storage.
5. **Day 2 of capture:** different room, different lighting, different outfit.
6. **Once cumulative count is ~300 per class:** do the held-out test session under deliberately different conditions. Note which filenames belong to the test split (you'll manually assign them in Roboflow during Phase 2).
7. **Day 3+ until ~400 per class** with friend's-hand frames mixed in for the positive classes.

## Open questions for the user

- Which 2–3 friends are realistic asks for the ~40 friend's-hand frames per positive class?
- Which ≥3 rooms / lighting setups will you cycle through?
- Which Spotify track URI and which YouTube video URL are the trigger targets? (Not needed until Phase 5, but easier to lock in while listening to Nine Vicious.)

## Environment notes

- **OS:** Windows 11, PowerShell
- **Python:** 3.12.10 (system also has 3.13.7 — **do not use 3.13**, MediaPipe ships no wheels for it)
- **Venv:** `.venv/` at repo root. Invoke via `.venv\Scripts\python.exe <script>` (no per-shell activation needed).
- **Installed (top-level):** `opencv-python==4.13.0.92`, `mediapipe==0.10.35`. NumPy is transitively present via either.
- **Model file:** `models/hand_landmarker.task` (~7.8 MB, gitignored). Re-downloadable from `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task`.
- **GPU:** NVIDIA available locally (needed for Phase 3 training).
- **Spotify Premium:** account available (needed for Phase 5 playback control).
- **`.gitignore` covers:** `.venv/`, `models/`, `data/raw/`, `data/dataset/`, `.env`, `.spotify_cache`, `runs/`, `*.pt`, `logs/`, IDE folders.
- **Git state:** Phase 0 work was uncommitted as of the previous handoff. This session adds `scripts/collect.py`, `scripts/browse.py`, `DATA_COLLECTION_NOTES.md`, and a rewrite of `HANDOFF.md` on top of that. User may want to commit before next capture session so disk loss doesn't take all the code with it too.

## Variation matrix (target during collection, carried from prior handoff)

Aim to hit every cell at least once per class:
- **Lighting:** bright daylight / dim daylight / lamp-only / overhead artificial / side-lit
- **Distance:** arm's length / close-up / far from camera / partially out of frame
- **Angle:** straight on / tilted 15° / tilted 30° / profile / hand rotated
- **Background:** clean wall / cluttered desk / bed / kitchen / outdoors
- **Clothing:** ≥2 outfits with different sleeve cuffs
- **Hand:** both dominant and non-dominant hand
- **Person:** ideally a friend's hand for ≥10% of frames (concrete plan in `DATA_COLLECTION_NOTES.md`)
