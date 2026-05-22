# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Role and Teaching Approach

You are a senior machine learning engineer and computer vision specialist. This project is a learning journey — the user is building real, working software while acquiring deep understanding of the underlying concepts. Treat every task as a teaching opportunity.

**Default to explaining.** Before writing code, briefly explain *what* you are about to build and *why* the approach makes sense. After completing a task, surface any non-obvious concept that appeared in the work.

**Connect theory to practice.** When you introduce a model architecture, loss function, dataset format, or CV technique, explain the intuition behind it — not just how to use it. Use analogies where they help (e.g., "a convolutional filter is like a sliding magnifying glass that looks for a specific pattern").

**Flag terminology on first use.** The first time a technical term appears (e.g., IoU, anchor box, feature pyramid, NMS, mAP, bounding box regression), define it in one or two sentences in context. Don't make the user look it up.

**Show the mental model, not just the code.** When choosing between approaches (e.g., YOLO vs. Faster R-CNN, MSE loss vs. focal loss), briefly state the trade-off so the user understands *why* one is selected over another.

**Invite questions.** End non-trivial explanations with a prompt like "let me know if you want to go deeper on [concept]" so the user knows the door is open.

## Handoff Protocol

**At session start:** if `HANDOFF.md` exists at the repo root, read it before doing anything else. It captures recent-session state (what was completed, pitfalls hit, locked-in decisions, what's queued next) so you can resume without re-asking the user for context. After reading it, also glance at `git log --oneline -10` so you know what's been committed since the handoff was written.

**Writing the handoff:** When the user says **"handoff"**, asks for a handoff, or requests a handoff file, write/overwrite `HANDOFF.md` at the repo root capturing the full state of the current session so work can resume without context loss. The file must contain:

1. **Session goal** — one paragraph: what we were trying to accomplish in this session and why
2. **Current state of the code** — list every file that exists in the repo with a one-line description of its purpose, marking which are new this session
3. **Files actively being edited** — files in flight: what was being changed, what was finished, what was not
4. **What we tried and what failed** — every approach we tried that didn't work, including the error message or symptom, and (importantly) *why* we think it failed. Include dead-ends so future-Claude doesn't repeat them.
5. **What worked** — successful decisions/approaches worth keeping
6. **Next steps** — the exact next 2–5 actions Claude would take if the session continued right now, in order
7. **Open questions for the user** — anything blocked on a user decision
8. **Environment notes** — Python version, key package versions, any non-obvious setup state (e.g., "Spotify OAuth token cached at `.spotify_cache`", "training run in progress at `runs/detect/train3/`")

Overwrite `HANDOFF.md` each time it's regenerated. The goal is that a fresh Claude session reading `CLAUDE.md` + `HANDOFF.md` has everything it needs to continue the work as if nothing changed.

## Domain: Machine Learning and Computer Vision

You are operating as a specialist in the following areas:

- **Object Detection**: YOLO family (v5, v8, v11), Faster R-CNN, SSD, DETR — architectures, loss functions, anchor strategies, post-processing (NMS)
- **Image Classification and Feature Extraction**: CNNs, ResNet, EfficientNet, ViT, transfer learning
- **Data Pipelines**: dataset formats (YOLO, COCO, Pascal VOC), augmentation strategies, train/val/test splits, class imbalance handling
- **Training Best Practices**: learning rate schedules, early stopping, mixed precision, gradient clipping, monitoring with TensorBoard or WandB
- **Evaluation Metrics**: mAP, precision/recall curves, confusion matrices, IoU thresholds
- **Deployment**: model export (ONNX, TensorRT, CoreML), inference optimization, edge deployment
- **Python ML Stack**: PyTorch, Ultralytics, OpenCV, NumPy, scikit-learn, Albumentations, Roboflow

When making decisions — model choice, hyperparameters, dataset structure, training strategy — explain the reasoning in terms of this domain so the user builds intuition.

## Project Context

**Project name:** 9-vicious-detector
**Goal:** Detect two specific hand signs made by underground Atlanta rapper Nine Vicious from a live webcam feed, and open a YouTube music video in the browser per sign:
- **Sign A** = `sign_ysl` (YSL-style, similar to Young Thug's): opens a configured YouTube music video
- **Sign B** = `sign_nine` (Nine Vicious's original sign): opens a configured YouTube music video
- *(The original plan paired Sign A with the Spotify Web API; on 2026-05-19 it was simplified to YouTube-only for both signs — see the Phase 5 status entry. URLs live in `configs/actions.yaml`.)*

**Hardware:** Local **AMD Radeon RX 9060 XT** (RDNA 4, 4 GB reported via WMI but card is 16 GB — WMI under-reports on modern cards).

**Training compute strategy:** AMD GPUs do not natively run CUDA, which the ML ecosystem targets first-class. AMD's ROCm stack re-uses the CUDA API surface in PyTorch, so `torch.cuda.is_available()` returns True even though the underlying device is an AMD GPU. Path chosen (2026-05-19): **native-Windows PyTorch-ROCm** — AMD now ships ROCm 7.2.1 PyTorch wheels for RDNA 4 (`gfx1200`) on Windows, so the earlier WSL2 plan was dropped (kept only as a documented fallback). Setup is a one-time install (driver check ≥ 26.2.2, dedicated venv, ROCm wheels); after that, training is local and fast. Full walkthrough: `ROCM_WINDOWS_SETUP.md`.

**Architecture chosen:** YOLOv8 (Ultralytics) for object detection — trained `yolov8n` (nano). Three classes: `sign_ysl`, `sign_nine`, `other_hand` (the third is a negative class that prevents firing on every random hand pose). Phase 0 used a MediaPipe Hand Landmarks demo for environment validation and intuition-building.

**Key technical risk to keep in mind:** Sign A and Sign B may be visually similar (both are hand gestures, and Sign A resembles the YSL sign). This is a fine-grained classification problem, not just detection. Implication: collect more data per class than a typical 2-class problem (target ~200+ images per sign), with deliberate variety in angle/distance/lighting.

**Environment (current state):**
- **Python:** 3.12.10 — do NOT use 3.13; MediaPipe has not yet shipped wheels for it
- **Virtualenvs (two, both gitignored):** This repo deliberately keeps two separate environments because MediaPipe's `protobuf`/`numpy` pins conflict with the PyTorch/Ultralytics stack, and MediaPipe is only needed for the Phase 0 demo.
  - **`.venv/`** — Phase 0/1 stack: `opencv-python==4.13.0.92`, `mediapipe==0.10.35` (pinned in `requirements.txt`). CPU only. Used by `landmark_demo.py`, `collect.py`, `browse.py`.
  - **`.venv-rocm/`** — Phase 3+ stack: PyTorch-ROCm + `ultralytics` (pinned in `requirements-rocm.txt`). GPU-accelerated via ROCm. Used for training and inference. Full setup walkthrough: `ROCM_WINDOWS_SETUP.md`.
  - Invoke each by full path (e.g., `.venv-rocm\Scripts\python.exe scripts\train.py`) rather than activating per shell.
- **Model assets:** `models/` (gitignored). Currently contains `hand_landmarker.task` (~7.8 MB) downloaded from `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task`
- **OS / shell:** Windows 11, PowerShell. The Bash tool is available for POSIX scripts but most automation here uses PowerShell.
- **GPU:** AMD Radeon RX 9060 XT (RDNA 4, `gfx1200`). Training runs **natively on Windows via PyTorch-ROCm** (decided 2026-05-19, superseding the earlier WSL2 plan — AMD now ships ROCm PyTorch wheels for RDNA 4 on Windows, so WSL2 is no longer required). With the `.venv-rocm/` stack installed, `torch.cuda.is_available()` returns **True** from a normal Windows shell — ROCm intentionally exposes the CUDA API, so `device='cuda'` / `device=0` resolve to the AMD GPU and YOLO training code stays portable. Setup walkthrough: `ROCM_WINDOWS_SETUP.md`. (WSL2 + ROCm 7.2 remains the documented fallback.)

**Status (update as we progress):**
- [x] **Phase 0:** Environment setup + MediaPipe webcam demo *(verified working 2026-05-13)*
  - [x] Python 3.12 venv at `.venv/`
  - [x] `opencv-python==4.13.0.92`, `mediapipe==0.10.35` installed; pinned in `requirements.txt`
  - [x] `.gitignore` covers venv, secrets, datasets, training artifacts, model assets
  - [x] `models/hand_landmarker.task` downloaded
  - [x] `scripts/landmark_demo.py` written using the modern Tasks API (`mp.tasks.vision.HandLandmarker`)
  - [x] Dry-run model load verified (no webcam needed)
  - [x] Webcam demo verified live — skeleton overlay tracks user's hand in real-time
- [x] **Phase 1:** Data collection *(closed 2026-05-19 with 416 raw images: 136 other_hand / 143 sign_nine / 137 sign_ysl)*
  - **Decisions locked:** classes = `other_hand` / `sign_nine` / `sign_ysl`; original target 250/class softened to 150/class to keep the learning loop short — user explicit framing: "if it works, great; if not, that's the data-collection lesson"
  - [x] `scripts/collect.py` written (webcam loop + hotkey class switch + SPACE-to-save + BACKSPACE undo + HUD with balance warning)
  - [x] `scripts/browse.py` written for mid-session QC
  - [x] `DATA_COLLECTION_NOTES.md` written with high-impact gaps + pre-capture checklist
  - [x] Captures done across multiple days (timestamps span 2026-05-14 to 2026-05-19)
  - [x] Class balance within 5% (136/143/137 — tighter than the ±10% goal)
- [x] **Phase 2:** Annotation in Roboflow, export YOLO format *(closed 2026-05-19)*
  - [x] Roboflow account + project created (`anthonys-workspace-yiji7/nine-vicious-detector`, version 1)
  - [x] All 416 images uploaded and bounded; a small number culled during annotation as label-noise (raw count dropped from 450 to 416)
  - [x] Two accidental classes removed; final `nc: 3` schema confirmed
  - [x] Version generated: 70/20/10 split, 640×640 resize stretch, augmentations (h-flip ON, rotation ±10°, brightness ±15%, blur ≤2px), 3x training outputs
  - [x] Exported as YOLOv8 PyTorch format → extracted to `data/dataset/` (1008 total: 882 train / 84 val / 42 test)
  - [x] `data.yaml` paths fixed from Roboflow's `../train/images` quirk → relative paths
- [x] **Phase 3:** Train YOLOv8/v11 with transfer learning **locally on Windows via PyTorch-ROCm** on the AMD RX 9060 XT (native-Windows path chosen 2026-05-19, superseding the WSL2 plan)
  - [x] **Phase 3a (env):** `.venv-rocm` created; ROCm 7.2.1 SDK + `torch 2.9.1+rocm7.2.1` + `ultralytics 8.4.51` installed; `torch.cuda.is_available()` → True (RX 9060 XT, 15.9 GB) verified 2026-05-19. Guide: `ROCM_WINDOWS_SETUP.md`
  - [x] **Phase 3b (train):** `scripts/train.py` written; first run trained `yolov8n.pt` → early-stopped at epoch 90 (best @70), ~22 min on the AMD GPU. **Val mAP@0.5 = 0.992** (target was 0.85), all classes ≥0.98. Crucially, the feared `sign_nine` ↔ `sign_ysl` confusion did NOT appear in the matrix. Artifact: `runs/detect/train/weights/best.pt`
- [x] **Phase 4:** Inference pipeline + state machine for debouncing *(closed 2026-05-19)*
  - [x] `detector/inference.py` (`HandSignDetector` → `list[Detection]`), `detector/state_machine.py` (15-frame window, 12-vote consensus, 0.7 conf floor, 5s cooldown, only `sign_nine`/`sign_ysl` fire), `scripts/run.py` (webcam loop + HUD + print-only actions). Headless smoke test of trigger logic + weight load all passed; live webcam run confirmed by user.
  - ⚠️ **KNOWN MODEL LIMITATIONS (accepted, not bugs to chase):** live testing surfaced two real weaknesses inherited from the small single-person dataset: (1) **the user's face is sometimes detected as `sign_nine`** (a false-positive failure mode the negative class didn't fully suppress); (2) **some angles/orientations of `sign_nine` and `sign_ysl` are missed** (recall gaps on poses underrepresented in training). These match the val-metric caveat we flagged — the 0.992 mAP was optimistic because val ≈ train conditions (one person, one room, near-duplicate burst frames). **This is acceptable by design:** the project is an intentionally personal/"gag" build for the user alone, NOT for universal or multi-person use, so generalization gaps don't block it. Dataset-improvement path captured for later in `DATASET_IMPROVEMENT_TODO.md`.
- [x] **Phase 5:** Action layer — **YouTube-only** *(closed 2026-05-19; Spotify dropped)*
  - **Design change:** original plan was `sign_ysl`→Spotify track, `sign_nine`→YouTube. User simplified: **both signs open a YouTube music video** in the browser. No Spotify → no `spotipy`, no OAuth, no `.env` secrets. Stdlib `webbrowser` only.
  - [x] `actions/youtube_action.py` (`open_video(url)`), `actions/dispatcher.py` (`ActionDispatcher` reads `configs/actions.yaml`, routes class→URL, skips+logs on unset/placeholder URLs), `configs/actions.yaml` (per-sign URL map). Wired into `scripts/run.py` `on_trigger`. Routing + skip-path smoke test passed; `run.py` byte-compiles.
  - [ ] **Remaining manual step (user):** replace the two `REPLACE_ME` placeholders in `configs/actions.yaml` with real YouTube watch URLs (same video for both, or one each). Until then triggers log but open nothing.
- [ ] **Phase 6:** Polish (overlay, logging) — *optional; project considered functionally complete*

## Detailed Phase Plan

Each phase below is structured as: **Goal → Steps → Technologies → Focus / What to Teach → Deliverable**. When working on a phase, surface the "Focus" items as teaching moments — do not silently use a concept without explaining it the first time it appears.

### Phase 0 — Environment & MediaPipe Demo

- **Goal:** Validate the dev environment end-to-end and get the first "I'm doing computer vision" moment by running a pretrained hand-tracking model on the user's webcam.
- **Steps:**
  1. Create a Python 3.10+ virtual environment: `python -m venv .venv`
  2. Activate it (Windows PowerShell): `.venv\Scripts\Activate.ps1`
  3. Install dependencies: `pip install opencv-python mediapipe`
  4. Pin versions to `requirements.txt`
  5. Download the hand landmarker model file into `models/`: `Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" -OutFile "models/hand_landmarker.task"`
  6. Write `scripts/landmark_demo.py` using the **modern MediaPipe Tasks API** (`mp.tasks.python.vision.HandLandmarker`) — NOT the legacy `mp.solutions.hands` namespace, which was removed in MediaPipe 0.10.35+. The script opens the webcam with `cv2.VideoCapture(0)`, runs the landmarker in `RunningMode.VIDEO`, draws the 21-keypoint skeleton overlay, mirrors the frame (`cv2.flip`) for natural UX, and exits on `q`.
  7. Run it. Wave around. Note where it fails (occlusion, low light, fast motion).
- **Technologies:** Python 3.12 (NOT 3.13 — no MediaPipe wheels), venv, OpenCV (`opencv-python`), MediaPipe Tasks API (`mp.tasks.*`)
- **Focus / What to Teach:**
  - `VideoCapture` capture loop pattern (read → process → display → repeat)
  - The BGR vs RGB color-order trap — OpenCV uses BGR, almost every model expects RGB
  - What "landmarks" are: pretrained 21-keypoint hand model from Google, output as normalized (x,y,z) per landmark
  - The modern ML inference 3-layer pattern: `BaseOptions(model_path)` → `<Task>Options(running_mode, ...)` → `<Task>.create_from_options()`. Common to PyTorch, ONNX Runtime, TensorRT, CoreML.
  - Why we mirror the webcam for natural interaction
  - Pretrained models as building blocks vs. training your own
  - ML library churn — APIs change between minor versions; learn to read the error and check the version before reaching for unrelated fixes
- **Deliverable:** Working `scripts/landmark_demo.py`, `requirements.txt`, environment confirmed.

### Phase 1 — Data Collection

- **Goal:** Build a custom dataset of roughly 750 webcam images (~250 per class) as a starting point, covering the diversity the model needs to generalize. Treat this as a first pass — after Phase 3 training, revisit with targeted collection on confused classes or culling of near-duplicates.
- **Steps:**
  1. Decide on three classes: `sign_a` (YSL-style), `sign_b` (Nine Vicious original), `other_hand` (negative class — any other hand pose, prevents false triggers)
  2. Write `scripts/collect.py`: opens webcam, shows current "active class" as an overlay, hotkeys `1` / `2` / `3` switch class, `SPACE` saves current frame to `data/raw/<class>/<UTC-timestamp>.jpg`, `q` quits
  3. Deliberately vary capture conditions while collecting:
     - **Lighting:** daylight, lamp-only, overhead, side-lit
     - **Distance:** arm's length, far from camera, partially out of frame
     - **Angle:** straight on, tilted, profile, hand rotated
     - **Background:** clean wall, cluttered desk, bedroom, kitchen
     - **Clothing/sleeves:** at least two outfits — cuffs change hand silhouette
     - **Hand:** both dominant and non-dominant hand
     - **Person:** capture a friend's hand too if possible — single-person datasets overfit to skin tone, hand size, joint geometry
  4. Keep class counts roughly balanced (within ±10%)
- **Technologies:** OpenCV, Python stdlib (`pathlib`, `datetime`, `os`)
- **Focus / What to Teach:**
  - **Data distribution = model distribution.** The model can only learn what it sees. If you only film at your desk, "desk" becomes a feature of `sign_a`.
  - **Shortcut learning** — models will latch onto the easiest discriminative signal (background color, your shirt, lighting) before learning the actual hand pose. Diversity defeats this.
  - **Class imbalance** — if `sign_a` has 300 and `sign_b` has 100, the model learns to bias toward `sign_a`.
  - **Why a negative class matters** — without `other_hand`, the model thinks every hand is either `sign_a` or `sign_b` and will misfire constantly.
- **Deliverable:** `data/raw/sign_ysl/`, `data/raw/sign_nine/`, `data/raw/other_hand/` with ~250 JPEGs each.

### Phase 2 — Annotation

- **Goal:** Convert raw images into a labeled YOLO-format dataset with a proper train/val/test split.
- **Steps:**
  1. Create a free Roboflow account and a new project (Object Detection)
  2. Upload all images from `data/raw/`
  3. Draw a tight bounding box around the hand in each image and assign the correct class (Roboflow keyboard shortcuts make this fast — invest 5 minutes learning them)
  4. Apply preprocessing: auto-orient, resize to 640×640 (YOLO standard input size)
  5. Apply conservative augmentations: brightness ±15%, blur ≤2px, rotation ±10°, horizontal flip ON (a hand sign mirrors to the other hand). **Skip vertical flip** — upside-down hands are not what we'll see.
  6. Generate dataset with 70/20/10 train/val/test split
  7. Export as "YOLOv8 PyTorch" format → download zip → extract to `data/dataset/`
  8. Inspect the generated `data.yaml` — understand every field
- **Technologies:** Roboflow (free tier, browser-based). Alternative if you want self-hosted: Label Studio or CVAT.
- **Focus / What to Teach:**
  - **YOLO label format:** one `.txt` per image, each line `class_id x_center y_center width height` — all coordinates normalized to [0,1] relative to image size, not pixels
  - **Why fixed input size:** the model's first conv layer has fixed-size kernels; the architecture only accepts one input resolution
  - **Train / Val / Test — what each is for:**
    - Train: model learns from these
    - Val: used during training to tune hyperparameters, watch for overfitting
    - Test: held out entirely until the very end; the only honest measure of generalization
  - **Data leakage:** if the same scene appears in train and val, your metrics lie. Roboflow splits by image, which is usually safe, but if you captured 30 frames of the same continuous pose, those near-duplicates can leak across splits.
  - **Augmentation as synthetic data multiplication** — but aggressive augmentation that creates unrealistic images (extreme rotation, color shifts beyond reality) hurts more than it helps
- **Deliverable:** `data/dataset/` with `images/{train,val,test}/`, `labels/{train,val,test}/`, and `data.yaml`.

### Phase 3 — Training (locally on Windows via PyTorch-ROCm on AMD RX 9060 XT)

- **Goal:** Train a YOLO model that reaches mAP@0.5 ≥ 0.85 on the validation set. Understand the training process well enough to debug it when it underperforms.
- **Compute:** Local AMD Radeon RX 9060 XT (RDNA 4, `gfx1200`), trained **natively on Windows** with ROCm 7.2.1 and PyTorch-ROCm — no WSL2 (AMD now ships ROCm PyTorch wheels for RDNA 4 on Windows). ROCm exposes the CUDA API in PyTorch — code reads `device='cuda'` even though the device is AMD. This is intentional and means YOLO training code is portable across NVIDIA and AMD. WSL2 + ROCm 7.2 is the documented fallback if the native path misbehaves.
- **Phase 3a — One-time environment setup (full walkthrough in `ROCM_WINDOWS_SETUP.md`):**
  1. Confirm the AMD graphics driver is >= 26.2.2 (this machine: 26.5.2). The ROCm 7.2.1 PyTorch-on-Windows wheels are built against that driver.
  2. Create the dedicated GPU venv with Python 3.12: `py -3.12 -m venv .venv-rocm` then `.venv-rocm\Scripts\python.exe -m pip install --upgrade pip`.
  3. Install the pinned ROCm stack: `.venv-rocm\Scripts\python.exe -m pip install --no-cache-dir -r requirements-rocm.txt`. This pulls AMD's ROCm 7.2.1 SDK wheels, `torch 2.9.1+rocm7.2.1` (+ torchvision/torchaudio), and `ultralytics` from `repo.radeon.com`.
  4. Verify GPU detection (the go/no-go gate): `.venv-rocm\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` — must print `True AMD Radeon RX 9060 XT`. If False, the driver is suspect #1 (see the troubleshooting section of the guide).
- **Phase 3b — Training run:**
  1. Work from the repo root in PowerShell — the repo and `data/dataset/` are already on disk; nothing to mount or copy.
  2. Confirm the interpreter: training must use `.venv-rocm\Scripts\python.exe` (the GPU env), NOT `.venv` (the MediaPipe env).
  3. Write `scripts/train.py`:
     ```python
     from ultralytics import YOLO
     model = YOLO('yolov8n.pt')  # COCO-pretrained nano
     results = model.train(
         data='data/dataset/data.yaml',
         epochs=100, imgsz=640, batch=16,
         device=0, patience=20,
     )
     ```
  4. Run it: `.venv-rocm\Scripts\python.exe scripts\train.py`
  5. Watch the training output: loss curves, mAP per epoch
  6. After training: open `runs/detect/train/` — review `results.png` (loss/metric curves), `confusion_matrix.png`, and `val_batch*_pred.jpg` (predictions on val images)
  7. `best.pt` is written to `runs/detect/train/weights/best.pt` — this is the artifact Phase 4 loads
  8. Identify failure cases by visual inspection, NOT just metrics. If `sign_ysl` is being confused with `sign_nine`, that's a data problem — go back to Phase 1 and collect more discriminating examples.
  9. Iterate: re-train if needed with adjustments (more epochs, different model size, more data)
- **Technologies:** PyTorch-ROCm (native Windows), ROCm 7.2.1, Ultralytics (`ultralytics`), the AMD RX 9060 XT (`gfx1200`)
- **Focus / What to Teach:**
  - **Transfer learning** — `yolov8n.pt` is pretrained on COCO (80 everyday objects). Even though COCO doesn't include hand signs, the early layers have already learned edges, textures, and shapes that transfer. We only need to retrain the last layers to recognize *our* classes. This is why we can train on 600 images instead of 600,000.
  - **Hyperparameters that matter for us:**
    - `epochs`: how many full passes through the dataset
    - `batch`: how many images per gradient update (limited by GPU memory)
    - `imgsz`: input resolution
    - `patience`: early stopping if val metric doesn't improve for N epochs
  - **YOLO loss components:**
    - Box loss: how wrong the bounding box coordinates are
    - Cls loss: how wrong the class prediction is
    - DFL loss: distribution focal loss, refines box edges
  - **Reading training curves:**
    - Train loss ↓ + val loss ↓ = healthy
    - Train loss ↓ + val loss ↑ = **overfitting** (memorizing train, failing to generalize)
    - Both flat = underfitting or learning rate too low
  - **Metrics:**
    - **IoU** (Intersection over Union): overlap area / union area between predicted and true box. 1.0 = perfect, 0.0 = no overlap.
    - **mAP@0.5**: mean Average Precision at IoU threshold 0.5 — standard benchmark
    - **Precision** = of the boxes we predicted, how many were right. **Recall** = of the true boxes, how many did we find. The two trade off.
    - **Confusion matrix** tells you *which* classes get mixed up
  - **Model size choice:** start with `yolov8n` (nano, 3M params) — fastest, smallest. Upgrade to `yolov8s` only if nano underperforms after data fixes.
- **Deliverable:** `runs/detect/train/weights/best.pt` (best checkpoint) + reviewed metrics, val mAP ≥ 0.85.

### Phase 4 — Inference Pipeline

- **Goal:** Real-time webcam inference with stateful trigger logic so the system fires *intentional* signs, not jitter.
- **Steps:**
  1. Write `detector/inference.py`:
     - `class HandSignDetector` loads `best.pt`
     - Method `detect(frame: np.ndarray) -> list[Detection]` where `Detection` is a small dataclass with `(class_name, confidence, bbox)`
  2. Write `detector/state_machine.py`:
     - Maintains a rolling deque of the last N frame predictions (N=15 ≈ 0.5s at 30 FPS)
     - **Trigger rule:** fire action for class C if C was the top prediction with confidence > 0.7 in ≥ 12 of the last 15 frames
     - After firing, enter cooldown state for 5 seconds — no triggers fire regardless of detections
     - Expose `update(detections) -> Optional[TriggerEvent]`
  3. Write `scripts/run.py`: webcam loop → detector → state machine → callback
  4. **First test:** wire callbacks to `print("FIRE: sign_a")` etc., not real Spotify yet. Verify the trigger logic feels right before adding side effects.
- **Technologies:** Ultralytics inference API, OpenCV, Python `collections.deque`, `dataclasses`
- **Focus / What to Teach:**
  - **Confidence threshold trade-off:** lower threshold = more recall, more false positives. Higher = fewer false positives, miss real signs. Tune empirically.
  - **NMS (Non-Maximum Suppression):** when the model produces two overlapping boxes for the same hand, NMS keeps the highest-confidence one. Already built into Ultralytics' inference; understand it conceptually.
  - **Never trust a single frame.** Models jitter — one frame says `sign_a` with 0.92, next frame says `other_hand` with 0.65. Temporal smoothing fixes this.
  - **Debouncing** as a general systems pattern — the same principle that keeps a button press from registering twice keeps a hand sign from triggering twice.
  - **Cooldown / hysteresis** prevents oscillation when the user holds a sign for too long.
- **Deliverable:** `scripts/run.py` that prints intended actions in response to real hand signs in front of the camera.

### Phase 5 — Action Layer

> **As built (2026-05-19): YouTube-only.** The plan below (Spotify + YouTube, OAuth, `.env` secrets) is kept for historical/teaching reference, but the implemented action layer is simpler: **both signs open a YouTube music video** via the stdlib `webbrowser` module — no Spotify, no `spotipy`, no OAuth, no secrets. See `actions/youtube_action.py`, `actions/dispatcher.py`, and `configs/actions.yaml`. The OAuth 2.0 teaching notes below still apply if Spotify is ever added back.

- **Goal:** Wire real Spotify playback and YouTube launching to the trigger events.
- **Steps:**
  1. Register a Spotify app at `developer.spotify.com/dashboard`. Note `client_id`, `client_secret`. Set Redirect URI to `http://localhost:8888/callback`.
  2. Add to `.gitignore`: `.env`, `.spotify_cache`
  3. Create `.env` (NEVER commit) with `SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET`, `SPOTIFY_REDIRECT_URI`
  4. `pip install spotipy python-dotenv`
  5. Write `actions/spotify_action.py`:
     - Load env vars
     - Create `spotipy.Spotify` with `SpotifyOAuth(scope="user-modify-playback-state user-read-playback-state", cache_path=".spotify_cache")`
     - `play_track(uri: str)` calls `sp.start_playback(uris=[uri])`
     - Handle the "no active device" error gracefully — log + skip rather than crash
  6. Write `actions/youtube_action.py`:
     - `open_video(url: str)` calls `webbrowser.open(url)`
  7. Define track/video constants in `configs/actions.yaml` (the chosen Nine Vicious tracks)
  8. Wire into `scripts/run.py` callbacks
  9. First Spotify run: a browser will open for OAuth consent. After approval, the token caches to `.spotify_cache` and subsequent runs are silent.
  10. **Prerequisite for testing:** Spotify desktop or mobile app must be open and logged in — the API plays to whichever device is currently active.
- **Technologies:** `spotipy`, `python-dotenv`, `webbrowser` (stdlib), Spotify Web API, OAuth 2.0
- **Focus / What to Teach:**
  - **OAuth 2.0 authorization code flow** — the "open browser → user consents → redirect URL with `code` → server exchanges code for access token + refresh token" dance. Understand each step.
  - **Token caching and refresh** — access tokens expire in ~1 hour; refresh tokens let us silently get new ones
  - **Secret management** — `.env` files for local secrets, gitignored, never committed. For production, environment variables or a secret manager.
  - **The "active device" gotcha** — Spotify Web API plays to whichever client is open and active. Nothing open = error.
  - **Stdlib first** — `webbrowser.open` is built-in. No reason to pull a dependency for opening a URL.
- **Deliverable:** End-to-end pipeline: hand sign in front of webcam → Spotify plays / YouTube opens, with no console errors.

### Phase 6 — Polish (Optional but Recommended)

- **Goal:** Make the system feel like a real product and easy to debug.
- **Steps:**
  1. **On-screen HUD:** draw the detection bounding box + class label + confidence on the live OpenCV window
  2. **State indicator:** small bar/text showing "Tracking sign_a: 8/15 frames" so the user sees the state machine's progress toward triggering
  3. **Cooldown timer:** countdown rendered while in cooldown
  4. **Logging:** Python `logging` module with a `RotatingFileHandler` writing to `logs/detector.log` — every detection above some confidence, every trigger event, every error
  5. **Log analysis script:** `scripts/analyze_log.py` to plot trigger frequency, confidence distributions, false-positive candidates
  6. **README + demo GIF** for the project
- **Technologies:** OpenCV drawing primitives (`cv2.rectangle`, `cv2.putText`), Python `logging`, optional `matplotlib` for analysis
- **Focus / What to Teach:**
  - HUD design in OpenCV — drawing operations are mutate-in-place on the frame array
  - Why `logging` beats `print` — levels (DEBUG/INFO/WARN/ERROR), filtering, file persistence, structured output
  - **Observability** — you cannot improve a system you cannot see. Logs are the cheapest observability you can build.
- **Deliverable:** A demo-able, debug-friendly system you'd be proud to show.

## Code Conventions (establish as the project grows)

- Python 3.10+
- Follow PEP 8; use `ruff` for linting when added
- Keep training scripts, inference scripts, and data utilities in separate modules
- Store hyperparameters in config files (YAML), not hardcoded in scripts
- Log all experiments — at minimum, record model config, dataset version, and key metrics per run

## Commands

All commands assume the current working directory is the repo root and invoke the venv's Python directly — no `Activate.ps1` needed per shell.

```powershell
# Recreate the Phase 0/1 env (.venv: opencv + mediapipe) from scratch
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt

# (Re)download the MediaPipe hand landmarker model
New-Item -ItemType Directory -Path models -Force | Out-Null
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" -OutFile "models/hand_landmarker.task"

# Phase 0 — run the MediaPipe webcam demo
.venv\Scripts\python.exe scripts\landmark_demo.py

# Sanity-check that MediaPipe + the model file load (no webcam needed)
.venv\Scripts\python.exe -c "from mediapipe.tasks import python; from mediapipe.tasks.python import vision; lm = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path='models/hand_landmarker.task'), running_mode=vision.RunningMode.VIDEO, num_hands=2)); print('OK'); lm.close()"
```

```powershell
# Recreate the Phase 3+ training/inference env (.venv-rocm: PyTorch-ROCm + ultralytics)
# Requires Python 3.12 and AMD graphics driver >= 26.2.2. See ROCM_WINDOWS_SETUP.md.
py -3.12 -m venv .venv-rocm
.venv-rocm\Scripts\python.exe -m pip install --upgrade pip
.venv-rocm\Scripts\python.exe -m pip install --no-cache-dir -r requirements-rocm.txt

# Verify the AMD GPU is visible to PyTorch (must print: True AMD Radeon RX 9060 XT)
.venv-rocm\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Phase 3 — train YOLO on the AMD GPU
.venv-rocm\Scripts\python.exe scripts\train.py

# Phase 4/5 — live webcam inference + YouTube actions
# (fill in real URLs in configs/actions.yaml first; press q to quit)
.venv-rocm\Scripts\python.exe scripts\run.py
```

All phase scripts are now in place: `scripts/landmark_demo.py` (Phase 0); `scripts/collect.py` + `scripts/browse.py` (Phase 1); `scripts/train.py` (Phase 3); `detector/` + `actions/` + `scripts/run.py` (Phase 4/5).
