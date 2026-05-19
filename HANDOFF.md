# HANDOFF.md

**Generated:** 2026-05-19 (end-of-session)
**Last completed:** Phase 2 (Annotation + Roboflow export) — dataset on disk, `data.yaml` cleaned.
**Currently in:** Pre-Phase 3 — compute strategy decided, environment not yet built.
**Up next:** WSL2 + ROCm + PyTorch-ROCm setup, then write `scripts/train.py`, then train.

## Session goal

Push the project from "raw annotated images" to "trained `best.pt`-ready environment." Concretely:

1. Walk the user through Roboflow's Generate Version flow (preprocessing, augmentation, split)
2. Get the exported YOLOv8 dataset into the repo and verify its structure
3. Pick a Phase 3 compute path that fits the user's hardware

Phase 1 and Phase 2 both closed this session. Phase 3 setup is queued for tonight/tomorrow.

## Current state of the code

| File | Status | Purpose |
|---|---|---|
| `CLAUDE.md` | **edited this session** | Project config — hardware line corrected from "NVIDIA" → "AMD RX 9060 XT", Phase 1+2 marked complete with details, Phase 3 plan rewritten for WSL2+ROCm with a 3a/3b setup split, session log entry added for 2026-05-19 |
| `HANDOFF.md` | **rewritten this session** | This file |
| `.gitignore` | unchanged | Already covers `data/raw/`, `data/dataset/`, `.venv/`, `models/`, `runs/`, `*.pt`, `.env`, etc. Verified |
| `requirements.txt` | unchanged | `opencv-python==4.13.0.92`, `mediapipe==0.10.35` — Phase 3 will require adding `ultralytics`, `torch`, `torchvision` but those will be installed in a **WSL2 venv**, not the existing Windows `.venv/`. The Windows venv is fine as-is for collection/inference work |
| `scripts/landmark_demo.py` | unchanged | Phase 0 MediaPipe webcam demo |
| `scripts/collect.py` | unchanged | Phase 1 collection script |
| `scripts/browse.py` | unchanged | Mid-session QC grid viewer |
| `DATA_COLLECTION_NOTES.md` | unchanged | Capture-discipline reminders. Phase 1 done so this is now reference/history |
| `data/raw/<class>/*.jpg` | **populated, gitignored** | 416 captured frames: 136 other_hand / 143 sign_nine / 137 sign_ysl |
| `data/dataset/` | **NEW this session, gitignored** | Roboflow YOLOv8 export. 1008 total images (882 train / 84 valid / 42 test). Each split has parallel `images/` and `labels/` directories. `data.yaml` at root |
| `data/dataset/data.yaml` | **NEW this session, gitignored** | YOLO config. `nc: 3`, `names: ['other_hand', 'sign_nine', 'sign_ysl']`. **Paths were edited** from Roboflow's default `../train/images` → `train/images` (Roboflow assumes its data.yaml lives one level deeper than where unzip places it) |
| `models/hand_landmarker.task` | unchanged, gitignored | MediaPipe Hand Landmarker model from Phase 0 |
| `.venv/` | unchanged, gitignored | Python 3.12.10 Windows venv. Used for collection/QC scripts. **Will NOT be the training environment** — training runs inside a separate WSL2 Linux venv |
| `runs/` | does not exist yet | Will be created by Ultralytics on first training run. Path: `runs/detect/train/weights/best.pt` |

## Files actively being edited

None. All in-flight edits committed (`d0ac02f`, `da5e72c`). Working tree is clean.

## What we tried and what failed

### Initial Phase 3 compute recommendation (overcautious, reversed)

When the user asked "what happens if I have an AMD GPU?", first response defaulted to **Google Colab free T4** as the recommendation, dismissing local AMD options as "driver hell, skip for a learning project." User pushed back appropriately — they had just upgraded from a 2070 SUPER to an RX 9060 XT and wanted to use the hardware. Re-evaluated and reversed the recommendation to **WSL2 + ROCm**, which is a legitimate path on RDNA 4 with ROCm 6.2+.

**Lesson:** Don't reflexively recommend cloud compute when the user has capable local hardware. The AMD ML toolchain has matured significantly in 2024–2025; ROCm-on-WSL2 is officially supported and the setup is a one-evening investment, not a quagmire. Memory saved at `project_gpu_amd.md` to prevent this misframe in future sessions.

### Roboflow `data.yaml` path bug (resolved)

Roboflow exports `data.yaml` with paths like `train: ../train/images`. The `../` assumes the file lives inside a deeper subfolder, but `unzip` lands it at the same level as `train/`. Ultralytics would then look for `data/train/images` instead of `data/dataset/train/images` and fail.

**Resolution:** Stripped `../` from all three split paths. Now reads `train/images`, `val: valid/images`, `test: test/images`. Verified to resolve correctly relative to `data.yaml`'s location.

## What worked / decisions locked in this session

### Phase 1 closeout

- **Final raw counts:** 136 other_hand / 143 sign_nine / 137 sign_ysl = **416 images** (~150/class, softened from the original 250/class target)
- User explicit framing: *"If the model works, great. If it doesn't, I learn the data-collection lesson. Both are wins."* Right mindset for an iterative ML loop.
- Class balance came out tight: max-to-min gap is 7 frames (~5%), well inside the ±10% tolerance.

### Phase 2 closeout

- Roboflow project: `anthonys-workspace-yiji7/nine-vicious-detector`, version 1
- Two accidental classes purged from the schema before generating version → final `nc: 3` is clean
- **Generate Version settings locked in:**
  - Preprocessing: Auto-Orient ON, Resize Stretch to 640×640
  - Augmentations: horizontal flip ON, vertical flip OFF, rotation ±10°, brightness ±15%, blur ≤2px
  - 3x outputs per training example (only training split is augmented)
  - 70/20/10 train/val/test split
- Output: 1008 images (882 train / 84 val / 42 test). Val/test stay un-augmented so metrics remain honest.

### Phase 3 compute decision

- **Path chosen: WSL2 + ROCm + PyTorch-ROCm + Ultralytics on Ubuntu 22.04** inside WSL2
- Rationale: AMD officially supports RDNA 4 on ROCm 6.2+ via WSL2. RX 9060 XT is more capable than Colab's free T4 in raw compute. After ~1 evening of setup, every future training run is fast and local. Skill transfers to other ML projects.
- ROCm reuses the CUDA API in PyTorch — code that says `device='cuda'` runs on the AMD GPU through ROCm. YOLO training code is portable between NVIDIA and AMD.

## Next steps (exact order if a fresh session picked this up)

1. **Confirm Windows-side prerequisites.** AMD Adrenalin driver current (Settings → AMD Software → updates). Virtualization enabled in BIOS (almost certainly already on, but worth a glance at Task Manager → Performance → CPU → "Virtualization: Enabled").
2. **Confirm ~50 GB free on C: drive.** WSL2's VHDX lives there by default. Ubuntu + ROCm + PyTorch end up ~15–20 GB.
3. **Enable WSL2 + install Ubuntu 22.04.** From elevated PowerShell: `wsl --install -d Ubuntu-22.04`. Reboot when prompted. Create Ubuntu user on first launch.
4. **Install ROCm inside Ubuntu.** Follow AMD's official ROCm-on-WSL guide (URL drifts — search "ROCm WSL2 install" or check rocm.docs.amd.com). Verify with `rocminfo | grep gfx` — should list a `gfx12xx` agent.
5. **Create a Python venv inside WSL** at `~/venvs/9-vicious`, install PyTorch-ROCm via the official wheel index (`pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.x` — check pytorch.org/get-started/locally for the current ROCm version pin), then `pip install ultralytics`.
6. **Verify GPU detection** from inside WSL: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` — must print `True` and the AMD device name. **This is the green-light moment** — if this fails, the rest won't work.
7. **Write `scripts/train.py`** (10 lines: `YOLO('yolov8n.pt').train(data='data/dataset/data.yaml', epochs=100, imgsz=640, batch=16, device=0, patience=20)`).
8. **Run training from inside WSL** at `/mnt/d/coding_files/computer-vision-projects/9-vicious-detector/`. The WSL auto-mount means no dataset copying needed.
9. **Review `runs/detect/train/`** — `results.png`, `confusion_matrix.png`, `val_batch*_pred.jpg`. `best.pt` is the artifact to keep for Phase 4.

## Open questions for the user

None blocking. The Phase 3 compute path is locked, all data is in place. Tonight's session is a guided setup walkthrough — main unknown is whether the user hits driver/version pain during ROCm install. If so, fall back paths in order: `torch-directml` on Windows (slower, requires Ultralytics device patches) → Google Colab T4 (zero setup, ~30 min training).

Long-standing carry-overs (not blocking Phase 3):
- Which 2–3 friends are realistic asks for ~40 friend's-hand frames per positive class? (Only relevant if Phase 3 training reveals a "this is only your hand" overfitting failure mode that needs Phase 1 re-capture.)
- Which Spotify track URI and which YouTube video URL are the Phase 5 trigger targets?

## Environment notes

- **OS:** Windows 11 Pro, PowerShell as primary shell
- **Local Python (Windows-side):** 3.12.10 at `.venv/`. Used for `scripts/collect.py`, `scripts/browse.py`, `scripts/landmark_demo.py`. **Not the training environment.**
- **Training Python (planned, WSL2):** Will be created at `~/venvs/9-vicious` inside Ubuntu 22.04 WSL2. PyTorch-ROCm + Ultralytics installed there. **Do NOT install training deps into the Windows `.venv/`** — they won't see the GPU.
- **Hardware:** AMD Radeon RX 9060 XT (RDNA 4, gfx12xx). 16 GB VRAM (WMI reports 4 GB but that's a driver-reporting quirk on modern cards).
- **GPU on Windows:** `torch.cuda.is_available()` will be False from a Windows shell — no CUDA path here. This is expected.
- **GPU in WSL2 + ROCm:** `torch.cuda.is_available()` will be True (ROCm intentionally exposes the CUDA API surface). `device='cuda'` and `device=0` work and resolve to the AMD GPU.
- **Spotify Premium:** account ready for Phase 5.
- **`.gitignore` covers:** `.venv/`, `models/`, `data/raw/`, `data/dataset/`, `.env`, `.spotify_cache`, `runs/`, `*.pt`, `logs/`, IDE folders. Verified clean: all 1008 dataset images and 416 raw images on disk are properly excluded.
- **Git state:** Working tree clean. Two commits made this session — `d0ac02f data collection complete - updating claude` and `da5e72c additional changes to reflect hardware restrictions`. Everything to do with Phase 1+2 closure and the WSL2+ROCm decision is committed.
- **Memory saved:** `project_gpu_amd.md` captures the AMD hardware reality + the WSL2+ROCm decision + the "don't repeat the dismissive framing" lesson. Future sessions will read this on startup.

## Variation matrix (historical, Phase 1 complete)

Carry-over from earlier handoffs. Phase 1 is closed, but if Phase 3 training reveals a failure mode (e.g., "model only recognizes signs at one lighting level"), the gaps in this matrix tell you where to capture more:

- **Lighting:** bright daylight / dim daylight / lamp-only / overhead artificial / side-lit
- **Distance:** arm's length / close-up / far from camera / partially out of frame
- **Angle:** straight on / tilted 15° / tilted 30° / profile / hand rotated
- **Background:** clean wall / cluttered desk / bed / kitchen / outdoors
- **Clothing:** ≥2 outfits with different sleeve cuffs
- **Hand:** both dominant and non-dominant hand
- **Person:** ideally a friend's hand for ≥10% of frames
