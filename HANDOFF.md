# HANDOFF.md

**Generated:** 2026-05-19 (end-of-session)
**Last completed:** Phase 3a — native-Windows ROCm training environment built and **verified working** on the AMD GPU.
**Currently in:** Pre-Phase 3b — environment is ready, no training script written yet.
**Up next:** Write `scripts/train.py`, kick off the first YOLO training run, review metrics.

> Fresh-session reader: `CLAUDE.md` + this file = full context. The single most
> important fact: **training runs natively on Windows now (no WSL2)** via the
> `.venv-rocm/` environment, and `torch.cuda.is_available()` is already confirmed
> `True` on the AMD RX 9060 XT.

## Session goal

Resume at Phase 3 and answer "what do we need to install to train on an AMD GPU?"
Concretely: pick and stand up a compute path for the RX 9060 XT, install the full
training stack, verify the GPU is actually usable from PyTorch, and document the
environment so it's reproducible. All achieved. Phase 3a is closed; Phase 3b
(the actual training run) is the next session's work.

## Current state of the code

| File | Status | Purpose |
|---|---|---|
| `CLAUDE.md` | **edited this session** | Project config. Environment section now documents BOTH venvs; Phase 3 rewritten WSL2 → native-Windows ROCm; Commands section adds the `.venv-rocm` recreate + verify + train commands; Phase 3 status set to `[~]` with 3a done / 3b pending; new Session Log entry added |
| `HANDOFF.md` | **rewritten this session** | This file |
| `ROCM_WINDOWS_SETUP.md` | **NEW this session** | Standalone install guide: why native Windows, the two-venv layout, prereqs (driver ≥ 26.2.2, Python 3.12), install steps, verification, troubleshooting, AMD doc sources. Has a "verified on this machine" stamp |
| `requirements-rocm.txt` | **NEW this session** | Pinned ROCm 7.2.1 wheel URLs (SDK + torch/vision/audio) + `ultralytics`, with a header explaining it's Windows/ROCm/cp312-specific and must go in `.venv-rocm/` |
| `requirements.txt` | unchanged | Phase 0 stack only: `opencv-python==4.13.0.92`, `mediapipe==0.10.35` (the `.venv/` env) |
| `.gitignore` | **edited this session** | Added `.venv-rocm/` (with comments distinguishing the two venvs) |
| `DATA_COLLECTION_NOTES.md` | unchanged | Phase 1 capture-discipline reference |
| `scripts/landmark_demo.py` | unchanged | Phase 0 MediaPipe webcam demo (uses `.venv/`) |
| `scripts/collect.py` | unchanged | Phase 1 collection script (uses `.venv/`) |
| `scripts/browse.py` | unchanged | Phase 1 QC grid viewer (uses `.venv/`) |
| `scripts/train.py` | **does not exist yet** | Phase 3b deliverable — write this next |
| `data/raw/<class>/*.jpg` | populated, gitignored | 416 raw frames: 136 other_hand / 143 sign_nine / 137 sign_ysl |
| `data/dataset/` | populated, gitignored | Roboflow YOLOv8 export. 1008 imgs (882 train / 84 val / 42 test). `data.yaml` paths already fixed (relative, not `../`) |
| `models/hand_landmarker.task` | unchanged, gitignored | MediaPipe model from Phase 0 |
| `.venv/` | unchanged, gitignored | Python 3.12.10. Phase 0/1 env: opencv + mediapipe. **Not** the training env |
| `.venv-rocm/` | **NEW this session, gitignored** | Python 3.12.10. Phase 3+ env: ROCm 7.2.1 + PyTorch-ROCm + Ultralytics. **This is the GPU/training env** |
| `runs/` | does not exist yet | Ultralytics will create it on first train → `runs/detect/train/weights/best.pt` |

## Files actively being edited

None. All documentation edits are saved. Working tree changes are committable but
**not yet committed** (user commits on request):
- Modified: `.gitignore`, `CLAUDE.md`
- New/untracked: `ROCM_WINDOWS_SETUP.md`, `requirements-rocm.txt`
- (`.venv-rocm/` is on disk but gitignored.)

## What we tried and what failed

### The "locked" WSL2 plan was based on a stale assumption (corrected)
The previous handoff locked in **WSL2 + ROCm** and even claimed "ROCm 6.2+ via WSL2."
Verifying against live AMD docs before installing revealed two errors:
1. RDNA 4 on **WSL2** actually needs **ROCm 7.2** (+ Adrenalin 26.1.1), not 6.2.
2. More importantly, AMD now supports **PyTorch-ROCm natively on Windows** for RDNA 4
   (`gfx1200`), so WSL2 is unnecessary entirely.
**Lesson:** re-verify fast-moving hardware/driver support against current docs
before committing to a multi-step install — a "locked" decision from a prior
session can be outdated. We surfaced the new option to the user, who chose native
Windows. WSL2 + ROCm 7.2 is retained only as a documented fallback.

### Bash tool vs PowerShell syntax (minor, recurring)
A couple of diagnostic commands were accidentally sent through the Bash tool and
failed on PowerShell-isms (`2>$null`, `Get-CimInstance`). **Lesson:** use the
PowerShell tool for PowerShell; the Bash tool runs POSIX bash.

### No real failures in the install itself
The install chain (SDK → torch → ultralytics) succeeded first try, exit code 0.

## What worked / decisions locked in this session

- **Compute path: native-Windows PyTorch-ROCm** into a dedicated `.venv-rocm/`.
  No WSL2, no Colab. Dataset on D: reads directly (no `/mnt/` indirection).
- **Two-venv architecture** (deliberate, documented): `.venv/` for the MediaPipe
  Phase 0 demo (its protobuf/numpy pins conflict with the ML stack), `.venv-rocm/`
  for training/inference. MediaPipe isn't needed for Phase 3+.
- **Install order matters:** torch BEFORE ultralytics, so pip sees `torch>=1.8.0`
  already satisfied by the ROCm build and does NOT overwrite it with a CPU wheel.
  Confirmed in the pip log (line: "Requirement already satisfied: torch ... 2.9.1+rocm7.2.1").
- **GO/NO-GO GATE PASSED** (the critical result): from a normal Windows shell,
  `torch.cuda.is_available()` → `True`, device = `AMD Radeon RX 9060 XT`,
  `15.9 GB` VRAM, `torch.version.hip = 7.2.53211`, and an on-GPU 4096×4096 matmul
  ran. The card is genuinely usable for training.
- **Driver was already fine:** Adrenalin **26.5.2** ≥ required 26.2.2 — no driver
  install needed. (Registry key `RadeonSoftwareVersion` is how we read the
  marketing version; WDDM `32.0.31007.5012` doesn't map obviously.)
- **VRAM confirmed 16 GB** (15.9 usable) — the WMI "4 GB" reading is the known
  under-reporting quirk. This means `batch=16` is conservative; we can go higher.

## Next steps (exact order if a fresh session picked this up)

1. **Write `scripts/train.py`** (per CLAUDE.md Phase 3b):
   ```python
   from ultralytics import YOLO
   model = YOLO('yolov8n.pt')        # COCO-pretrained nano (transfer learning)
   results = model.train(
       data='data/dataset/data.yaml',
       epochs=100, imgsz=640, batch=16,   # consider batch=32 or batch=-1 (auto) — 15.9 GB VRAM
       device=0, patience=20,
   )
   ```
   Store hyperparameters thoughtfully; CLAUDE.md prefers config-driven, but a
   first-pass script is fine. `yolov8n.pt` auto-downloads on first run.
2. **Run it:** `.venv-rocm\Scripts\python.exe scripts\train.py` (NOT `.venv`).
   Watch for the first-epoch ROCm kernel compilation pause; it's normal.
3. **Sanity-check the start:** confirm the run banner shows the AMD GPU and that
   loss is decreasing in the first few epochs before walking away.
4. **After training:** review `runs/detect/train/` — `results.png` (loss/mAP
   curves), `confusion_matrix.png`, `val_batch*_pred.jpg`. Target val mAP@0.5 ≥ 0.85.
5. **Diagnose by eye, not just metrics:** if `sign_ysl` ↔ `sign_nine` confusion
   shows in the matrix, that's a data problem (the known fine-grained-similarity
   risk) → targeted Phase 1 re-capture, not just more epochs.

## Open questions for the user

- **Batch size for the first run:** keep planned `batch=16`, or exploit the real
  15.9 GB VRAM with `batch=32` / `batch=-1` (Ultralytics auto)? (Asked at end of
  session; user opted to handoff first. Default to 16 if unspecified — safe, and
  isolates the GPU-works variable from the tuning variable on run #1.)
- **Model size:** plan starts at `yolov8n`; only escalate to `yolov8s`/yolov11 if
  nano underfits after data fixes. (No decision needed yet.)
- Long-standing carry-overs (not blocking): which friends for ~40 friend's-hand
  frames per positive class if overfitting shows; which Spotify track URI + YouTube
  URL are the Phase 5 trigger targets.

## Environment notes

- **OS / shell:** Windows 11 Pro, PowerShell primary. Use the PowerShell tool for
  PS commands; Bash tool is POSIX bash (don't mix syntaxes).
- **CPU:** AMD Ryzen 5 7500F. Virtualization on (irrelevant now — no WSL2 needed).
- **GPU:** AMD Radeon RX 9060 XT (RDNA 4, `gfx1200`), 16 GB VRAM (15.9 usable).
  Driver: Adrenalin **26.5.2**.
- **Disk:** C: ~65 GB free, D: ~768 GB free. Repo is on D:. (C: pressure was a
  concern for the abandoned WSL2 path; irrelevant now.)
- **Two Python envs, both Python 3.12.10, both gitignored, invoke by full path:**
  - `.venv/` — `requirements.txt`: `opencv-python==4.13.0.92`, `mediapipe==0.10.35`. CPU. Phase 0/1 scripts.
  - `.venv-rocm/` — `requirements-rocm.txt`. **GPU.** Phase 3+ training/inference. Key versions:
    - `torch==2.9.1+rocm7.2.1`, `torchvision==0.24.1+rocm7.2.1`, `torchaudio==2.9.1+rocm7.2.1`
    - `rocm==7.2.1` (+ `rocm-sdk-core/devel/libraries-custom==7.2.1`)
    - `ultralytics==8.4.51`, `numpy==2.4.6`, `opencv-python==4.13.0.92`
- **Wheels source:** `repo.radeon.com/rocm/windows/rocm-rel-7.2.1/` (Python 3.12 /
  win_amd64 only). Full URLs pinned in `requirements-rocm.txt`.
- **Recreate the training env from scratch:**
  ```powershell
  py -3.12 -m venv .venv-rocm
  .venv-rocm\Scripts\python.exe -m pip install --upgrade pip
  .venv-rocm\Scripts\python.exe -m pip install --no-cache-dir -r requirements-rocm.txt
  ```
- **Verify GPU (the gate):**
  ```powershell
  .venv-rocm\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  ```
  Expect: `True AMD Radeon RX 9060 XT`. If `False` → driver is suspect #1
  (see ROCM_WINDOWS_SETUP.md troubleshooting).
- **Git:** working tree has uncommitted doc changes (see "Files actively being
  edited"). Nothing committed this session — user commits on request.
- **Memory updated:** `project_gpu_amd.md` now reflects native-Windows ROCm (was
  stale: "train on Colab T4"). MEMORY.md index line updated to match.
