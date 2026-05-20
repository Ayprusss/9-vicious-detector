# ROCm on Windows — Training Environment Setup (Phase 3)

How to get PyTorch training/inference running on the **AMD Radeon RX 9060 XT**
(RDNA 4, `gfx1200`) **natively on Windows**, no WSL2 required.

> **TL;DR:** install AMD's ROCm-flavored PyTorch wheels into a dedicated
> `.venv-rocm/` venv. `torch.cuda.is_available()` returns `True` because ROCm
> deliberately exposes the CUDA API surface — your code says `device='cuda'`
> and it runs on the AMD GPU.

---

## Why native Windows (and not WSL2 / Colab)

When Phase 3 was first planned, the assumed path was **WSL2 + ROCm**, because
AMD's compute stack historically only ran on Linux. That changed: AMD now ships
**PyTorch-ROCm wheels that run directly on Windows** for RDNA 4 cards
(introduced ROCm 6.4.4, matured through 7.1.1+, RX 9060 XT officially supported
from ROCm 7.0.2 and on WSL2 from 7.2). Native Windows is the lighter path:

| Path | Setup cost | Notes |
|---|---|---|
| **Native Windows ROCm** *(chosen)* | ~20 min, just `pip install` | No Ubuntu, no `/mnt/` path juggling, no pressure on the small C: drive. Newest combo, so occasionally a rough edge. |
| WSL2 + ROCm 7.2 | ~an evening | Most battle-tested ML env, but a multi-GB Ubuntu+ROCm install; requires driver 26.1.1. Documented as the fallback. |
| Colab T4 | zero local | Doesn't use the local GPU; ~12h session cap. Last resort. |

---

## The two virtual environments in this repo

This project intentionally keeps **two** separate venvs. They have conflicting
dependency constraints (MediaPipe pins older `protobuf`/`numpy`), and MediaPipe
is only needed for the Phase 0 demo — not for training or YOLO inference.

| venv | Python | Installed from | Used by | GPU? |
|---|---|---|---|---|
| `.venv/` | 3.12.10 | `requirements.txt` (`opencv-python`, `mediapipe`) | Phase 0 `landmark_demo.py`, Phase 1 `collect.py` / `browse.py` | No (CPU) |
| `.venv-rocm/` | 3.12.10 | `requirements-rocm.txt` (PyTorch-ROCm + `ultralytics`) | Phase 3 training, Phase 4 inference | **Yes (AMD GPU via ROCm)** |

Both are gitignored. Always invoke a venv's Python by its full path rather than
activating per shell, e.g. `.venv-rocm\Scripts\python.exe scripts\train.py`.

---

## Prerequisites

1. **Python 3.12** (NOT 3.13). The wheels are `cp312`. Confirm:
   ```powershell
   py -3.12 --version    # -> Python 3.12.x
   ```
2. **AMD graphics driver >= 26.2.2** (Adrenalin Edition). ROCm 7.2.1's
   PyTorch-on-Windows release is built against driver 26.2.2. Check your
   installed marketing version:
   ```powershell
   (Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}\0000').RadeonSoftwareVersion
   ```
   If it's below 26.2.2, install the driver from
   <https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-WIN-26-2-2.html>
   (or newer) and reboot.
   - *This machine, verified 2026-05-19:* driver **26.5.2** (newer than 26.2.2 — OK).
3. **~3 GB free disk** for the wheels (the ROCm runtime is bundled inside them).

---

## Install

From the repo root (`9-vicious-detector/`):

```powershell
# 1. Create the dedicated training venv and upgrade pip
py -3.12 -m venv .venv-rocm
.venv-rocm\Scripts\python.exe -m pip install --upgrade pip

# 2. Install everything pinned in requirements-rocm.txt
.venv-rocm\Scripts\python.exe -m pip install --no-cache-dir -r requirements-rocm.txt
```

`requirements-rocm.txt` lists, in order:

- **ROCm 7.2.1 SDK wheels** — `rocm_sdk_core`, `rocm_sdk_devel`,
  `rocm_sdk_libraries_custom`, and `rocm-7.2.1.tar.gz`. These carry the ROCm
  runtime, since the full ROCm stack isn't natively packaged for Windows yet.
- **PyTorch-ROCm** — `torch 2.9.1+rocm7.2.1`, `torchvision 0.24.1`,
  `torchaudio 2.9.1` (all `cp312`, from `repo.radeon.com`).
- **Ultralytics** — installed last, so pip sees the ROCm `torch` already
  satisfies the dependency and doesn't pull a CPU-only `torch` over the top.

> If you prefer AMD's two-explicit-step form (SDK first, then PyTorch), the exact
> commands are in their docs (linked below). The single `-r` install does the
> same thing in one resolver pass.

---

## Verify (the go/no-go gate)

```powershell
.venv-rocm\Scripts\python.exe -c "import torch; print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
```

Expected:

```
cuda available: True
device: AMD Radeon RX 9060 XT
```

A quick on-device math check (proves kernels actually run, not just enumerate):

```powershell
.venv-rocm\Scripts\python.exe -c "import torch; x=torch.rand(2048,2048,device='cuda'); print('matmul ok:', (x@x).sum().item() > 0)"
```

> **Verified on this machine (2026-05-19):** `torch 2.9.1+rocm7.2.1`,
> `torch.version.hip = 7.2.53211`, `is_available() = True`, device =
> *AMD Radeon RX 9060 XT*, 15.9 GB VRAM, on-GPU matmul OK. Driver was Adrenalin
> 26.5.2 (already above the 26.2.2 requirement), so no driver install was needed.

---

## Troubleshooting

- **`torch.cuda.is_available()` is `False`.** Driver is suspect #1. Confirm the
  graphics driver is >= 26.2.2 (see Prerequisites). If on the mainline Adrenalin
  driver and it still fails, install AMD's dedicated *"AMD Software: PyTorch on
  Windows Edition"* driver, which is built specifically for the ROCm wheels.
  Reboot after any driver change.
- **`pip` can't find a matching wheel / "not a supported wheel on this platform".**
  You're not on Python **3.12** x64. The wheels are `cp312` / `win_amd64` only.
  Recreate the venv with `py -3.12`.
- **Ultralytics replaced torch with a CPU build.** Symptom: `is_available()` was
  `True`, then `False` after `pip install ultralytics`. Reinstall the torch wheel
  from `requirements-rocm.txt` and avoid `pip install --upgrade` against torch.
- **First GPU op is slow / compiles.** ROCm may JIT-compile kernels for `gfx1200`
  on first use. Subsequent runs are fast.
- **`HSA_OVERRIDE_GFX_VERSION`** — sometimes needed to spoof an architecture for
  *unsupported* cards. The RX 9060 XT (`gfx1200`) is officially supported on
  ROCm 7.2.1, so you should **not** need this. Only reach for it if a card isn't
  recognized at all.

---

## Sources

- AMD — *Install PyTorch for ROCm on Radeon (Windows, PIP)*:
  <https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html>
- AMD — *Windows support matrix by ROCm version*:
  <https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/windows/windows_compatibility.html>
- AMD — *Radeon graphics driver 26.2.2 release notes*:
  <https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-WIN-26-2-2.html>
- AMD GPUOpen — *Deploying LLMs with AMD on Windows using PyTorch*:
  <https://gpuopen.com/learn/pytorch-windows-amd-llm-guide/>
