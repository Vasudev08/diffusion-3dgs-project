Single Image Pipeline

Goal: start simple. Turn one image into 12–24 views, write Nerfstudio poses, and train Splatfacto. COLMAP comes later as an optional ablation.

Installation (WSL2 + Nerfstudio)
- Prereqs: Windows 11, latest NVIDIA GPU driver, WSL2 Ubuntu‑22.04.
- Cap resources in WSL Settings GUI:
  - Processor Count: 16
  - Memory Size: 16384 MB
  - Swap Size: 24576–32768 MB
  - Apply, then run `wsl --shutdown` and relaunch Ubuntu 22.04.
- Inside Ubuntu (keep project/data under `~`, not `/mnt/c`):
  - Follow Nerfstudio’s official installation guide (conda + dependencies as in docs). This project’s env was set up that way.
  - Install PyTorch with CUDA (per the PyTorch selector) and verify GPU: `python -c "import torch; print(torch.cuda.is_available())"` → `True`.
  - Validate CLI: `ns-train --help` should display usage.
- Keep this repo under your WSL home, e.g., `~/projects/<repo>`.


Current COLMAP Status
- Current preprocessing was generated with COLMAP on CPU (GPU disabled in the command).
- Next: test GPU acceleration for feature extraction/matching.
  - Ensure your COLMAP binary is built with CUDA support.
  - In this repo’s helper, remove the `--no-gpu` flag (or add a switch) before retesting.


