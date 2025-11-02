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
  - Install Conda/Mamba (e.g., Mambaforge) and create env: `mamba create -n nerfstudio python=3.10 -y && mamba activate nerfstudio`
  - Install PyTorch with CUDA (match your driver via pytorch.org selector), then `pip install nerfstudio`
  - Validate GPU: `python -c "import torch;print(torch.cuda.is_available())"` should print `True`
  - Validate CLI: `ns-train --help` should display usage


Iteration #1 (no COLMAP)
- Input: a single image of an object.
- Generate views: create 12–24 views on a circular yaw orbit (fixed elevation 10–15°). You can stub this by duplicating the image to validate the flow, then replace with Zero123++ later.
- Poses: write `transforms.json` in Nerfstudio format (OpenGL camera-to-world). Use FOV ~55° if unknown; match your image width/height.
- Train: `ns-train splatfacto --data datasets/<object>` and consider `--pipeline.datamanager.camera-optimizer.mode SO3xR3` for small pose refinement.

Suggested Layout
- `datasets/<object>/source/` — original single image
- `datasets/<object>/images/` — generated views `view_000.png ...`
- `datasets/<object>/transforms.json` — Nerfstudio poses/intrinsics
- `results/<object>/` — training runs and exports

Typical Commands
- Generate views (placeholder duplicate): write N files to `datasets/<object>/images/`.
- Make poses: synthesize circular poses and write `datasets/<object>/transforms.json`.
- Train: `ns-train splatfacto --data datasets/<object>`
- View/export: use `ns-viewer`, `ns-render`, or `ns-export` to PLY/pointcloud.

Iteration #2 (optional COLMAP)
- Try COLMAP on the generated views and compare quality/stability. Synthetic multi-views may be inconsistent; COLMAP can be brittle, so treat it as an experiment.

Notes
- Prefer clean/segmented backgrounds for better reconstructions.
- Start with 12 views; scale to 24–48 if your generator supports it.
- On Windows, WSL2 + CUDA is recommended for PyTorch/Nerfstudio.
