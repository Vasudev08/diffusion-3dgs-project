"""
Reusable pipeline functions for the single-image → multi-view → Nerfstudio flow.

This module intentionally contains only function skeletons with clear docstrings
and type hints so you can implement each step incrementally. See
`single_image_pipeline/README.md` for the expected dataset and results layout.

Typical order of calls
- generate_views() → generate_transforms() → train_splatfco() → export_ply()
 - generate_views() → generate_transforms() → train_splatfco() → export_ply()
 - Alternatively for real multi-view: multiview_colmap() → train_splatfco() → export_ply()

Folder layout (as referenced by parameters)
- datasets/<object>/source/            # original single image
- datasets/<object>/images/            # generated views (view_000.png, ...)
- datasets/<object>/transforms.json    # Nerfstudio camera poses + intrinsics
- results/<object>/...                 # training runs, logs, exports
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union


PathLike = Union[str, Path]


def generate_views(
    image_path: PathLike,
    views_dir: PathLike,
    *,
    num_views: int = 12,
    method: str = "duplicate",
    size: Optional[int] = None,
    seed: Optional[int] = None,
    force: bool = False,
) -> List[Path]:
    """Generate multiple views from a single input image.

    Purpose
    - Produce N images under `views_dir` to simulate or synthesize multi-view data.
    - Start simple by duplicating the source image; later swap in a real generator
      (e.g., Zero123++, StableZero123, or your own model/API).

    Parameters
    - image_path: Path to the single input image (e.g., datasets/<obj>/source/img.png).
    - views_dir: Output directory to write generated views (e.g., datasets/<obj>/images/).
    - num_views: Number of views to generate (e.g., 12, 24).
    - method: Generation strategy. Suggest "duplicate" initially; later accept
      names like "zero123", "pipelineX" and branch accordingly.
    - size: Optional square resize (e.g., 512 or 1024). If provided, implementations
      can resize outputs to this resolution.
    - seed: Optional RNG seed for reproducibility in stochastic generators.
    - force: If True, overwrite/replace existing generated images. If False, allow
      idempotent skips when outputs already exist.

    Returns
    - List of paths to generated images (ordered), typically named
      views_dir/view_000.png ... view_{num_views-1:03d}.png

    Notes
    - Keep outputs in a flat folder, referenced relatively by transforms.json as
      "images/<filename>" from the dataset root.
    - This function should not perform camera pose math; that belongs to
      `generate_transforms`.
    """
    # TODO: Implement generation logic.
    # Suggested steps:
    # 1) Create views_dir
    # 2) If method == "duplicate": copy the source image num_views times with
    #    deterministic names (view_000.png, ...)
    # 3) Optionally resize to `size`
    # 4) Return the list of Path objects
    # For now, return an empty list as a placeholder.
    return []

def generate_transforms(
    images_dir: PathLike,
    transforms_path: PathLike,
    *,
    fov_deg: float = 55.0,
    elevation_deg: float = 12.0,
    orbit: str = "circular",
    up: Tuple[float, float, float] = (0.0, 1.0, 0.0),
    lookat: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    radius: float = 1.0,
    force: bool = True,
) -> Path:
    """Create Nerfstudio `transforms.json` for the generated images.

    Purpose
    - Synthesize camera poses on a simple orbit and write Nerfstudio-compatible
      transforms.json referencing the files inside `images_dir`.

    Parameters
    - images_dir: Folder containing generated views (PNG/JPG). File names must match
      what `generate_multiple_image` produced.
    - transforms_path: Where to write transforms.json (typically datasets/<obj>/transforms.json).
    - fov_deg: Horizontal field of view assumption if intrinsics unknown (e.g., 55°).
    - elevation_deg: Fixed camera elevation above the horizon in degrees (e.g., 10–15°).
    - orbit: Pose layout strategy. Keep "circular" for v0; later you may add/accept
      "spiral", or a custom list of angles.
    - up: World up vector in OpenGL convention (default Y-up).
    - lookat: Scene center the cameras should look at.
    - radius: Orbit radius around the lookat point (scene normalized units).
    - force: Overwrite existing transforms.json if True; otherwise validate and skip.

    Returns
    - Path to the written transforms.json file.

    Notes
    - Use OpenGL camera-to-world matrices (right-handed, -Z forward, Y up).
    - Each frame entry: {"file_path": "images/view_000.png", "transform_matrix": [...]}.
    - Include intrinsics fields: fl_x, fl_y, cx, cy, w, h. When unknown, compute focal
      from fov_deg and image width.
    """
    # TODO: Implement pose synthesis and JSON writing.
    # Suggested steps:
    # 1) Enumerate images in images_dir in sorted order
    # 2) Read image size (w, h) from one file
    # 3) Compute focal length from FOV: fx = 0.5 * w / tan(0.5 * fov)
    # 4) For each index i, compute yaw angle and build a c2w matrix at elevation_deg
    # 5) Write transforms.json with intrinsics and frames
    # For now, return the provided path as a Path object.
    return Path(transforms_path)


def multiview_colmap(
    image_sets: PathLike,
    dataset_dir: PathLike,
    *,
    matcher: str = "exhaustive",
    use_gpu: bool = True,
    colmap_cmd: str = "colmap",
    ns_process: bool = True,
    force: bool = True,
) -> Path:
    """Run COLMAP on one or more image folders and produce Nerfstudio transforms.

    Purpose
    - Take real or generated multi-view images, run feature extraction + matching +
      SfM reconstruction with COLMAP, and write a Nerfstudio-ready dataset under
      `dataset_dir` containing `images/` and `transforms.json`.

    Parameters
    - image_sets: Either a single images directory or a sequence of directories.
      You can pass multiple folders if you have separate capture sets to merge.
    - dataset_dir: Output root for the processed dataset. Expected to contain
      an `images/` folder and a `transforms.json` after processing.
    - matcher: COLMAP matching strategy (e.g., "sequential", "exhaustive", "vocab_tree").
    - camera_model: Camera model for feature extraction (e.g., "PINHOLE", "OPENCV").
    - single_camera: Assume shared intrinsics across all images (recommended for turntable/handheld).
    - camera_params: Optional explicit camera params list if you want to fix intrinsics.
    - use_gpu: Use GPU-accelerated extraction/matching if available.
    - image_downscale: Optional integer downscale factor to speed up COLMAP.
    - colmap_cmd: Name or path to the COLMAP executable.
    - ns_process: If True, prefer Nerfstudio's `ns-process-data` wrapper to produce
      transforms.json directly from images with COLMAP under the hood.
    - force: Overwrite existing outputs (`transforms.json`, copied images) if True.

    Returns
    - Path to the written `transforms.json` under `dataset_dir`.

    """
    from datetime import datetime
    import os
    import subprocess

    images_dir_p = Path(image_sets)
    if not images_dir_p.exists() or not images_dir_p.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir_p}")

    dataset_dir_p = Path(dataset_dir)
    logs_dir = dataset_dir_p / "logs"
    dataset_dir_p.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    transforms_path = dataset_dir_p / "transforms.json"
    if transforms_path.exists() and not force:
        return transforms_path

    if not ns_process:
        raise NotImplementedError("Direct COLMAP orchestration not implemented; set ns_process=True.")

    cmd: List[str] = [
        "ns-process-data",
        "images",
        "--data",
        str(images_dir_p),
        "--output-dir",
        str(dataset_dir_p),
        "--matching-method",
        str(matcher),
        "--sfm-tool",
        "colmap",
        "--no-gpu",
    ]

    log_file = logs_dir / f"ns_process_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    with log_file.open("w", encoding="utf-8") as lf:
        lf.write("Command: " + " ".join(cmd) + "\n\n")
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
            env=os.environ,
        )
        if proc.stdout is not None:
            lf.write(proc.stdout)

    if proc.returncode != 0:
        raise RuntimeError(
            f"ns-process-data failed with exit code {proc.returncode}. See log: {log_file}"
        )

    if not transforms_path.exists():
        raise RuntimeError(
            f"Expected transforms.json not found at {transforms_path}. See log: {log_file}"
        )

    return transforms_path


def train_splatfacto(
    dataset_dir: PathLike,
    results_dir: PathLike,
) -> Dict[str, Path]:
    """Train Nerfstudio Splatfacto on the prepared dataset.

    - Runs: ns-train splatfacto --data <dataset_dir> --output-dir <results_dir>
    - Captures logs to <results_dir>/logs/ns_train_*.log
    - Returns discovered paths to run_dir, config, and checkpoint (best effort).
    """
    from datetime import datetime
    import os
    import subprocess

    dataset_dir_p = Path(dataset_dir)
    results_dir_p = Path(results_dir)
    if not dataset_dir_p.exists():
        raise FileNotFoundError(f"dataset_dir not found: {dataset_dir_p}")

    logs_dir = results_dir_p / "logs"
    results_dir_p.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    cmd: List[str] = [
        "ns-train",
        "splatfacto",
        "--data",
        str(dataset_dir_p),
        "--output-dir",
        str(results_dir_p),
    ]

    log_file = logs_dir / f"ns_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    with log_file.open("w", encoding="utf-8") as lf:
        lf.write("Command: " + " ".join(cmd) + "\n\n")
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
            env=os.environ,
        )
        if proc.stdout is not None:
            lf.write(proc.stdout)

    if proc.returncode != 0:
        raise RuntimeError(
            f"ns-train failed with exit code {proc.returncode}. See log: {log_file}"
        )

    # Attempt to discover the created run directory and artifacts
    def _latest_dir(base: Path) -> Optional[Path]:
        dirs = [d for d in base.iterdir() if d.is_dir()]
        return max(dirs, key=lambda p: p.stat().st_mtime) if dirs else None

    run_dir = _latest_dir(results_dir_p)
    # Some layouts nest by experiment name → timestamp; descend one level if applicable
    if run_dir and any(child.is_dir() for child in run_dir.iterdir()):
        nested = _latest_dir(run_dir)
        if nested:
            run_dir = nested

    config_path: Optional[Path] = None
    checkpoint_path: Optional[Path] = None
    if run_dir:
        for name in ("config.yml", "config.yaml"):
            cand = run_dir / name
            if cand.exists():
                config_path = cand
                break
        ckpt_dir = run_dir / "checkpoints"
        if ckpt_dir.exists():
            ckpts = sorted(ckpt_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
            if ckpts:
                checkpoint_path = ckpts[0]

    return {
        "run_dir": run_dir or results_dir_p,
        "config": config_path or (results_dir_p / "config.yml"),
        "checkpoint": checkpoint_path or (results_dir_p / "checkpoints" / "latest.ckpt"),
    }


def export_ply(
    config_or_checkpoint: PathLike,
    out_ply_path: PathLike,
    force: bool = True,
) -> Path:
    """Export Gaussian Splats using Nerfstudio's exporter to an output directory.

    - Runs: ns-export gaussian-splat --load-config <config.yml> --output-dir <out_dir>
    - Returns the output directory path.

    Parameters
    - config_or_checkpoint: Path to the run directory containing config.yml, or directly
      to a config.yml/.yaml. If a checkpoint is given, tries to resolve sibling config.
    - out_ply_path: Output directory path for exported assets.
    - force: Currently not used to delete/overwrite; directory is created if missing.
    """
    from datetime import datetime
    import os
    import subprocess

    out_dir = Path(out_ply_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    cfg_input = Path(config_or_checkpoint)
    config_path: Optional[Path] = None
    if cfg_input.is_dir():
        for name in ("config.yml", "config.yaml"):
            cand = cfg_input / name
            if cand.exists():
                config_path = cand
                break
    elif cfg_input.is_file():
        if cfg_input.suffix.lower() in {".yml", ".yaml"}:
            config_path = cfg_input
        else:
            parent = cfg_input.parent
            for name in ("config.yml", "config.yaml"):
                cand = parent / name
                if cand.exists():
                    config_path = cand
                    break

    if not config_path or not config_path.exists():
        raise FileNotFoundError(
            f"Could not resolve a Nerfstudio config file from: {cfg_input}"
        )

    cmd: List[str] = [
        "ns-export",
        "gaussian-splat",
        "--load-config",
        str(config_path),
        "--output-dir",
        str(out_dir),
    ]

    log_file = logs_dir / f"ns_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    with log_file.open("w", encoding="utf-8") as lf:
        lf.write("Command: " + " ".join(cmd) + "\n\n")
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
            env=os.environ,
        )
        if proc.stdout is not None:
            lf.write(proc.stdout)

    if proc.returncode != 0:
        raise RuntimeError(
            f"ns-export failed with exit code {proc.returncode}. See log: {log_file}"
        )

    return out_dir
