"""
Run the multi-image → Nerfstudio → PLY pipeline.

This CLI wires together the functions in `pipeline_functions.py` to:
1) Process a directory of images with COLMAP via Nerfstudio (ns-process-data)
2) Train Splatfacto (ns-train)
3) Export Gaussian splats to an output directory (ns-export)

Usage (inside your Nerfstudio conda env):
  python single_image_pipeline/run_pipeline.py \
      --images-dir datasets_raw/object_X \
      [--object object_X] \
      [--dataset-dir datasets/object_X] \
      [--results-dir results/object_X] \
      [--exports-dir results/object_X/exports] \
      [--matcher exhaustive] \
      [--force]

Defaults: if you only pass --images-dir, the tool derives:
  object     = basename(images-dir)
  dataset-dir= datasets/<object>
  results-dir= results/<object>
  exports-dir= results/<object>/exports
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

# Robust import: works when run as script or module
try:
    from pipeline_functions import (
        multiview_colmap,
        train_splatfacto,
        export_ply,
    )
except ImportError:
    from single_image_pipeline.pipeline_functions import (
        multiview_colmap,
        train_splatfacto,
        export_ply,
    )


def _derive_defaults(images_dir: Path, object_slug: str | None) -> tuple[Path, Path, Path, str]:
    obj = object_slug or images_dir.name
    dataset_dir = Path("datasets") / obj
    results_dir = Path("results") / obj
    exports_dir = results_dir / "exports"
    return dataset_dir, results_dir, exports_dir, obj


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-image to PLY pipeline (Nerfstudio)")
    p.add_argument("--images-dir", required=True, type=Path, help="Input directory of images")
    p.add_argument("--object", dest="object_slug", type=str, default=None, help="Object slug (defaults to images-dir name)")
    p.add_argument("--dataset-dir", type=Path, default=None, help="Output dataset dir (defaults to datasets/<object>)")
    p.add_argument("--results-dir", type=Path, default=None, help="Training results dir (defaults to results/<object>)")
    p.add_argument("--exports-dir", type=Path, default=None, help="Export output dir (defaults to results/<object>/exports)")
    p.add_argument("--matcher", type=str, default="exhaustive", help="Matching method for ns-process-data (exhaustive/sequential)")
    p.add_argument("--force", action="store_true", help="Overwrite/skip guards when outputs already exist")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.images_dir.exists() or not args.images_dir.is_dir():
        raise SystemExit(f"--images-dir not found or not a directory: {args.images_dir}")

    dataset_dir, results_dir, exports_dir, obj = _derive_defaults(args.images_dir, args.object_slug)
    if args.dataset_dir:
        dataset_dir = args.dataset_dir
    if args.results_dir:
        results_dir = args.results_dir
    if args.exports_dir:
        exports_dir = args.exports_dir

    print(f"[1/3] Processing images with COLMAP via Nerfstudio → {dataset_dir}")
    transforms_path = multiview_colmap(
        image_sets=args.images_dir,
        dataset_dir=dataset_dir,
        matcher=args.matcher,
        ns_process=True,
        force=args.force,
    )
    print(f"      transforms.json: {transforms_path}")

    print(f"[2/3] Training Splatfacto → {results_dir}")
    train_out = train_splatfacto(dataset_dir=dataset_dir, results_dir=results_dir)
    run_dir = train_out.get("run_dir", results_dir)
    config = train_out.get("config")
    checkpoint = train_out.get("checkpoint")
    print(f"      run_dir: {run_dir}")
    print(f"      config: {config}")
    print(f"      checkpoint: {checkpoint}")

    print(f"[3/3] Exporting Gaussian splats → {exports_dir}")
    export_dir = export_ply(config_or_checkpoint=config, out_ply_path=exports_dir)
    print(f"      export_dir: {export_dir}")

    # Write a compact summary for reproducibility
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "summary.json"
    summary = {
        "object": obj,
        "images_dir": str(args.images_dir.resolve()),
        "dataset_dir": str(dataset_dir.resolve()),
        "results_dir": str(results_dir.resolve()),
        "exports_dir": str(Path(export_dir).resolve()),
        "transforms": str(Path(transforms_path).resolve()),
        "run_dir": str(Path(run_dir).resolve()) if run_dir else None,
        "config": str(Path(config).resolve()) if config else None,
        "checkpoint": str(Path(checkpoint).resolve()) if checkpoint else None,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Summary written to: {summary_path}")
    except Exception as e:
        print(f"Warning: failed to write summary.json: {e}")


if __name__ == "__main__":
    main()
