#!/usr/bin/env python3
"""
Script to profile VRAM usage of models at various resolutions.
This script loads each model and runs a dummy inference at different resolutions
to measure the peak VRAM usage.
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import models to ensure they are registered
from agentic_image2dataset.models.image_edit import QwenImageEditModel
from agentic_image2dataset.models.super_resolution import AdcSRModel, DiffBIRModel
from agentic_image2dataset.models.view_generation import StableVirtualCameraModel


def get_peak_memory_mb():
    """Get peak memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def reset_memory():
    """Reset memory stats and empty cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()


def create_dummy_image(width, height, output_path):
    """Create a dummy image for testing."""
    image = Image.fromarray(
        np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    )
    image.save(output_path)
    return output_path


def profile_model(model_name, model_class, resolutions, output_dir):
    """Profile a single model at multiple resolutions."""
    print(f"\nProfiling {model_name}...")
    results = {}

    try:
        # Initialize model
        reset_memory()
        print("  Initializing model...")
        model = model_class(device="cuda" if torch.cuda.is_available() else "cpu")
        init_mem = get_peak_memory_mb()
        print(f"  Initialization VRAM: {init_mem:.2f} MB")

        for width, height in resolutions:
            res_key = f"{width}x{height}"
            print(f"  Testing resolution {res_key}...")

            try:
                reset_memory()

                # Create dummy input
                input_path = output_dir / f"dummy_{width}x{height}.png"
                create_dummy_image(width, height, input_path)

                # Run inference
                # Note: We use a try-except block to catch OOM specifically
                start_time = time.time()

                # Specific handling for different model signatures if needed
                # For now assuming standard process(image_path, output_dir)
                model.process(input_path, output_dir)

                end_time = time.time()
                peak_mem = get_peak_memory_mb()

                print(f"    Peak VRAM: {peak_mem:.2f} MB")
                print(f"    Time: {end_time - start_time:.2f} s")

                results[res_key] = {
                    "peak_vram_mb": peak_mem,
                    "time_seconds": end_time - start_time,
                    "status": "success",
                }

            except torch.cuda.OutOfMemoryError:
                print("    OOM Error!")
                results[res_key] = {
                    "status": "oom",
                    "peak_vram_mb": get_peak_memory_mb(),  # Record what we hit before crash
                }
            except Exception as e:
                print(f"    Error: {e}")
                results[res_key] = {"status": "error", "error": str(e)}
            finally:
                # Cleanup
                if input_path.exists():
                    input_path.unlink()

    except Exception as e:
        print(f"  Failed to initialize model: {e}")
        return None

    return results


def main():
    parser = argparse.ArgumentParser(description="Profile VRAM usage of models.")
    parser.add_argument(
        "--output", type=str, default="vram_profile.json", help="Output JSON file"
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available. Cannot profile VRAM.")
        return

    output_dir = Path("temp_profile_output")
    output_dir.mkdir(exist_ok=True)

    # Models to profile with their specific resolutions
    # Format: (Name, Class, List of (Width, Height))
    models_to_profile = [
        (
            "diffbir",
            DiffBIRModel,
            [(128, 128), (256, 256), (512, 512)],
        ),  # SR model, outputs 4x input
        (
            "adcsr",
            AdcSRModel,
            [(128, 128), (256, 256), (512, 512)],
        ),  # SR model, outputs 4x input
        (
            "stable_virtual_camera",
            StableVirtualCameraModel,
            [(576, 576)],
        ),  # Fixed resolution model
        ("qwen_image_edit", QwenImageEditModel, [(512, 512), (768, 768), (1024, 1024)]),
    ]

    all_results = {}

    for name, cls, resolutions in models_to_profile:
        model_results = profile_model(name, cls, resolutions, output_dir)
        if model_results:
            all_results[name] = model_results

            # Calculate linear regression if we have enough data points
            valid_points = []
            for res, data in model_results.items():
                if data["status"] == "success":
                    w, h = map(int, res.split("x"))
                    pixels = w * h
                    mem = data["peak_vram_mb"]
                    valid_points.append((pixels, mem))

            if len(valid_points) >= 2:
                pixels = np.array([p[0] for p in valid_points])
                mems = np.array([p[1] for p in valid_points])

                # Simple linear regression: VRAM = Base + Coeff * Pixels
                # We can use np.polyfit(pixels, mems, 1)
                slope, intercept = np.polyfit(pixels, mems, 1)

                all_results[name]["regression"] = {
                    "slope_mb_per_pixel": float(slope),
                    "intercept_mb": float(intercept),
                    "formula": f"VRAM_MB = {intercept:.2f} + {slope:.2e} * Pixels",
                }
                print(f"  Regression: {all_results[name]['regression']['formula']}")

    # Save results
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {args.output}")

    # Cleanup
    import shutil

    shutil.rmtree(output_dir)


if __name__ == "__main__":
    main()
