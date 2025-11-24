"""
Utility functions for downloading model weights.
"""

import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def download_weights(weights_dir: Path | str = "weights") -> None:
    """
    Downloads required model weights from Hugging Face if they don't exist.

    Args:
        weights_dir: Directory to store the weights.
    """
    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)

    print(f"Checking/Downloading weights to {weights_dir.absolute()}...")

    # 1. AdcSR Weights
    # Repo: Guaishou74851/AdcSR
    # Directory in repo: weight/
    # Target: weights/
    adcsr_repo_id = "Guaishou74851/AdcSR"
    api = HfApi()

    try:
        # List all files in the repo
        files = api.list_repo_files(repo_id=adcsr_repo_id)

        # Filter for files in 'weight/' directory
        weight_files = [f for f in files if f.startswith("weight/") and f != "weight/"]

        for file_path in weight_files:
            # Determine local path
            # file_path is like "weight/pretrained/halfDecoder.ckpt"
            # relative_path should be "pretrained/halfDecoder.ckpt"
            relative_path = Path(file_path).relative_to("weight")
            local_path = weights_dir / relative_path

            if not local_path.exists():
                print(f"Downloading {file_path}...")
                # Download to cache
                cached_path = hf_hub_download(repo_id=adcsr_repo_id, filename=file_path)

                # Ensure parent directory exists
                local_path.parent.mkdir(parents=True, exist_ok=True)

                # Copy to target location
                shutil.copy2(cached_path, local_path)
                print(f"Saved to {local_path}")
            else:
                # print(f"Skipping {file_path}, already exists.")
                pass

    except Exception as e:
        print(f"Warning: Failed to check/download AdcSR weights: {e}")

    # 2. HYPIR Weights
    # Repo: lxq007/HYPIR
    # File: HYPIR_sd2.pth
    # Target: weights/HYPIR_sd2.pth
    hypir_repo = "lxq007/HYPIR"
    hypir_file = "HYPIR_sd2.pth"
    hypir_target = weights_dir / hypir_file

    try:
        if not hypir_target.exists():
            print(f"Downloading {hypir_file}...")
            cached_path = hf_hub_download(repo_id=hypir_repo, filename=hypir_file)
            shutil.copy2(cached_path, hypir_target)
            print(f"Saved to {hypir_target}")
        else:
            pass
    except Exception as e:
        print(f"Warning: Failed to check/download HYPIR weights: {e}")

    print("Weight check complete.")
