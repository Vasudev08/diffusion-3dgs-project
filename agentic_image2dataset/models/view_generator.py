"""
Stable Virtual Camera model wrapper for generating novel views.
"""

from pathlib import Path

import numpy as np
import torch
from seva.eval import run_one_scene
from seva.geometry import (
    get_default_intrinsics,
    get_preset_pose_fov,
)
from seva.utils import load_model

from .base import BaseProcessingModel, load_image, save_image


class StableVirtualCameraModel(BaseProcessingModel):
    """Stable Virtual Camera model for generating novel views."""

    def __init__(self, device: str = "cuda", model_version: float = 1.1, **kwargs):
        super().__init__(device, **kwargs)
        self.model_version = model_version
        self.model = load_model(
            model_version=self.model_version, device=self.device, verbose=False
        )

        self._run_one_scene = run_one_scene
        self._get_preset_pose_fov = get_preset_pose_fov
        self._get_default_intrinsics = get_default_intrinsics

    def analyze(
        self, image_path: str | Path
    ) -> dict[str, int | float | bool | str | list[str]]:
        """Analyze image for view generation suitability."""
        image_path = Path(image_path)
        image = load_image(image_path)

        height: int = image.shape[0]
        width: int = image.shape[1]
        aspect_ratio = width / height

        analysis: dict[str, int | float | bool | str | list[str]] = {
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
            "suitable_for_view_generation": True,
            "recommended_views": self._get_recommended_view_count(width, height),
            "camera_trajectory": "orbit",  # Default trajectory
        }

        # Check if image is suitable for view generation
        if width < 256 or height < 256:
            analysis["suitable_for_view_generation"] = False
            analysis["issues"] = ["Image resolution too low for view generation"]
        elif aspect_ratio < 0.5 or aspect_ratio > 2.0:
            analysis["issues"] = [
                "Unusual aspect ratio may affect view generation quality"
            ]

        return analysis

    def process(
        self,
        image_path: str | Path,
        output_dir: str | Path,
        num_views: int = 24,
        trajectory: str = "orbit",
        **kwargs: object,
    ) -> list[Path]:
        """Generate novel views from the input image."""
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load and prepare the input image
        image = load_image(image_path)
        height: int = image.shape[0]
        width: int = image.shape[1]
        aspect_ratio = width / height

        # Generate camera poses and intrinsics
        c2ws, fovs = self._get_preset_pose_fov(
            option=trajectory,
            num_frames=num_views,
            start_w2c=torch.eye(4),
            look_at=torch.Tensor([0, 0, 10]),
        )

        Ks = self._get_default_intrinsics(fovs, aspect_ratio=aspect_ratio)

        # Prepare input data
        input_images = [image]
        input_c2ws = [torch.eye(4)]  # Identity pose for input image
        input_Ks = [Ks[0]]  # Use first camera intrinsics

        # Generate views
        try:
            generated_images = self._run_one_scene(
                model=self.model,
                input_images=input_images,
                input_c2ws=input_c2ws,
                input_Ks=input_Ks,
                target_c2ws=c2ws,
                target_Ks=Ks,
                device=self.device,
                **kwargs,
            )

            # Save generated images
            output_paths = []
            for i, gen_image in enumerate(generated_images):
                if isinstance(gen_image, torch.Tensor):
                    gen_image = gen_image.detach().cpu().numpy()

                # Convert to PIL Image format if needed
                if gen_image.dtype != np.uint8:
                    gen_image = (gen_image * 255).astype(np.uint8)

                output_path = output_dir / f"view_{i:03d}.png"
                save_image(gen_image, output_path)
                output_paths.append(output_path)

            return output_paths

        except Exception as e:
            raise RuntimeError(f"Failed to generate views: {e}")

    def get_description(self) -> str:
        """Get model description."""
        return "Stable Virtual Camera model for generating novel views from a single input image"

    def get_requirements(self) -> dict[str, object]:
        """Get model requirements."""
        return {
            "dependencies": ["torch", "numpy", "PIL"],
            "model_size": "~5GB",
            "device": "CUDA recommended",
            "stable_virtual_camera": "Must be installed and accessible",
        }

    def _get_recommended_view_count(self, width: int, height: int) -> int:
        """Get recommended number of views based on image size."""
        total_pixels = width * height

        if total_pixels < 256 * 256:
            return 12
        elif total_pixels < 512 * 512:
            return 24
        else:
            return 36
