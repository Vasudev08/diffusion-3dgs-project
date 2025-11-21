"""
Qwen Image Edit model wrapper.
"""

from pathlib import Path
import torch
from PIL import Image

from .base import BaseProcessingModel


class QwenImageEditModel(BaseProcessingModel):
    """Qwen Image Edit model for editing images based on text prompts."""

    def __init__(self, device: str = "cuda", **kwargs):
        super().__init__(device, **kwargs)
        self._pipeline = None

    @property
    def pipeline(self):
        """Lazy load the pipeline when first accessed."""
        if self._pipeline is None:
            from diffusers import QwenImageEditPipeline

            # import sdnq to register it into diffusers and transformers
            from sdnq import SDNQConfig  # noqa: F401

            print("Loading Quantized Qwen Image Edit pipeline...")
            self._pipeline = QwenImageEditPipeline.from_pretrained(
                "Disty0/Qwen-Image-Edit-SDNQ-uint4-svd-r32",
                torch_dtype=torch.bfloat16,
            )
            self._pipeline.enable_model_cpu_offload()
            self._pipeline.set_progress_bar_config(disable=None)
        return self._pipeline

    def process(
        self,
        image_path: str | Path,
        output_dir: str | Path,
        prompt: str = "Generate a view parallel to the horizontal axis",
        negative_prompt: str = " ",
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        seed: int = 0,
        **kwargs,
    ) -> list[Path]:
        """
        Edit an image using Qwen Image Edit.

        Args:
            image_path: Path to the input image
            output_dir: Directory to save processed images
            prompt: Text prompt for editing
            negative_prompt: Negative prompt
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale (true_cfg_scale)
            seed: Random seed
            **kwargs: Additional parameters

        Returns:
            List of paths to generated images
        """
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Prepare inputs
        inputs = {
            "image": image,
            "prompt": prompt,
            "generator": torch.manual_seed(seed),
            "true_cfg_scale": guidance_scale,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
        }

        # Run inference
        with torch.inference_mode():
            # Pipeline with cpu offload handles device placement
            output = self.pipeline(**inputs)
            output_image = output.images[0]

        # Save output
        output_filename = f"{image_path.stem}_edited.png"
        output_path = output_dir / output_filename
        output_image.save(output_path)

        print(f"Saved edited image to {output_path}")

        return [output_path]

    def get_description(self) -> str:
        """Get model description."""
        return (
            "Qwen Image Edit model for editing images based on text prompts. "
            "Useful for changing view perspectives or modifying image content. "
            "Key parameters: prompt (default: 'Generate a view parallel to the horizontal axis'), "
            "num_inference_steps (default: 50), guidance_scale (default: 4.0), seed (default: 0)."
        )
