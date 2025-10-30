"""
DiffBIR model wrapper for super-resolution.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image

from .base import BaseProcessingModel, load_image

# Add DiffBIR to path
DIFFBIR_PATH = Path(__file__).parent.parent.parent / "DiffBIR"
if str(DIFFBIR_PATH) not in sys.path:
    sys.path.insert(0, str(DIFFBIR_PATH))

from diffbir.inference import BSRInferenceLoop


class DiffBIRModel(BaseProcessingModel):
    """DiffBIR model for super-resolution."""

    def __init__(
        self, device: str = "cuda", scale: int = 4, version: str = "v2.1", **kwargs
    ):
        super().__init__(device, **kwargs)
        self.scale: int = scale
        self.device: str = device
        self.version: str = version
        self.inference_loop: Optional[BSRInferenceLoop] = None

        # Initialize DiffBIR inference loop
        self._setup_diffbir()

    def _setup_diffbir(self) -> None:
        """Setup DiffBIR inference loop."""

        # Create a mock args object for DiffBIR
        class MockArgs:
            def __init__(self, device: str, scale: int, version: str) -> None:
                self.device: str = device
                self.upscale: int = scale
                self.version: str = version
                self.task: str = "sr"
                self.sampler: str = "edm_dpm++_3m_sde"
                self.steps: int = 10
                self.captioner: str = "none"
                self.pos_prompt: str = ""
                self.neg_prompt: str = "low quality, blurry, low-resolution, noisy, unsharp, weird textures"
                self.cfg_scale: float = 4.0
                self.cleaner_tiled: bool = False
                self.cleaner_tile_size: int = 512
                self.cleaner_tile_stride: int = 256
                self.vae_encoder_tiled: bool = False
                self.vae_encoder_tile_size: int = 256
                self.vae_decoder_tiled: bool = False
                self.vae_decoder_tile_size: int = 256
                self.cldm_tiled: bool = False
                self.cldm_tile_size: int = 512
                self.cldm_tile_stride: int = 256
                self.start_point_type: str = "noise"
                self.guidance: bool = False
                self.g_loss: str = "w_mse"
                self.g_scale: float = 0.0
                self.batch_size: int = 1
                self.n_samples: int = 1
                self.seed: int = 231
                self.precision: str = "fp16"
                self.llava_bit: str = "4"
                self.rescale_cfg: bool = False
                self.noise_aug: int = 0
                self.s_churn: float = 0
                self.s_tmin: float = 0
                self.s_tmax: float = 300
                self.s_noise: float = 1
                self.eta: float = 1
                self.order: int = 1
                self.strength: float = 1

        args = MockArgs(self.device, self.scale, self.version)
        self.inference_loop = BSRInferenceLoop(args)
        self.inference_loop.load_cleaner()
        self.inference_loop.load_pipeline()

    def analyze(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """Analyze image for super-resolution suitability."""
        image_path = Path(image_path)
        image = load_image(image_path)

        height: int = image.shape[0]
        width: int = image.shape[1]
        total_pixels: int = width * height

        analysis = {
            "width": width,
            "height": height,
            "total_pixels": total_pixels,
            "current_resolution": f"{width}x{height}",
            "upscaled_resolution": f"{width * self.scale}x{height * self.scale}",
            "would_benefit_from_sr": total_pixels < 1024 * 1024,  # Less than 1MP
            "recommended_scale": self.scale,
            "model_version": self.version,
        }

        # Determine if super-resolution would be beneficial
        if total_pixels < 256 * 256:
            analysis["sr_priority"] = "high"
            analysis["sr_benefit"] = "significant"
        elif total_pixels < 512 * 512:
            analysis["sr_priority"] = "medium"
            analysis["sr_benefit"] = "moderate"
        else:
            analysis["sr_priority"] = "low"
            analysis["sr_benefit"] = "minimal"

        return analysis

    def process(
        self,
        image_path: Union[str, Path],
        output_dir: Union[str, Path],
        scale: Optional[int] = None,
        **kwargs,
    ) -> List[Path]:
        """Apply super-resolution to the image using DiffBIR."""
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        scale = scale or self.scale

        # Load the image
        image = load_image(image_path)

        # Convert numpy array to PIL Image
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        pil_image = Image.fromarray(image)

        # Use DiffBIR for super-resolution
        # Create temporary input and output directories for DiffBIR
        temp_input_dir = output_dir / "temp_input"
        temp_output_dir = output_dir / "temp_output"
        temp_input_dir.mkdir(exist_ok=True)
        temp_output_dir.mkdir(exist_ok=True)

        # Save input image
        temp_input_path = temp_input_dir / image_path.name
        pil_image.save(temp_input_path)

        # Check if DiffBIR is available
        if self.inference_loop is None:
            raise RuntimeError("DiffBIR inference loop not initialized")

        # Update inference loop args for this specific scale
        self.inference_loop.args.upscale = scale

        # Run DiffBIR inference
        self.inference_loop.args.input = str(temp_input_dir)
        self.inference_loop.args.output = str(temp_output_dir)

        # Process the image
        self.inference_loop.run()

        # Find the output file
        output_files = list(temp_output_dir.glob("*"))
        if not output_files:
            raise RuntimeError("DiffBIR did not produce any output files")

        # Move the result to the final location
        final_output_path = output_dir / f"sr_{image_path.stem}.png"
        _ = output_files[0].rename(final_output_path)

        # Clean up temporary directories
        import shutil

        shutil.rmtree(temp_input_dir)
        shutil.rmtree(temp_output_dir)

        return [final_output_path]

    def _simple_upscale(
        self, image_path: Path, output_dir: Path, scale: int
    ) -> List[Path]:
        """Simple upscaling fallback using PIL."""
        image = load_image(image_path)

        pil_image = Image.fromarray(image)

        # Simple bicubic upscaling
        new_size = (pil_image.width * scale, pil_image.height * scale)
        upscaled = pil_image.resize(new_size, Image.Resampling.LANCZOS)

        # Save result
        output_path = output_dir / f"sr_{image_path.stem}.png"
        upscaled.save(output_path)

        return [output_path]

    def get_description(self) -> str:
        """Get model description."""
        return f"DiffBIR super-resolution model (version: {self.version}, scale factor: {self.scale}x)"
