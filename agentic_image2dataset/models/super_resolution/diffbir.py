"""
DiffBIR model wrapper for super-resolution.
"""

import sys
from pathlib import Path

from ..base import BaseProcessingModel

# Add DiffBIR to path
DIFFBIR_PATH = Path(__file__).parent.parent.parent.parent / "DiffBIR"
if str(DIFFBIR_PATH) not in sys.path:
    sys.path.insert(0, str(DIFFBIR_PATH))

# Import after path modifications
from diffbir.inference import BSRInferenceLoop  # type: ignore


class DiffBIRModel(BaseProcessingModel):
    """DiffBIR model for super-resolution."""

    def __init__(
        self, device: str = "cuda", scale: int = 4, version: str = "v2.1", **kwargs
    ):
        super().__init__(device, **kwargs)
        self.scale: int = scale
        self.device: str = device
        self.version: str = version
        self._inference_loop: BSRInferenceLoop | None = None  # Lazy initialization

    @property
    def inference_loop(self) -> BSRInferenceLoop:
        """Lazy load the DiffBIR inference loop when first accessed."""
        if self._inference_loop is None:
            self._setup_diffbir()
        return self._inference_loop

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
        self._inference_loop = BSRInferenceLoop(args)

    def process(
        self,
        image_path: str | Path,
        output_dir: str | Path,
        scale: int | None = None,
        **kwargs,
    ) -> list[Path]:
        """Apply super-resolution to the image(s) using DiffBIR.

        Args:
            image_path: Path to a single image file or a directory containing multiple images
            output_dir: Directory to save processed images
            scale: Scale factor for super-resolution (default: 4)
            **kwargs: Additional parameters

        Returns:
            List of paths to generated super-resolution images
        """
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        scale = scale or self.scale

        # Update inference loop args
        self.inference_loop.args.upscale = scale
        self.inference_loop.args.input = str(image_path)
        self.inference_loop.args.output = str(output_dir)

        # Run DiffBIR inference (handles both single file and directory)
        self.inference_loop.run()

        # Collect and rename outputs
        output_paths = []

        if image_path.is_dir():
            # DiffBIR supports these extensions
            image_extensions = {".jpg", ".jpeg", ".png"}
            input_files = [
                f
                for f in image_path.iterdir()
                if f.is_file() and f.suffix.lower() in image_extensions
            ]
        else:
            input_files = [image_path]

        for f in input_files:
            # Expected output filename from DiffBIR (it uses the same stem with .png)
            expected_output = output_dir / f"{f.stem}.png"
            final_output = output_dir / f"sr_{f.stem}.png"

            if expected_output.exists():
                # Rename to add prefix
                expected_output.rename(final_output)
                output_paths.append(final_output)
            elif final_output.exists():
                # Already renamed
                output_paths.append(final_output)
            elif not image_path.is_dir():
                # If single file input and output missing, raise error
                raise RuntimeError(
                    f"DiffBIR did not produce output file: {expected_output}"
                )

        return output_paths

    def get_description(self) -> str:
        """Get model description."""
        return (
            f"DiffBIR super-resolution model for upscaling images. "
            f"Version: {self.version}, default scale factor: {self.scale}x. "
            f"Can process single images or directories of images. "
            f"Key parameters: scale (can override default {self.scale}x upscaling)."
        )
