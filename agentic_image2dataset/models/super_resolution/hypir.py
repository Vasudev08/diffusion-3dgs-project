"""
HYPIR model wrapper for super-resolution.
"""

import sys
from pathlib import Path

from PIL import Image
from torchvision import transforms

from ..base import BaseProcessingModel


class HYPIRModel(BaseProcessingModel):
    """HYPIR (Harnessing Diffusion-Yielded Score Priors for Image Restoration) model."""

    def __init__(
        self,
        device: str = "cuda",
        base_model_type: str = "sd2",
        base_model_path: str = "Manojb/stable-diffusion-2-1-base",
        model_t: int = 200,
        coeff_t: int = 200,
        lora_rank: int = 256,
        lora_modules: list[str] | None = None,
        weight_path: str | Path | None = None,
        patch_size: int = 512,
        stride: int = 256,
        upscale: int = 4,
        **kwargs,
    ):
        super().__init__(device, **kwargs)
        self.device = device
        self.base_model_type = base_model_type
        self.base_model_path = base_model_path
        self.model_t = model_t
        self.coeff_t = coeff_t
        self.lora_rank = lora_rank
        self.lora_modules = lora_modules or [
            "to_k",
            "to_q",
            "to_v",
            "to_out.0",
            "conv",
            "conv1",
            "conv2",
            "conv_shortcut",
            "conv_out",
            "proj_in",
            "proj_out",
            "ff.net.2",
            "ff.net.0.proj",
        ]
        self.weight_path = Path("weights/HYPIR_sd2.pth")
        self.patch_size = patch_size
        self.stride = stride
        self.upscale = upscale

        self._model = None  # Lazy initialization
        self._captioner = None

    @property
    def model(self):
        """Lazy load the HYPIR model when first accessed."""
        if self._model is None:
            self._setup_hypir()
        if self._model is None:
            raise RuntimeError("Failed to initialize HYPIR model")
        return self._model

    def _setup_hypir(self) -> None:
        """Setup HYPIR model."""
        # Add HYPIR to sys.path if not present
        # Assuming HYPIR is located at project root / HYPIR
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent.parent
        hypir_path = project_root / "HYPIR"

        if str(hypir_path) not in sys.path and hypir_path.exists():
            print(f"Adding {hypir_path} to sys.path")
            sys.path.append(str(hypir_path))

        try:
            from HYPIR.enhancer.sd2 import SD2Enhancer
            from HYPIR.utils.captioner import EmptyCaptioner
        except ImportError as e:
            raise ImportError(
                f"Failed to import HYPIR dependencies. Make sure HYPIR is properly set up. Error: {e}"
            )

        if not self.weight_path.exists():
            raise FileNotFoundError(
                f"HYPIR model weights not found at {self.weight_path}. "
                "Please download it from the HYPIR repository."
            )

        print(f"Loading HYPIR model from {self.weight_path}...")
        self._model = SD2Enhancer(
            base_model_path=self.base_model_path,
            weight_path=str(self.weight_path),
            lora_modules=self.lora_modules,
            lora_rank=self.lora_rank,
            model_t=self.model_t,
            coeff_t=self.coeff_t,
            device=self.device,
        )
        self._model.init_models()
        self._captioner = EmptyCaptioner(self.device)
        print("HYPIR model loaded.")

    def process(
        self,
        image_path: str | Path,
        output_dir: str | Path,
        scale: int | None = None,
        batch_size: int = 1,
        **kwargs,
    ) -> list[Path]:
        """Apply super-resolution to the image(s) using HYPIR.

        Args:
            image_path: Path to a single image file or a directory containing multiple images
            output_dir: Directory to save processed images
            scale: Scale factor for super-resolution (default: 4)
            batch_size: Number of images to process in a single batch (default: 1 for sequential)
                       Higher values improve GPU utilization but require more VRAM
            **kwargs: Additional parameters

        Returns:
            List of paths to generated super-resolution images
        """
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # HYPIR is designed for 4x upscaling by default
        # If scale is provided and not 4, we might need to resize afterwards or warn
        target_scale = scale or self.upscale

        # Check if input is a directory or a single file
        if image_path.is_dir():
            # Process all images in the directory
            image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
            image_files = [
                f
                for f in image_path.iterdir()
                if f.is_file() and f.suffix.lower() in image_extensions
            ]

            if not image_files:
                raise ValueError(f"No image files found in directory: {image_path}")
        else:
            # Process single image - wrap in list for batched processing
            image_files = [image_path]

        # Use batched processing for all cases
        return self._process_batched(image_files, output_dir, target_scale, batch_size)

    def _process_batched(
        self,
        image_files: list[Path],
        output_dir: Path,
        scale: int,
        batch_size: int,
    ) -> list[Path]:
        """Process multiple images in batches for improved GPU utilization.

        Args:
            image_files: List of image file paths to process
            output_dir: Directory to save processed images
            scale: Scale factor for super-resolution
            batch_size: Number of images to process per batch

        Returns:
            List of paths to generated super-resolution images
        """
        import torch

        all_results = []
        to_tensor = transforms.ToTensor()

        # Process images in batches
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i : i + batch_size]

            # Load and prepare batch
            batch_images = []
            batch_sizes = []  # Store original sizes for potential resizing

            for img_file in batch_files:
                img_pil = Image.open(img_file).convert("RGB")
                batch_images.append(img_pil)
                batch_sizes.append(img_pil.size)

            # Find a common size for batching (use max dimensions to avoid quality loss)
            max_width = max(img.width for img in batch_images)
            max_height = max(img.height for img in batch_images)

            # Resize all images to common size and convert to tensors
            batch_tensors = []
            for img_pil in batch_images:
                if img_pil.size != (max_width, max_height):
                    # Resize to common dimensions
                    img_pil_resized = img_pil.resize(
                        (max_width, max_height), Image.Resampling.LANCZOS
                    )
                else:
                    img_pil_resized = img_pil
                batch_tensors.append(to_tensor(img_pil_resized))

            # Stack into batched tensor (batch_size, 3, H, W)
            lq_batch = torch.stack(batch_tensors)

            # Generate caption (empty for now as per default)
            prompt = self._captioner(batch_images[0]) if self._captioner else ""

            # Run batched inference
            result_batch = self.model.enhance(
                lq=lq_batch,
                prompt=prompt,
                scale_by="factor",
                upscale=self.upscale,
                patch_size=self.patch_size,
                stride=self.stride,
                return_type="pil",
            )

            # Save each result
            for idx, (result_pil, img_file, orig_size) in enumerate(
                zip(result_batch, batch_files, batch_sizes)
            ):
                # If requested scale is different from native 4x, resize
                if scale != self.upscale:
                    target_size = (
                        orig_size[0] * scale,
                        orig_size[1] * scale,
                    )
                    result_pil = result_pil.resize(
                        target_size, Image.Resampling.LANCZOS
                    )

                # Save result
                final_output_path = output_dir / f"sr_hypir_{img_file.stem}.png"
                result_pil.save(final_output_path)
                all_results.append(final_output_path)

        return all_results

    def get_description(self) -> str:
        """Get model description."""
        return (
            f"HYPIR (Harnessing Diffusion-Yielded Score Priors) super-resolution model. "
            f"Uses Stable Diffusion 2.1 prior for high-quality restoration. "
            f"Default scale: {self.upscale}x. "
            f"Can process single images or directories. "
            f"High VRAM usage but excellent quality."
        )
