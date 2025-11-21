"""
AdcSR model wrapper for super-resolution.
"""

import copy
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from ..base import BaseProcessingModel, load_image

# Add AdcSR to path
ADCSR_PATH = Path(__file__).parent.parent.parent.parent / "AdcSR"
if str(ADCSR_PATH) not in sys.path:
    sys.path.insert(0, str(ADCSR_PATH))


class AdcSRModel(BaseProcessingModel):
    """AdcSR (Adversarial Diffusion Compression for Real-World Image Super-Resolution) model."""

    def __init__(
        self,
        device: str = "cuda",
        scale: int = 4,
        epoch: int = 200,
        model_dir: str | Path | None = None,
        **kwargs,
    ):
        super().__init__(device, **kwargs)
        self.scale: int = scale
        self.device: str = device
        self.epoch: int = epoch
        self.model_dir: Path = Path(model_dir) if model_dir else ADCSR_PATH / "weight"
        self._model: torch.nn.Module | None = None  # Lazy initialization

    @property
    def model(self) -> torch.nn.Module:
        """Lazy load the AdcSR model when first accessed."""
        if self._model is None:
            self._setup_adcsr()
        if self._model is None:
            raise RuntimeError("Failed to initialize AdcSR model")
        return self._model

    def _setup_adcsr(self) -> None:
        """Setup AdcSR model."""
        try:
            from diffusers import StableDiffusionPipeline
            from diffusers.models.autoencoders.vae import Decoder
            from model import Net
        except ImportError as e:
            raise ImportError(
                f"Failed to import AdcSR dependencies. Make sure AdcSR is properly set up. Error: {e}"
            )

        device = torch.device(self.device)

        # Load Stable Diffusion pipeline
        model_id = "stabilityai/stable-diffusion-2-1-base"
        pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)

        unet = pipe.unet

        # Load half decoder
        halfdecoder_path = self.model_dir / "pretrained" / "halfDecoder.ckpt"
        if not halfdecoder_path.exists():
            raise FileNotFoundError(
                f"AdcSR half decoder not found at {halfdecoder_path}. "
                "Please download it from the AdcSR repository."
            )

        ckpt_halfdecoder = torch.load(
            str(halfdecoder_path), weights_only=False, map_location=device
        )  # type: ignore

        decoder = Decoder(  # type: ignore
            in_channels=4,
            out_channels=3,
            up_block_types=tuple(["UpDecoderBlock2D" for _ in range(4)]),
            block_out_channels=tuple([64, 128, 256, 256]),
            layers_per_block=2,
            norm_num_groups=32,
            act_fn="silu",
            norm_type="group",
            mid_block_add_attention=True,
        ).to(device)

        decoder_ckpt = {}
        for k, v in ckpt_halfdecoder["state_dict"].items():
            if "decoder" in k:
                new_k = k.replace("decoder.", "")
                decoder_ckpt[new_k] = v

        decoder.load_state_dict(decoder_ckpt, strict=True)  # type: ignore

        # Load main model
        model_path = self.model_dir / f"net_params_{self.epoch}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(
                f"AdcSR model weights not found at {model_path}. "
                f"Available epochs: {[f.stem.split('_')[-1] for f in self.model_dir.glob('net_params_*.pkl')]}"
            )

        model = torch.nn.DataParallel(Net(unet, copy.deepcopy(decoder)))

        model.load_state_dict(
            torch.load(str(model_path), weights_only=False, map_location=device)
        )

        self._model = torch.nn.Sequential(
            model.module,
            *decoder.up_blocks,
            decoder.conv_norm_out,
            decoder.conv_act,
            decoder.conv_out,
        ).to(device)
        self._model.eval()

    def process(
        self,
        image_path: str | Path,
        output_dir: str | Path,
        scale: int | None = None,
        **kwargs,
    ) -> list[Path]:
        """Apply super-resolution to the image(s) using AdcSR.

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

        # Check if input is a directory or a single file
        if image_path.is_dir():
            # Process all images in the directory
            image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
            image_files = [
                f
                for f in image_path.iterdir()
                if f.is_file() and f.suffix.lower() in image_extensions
            ]

            if not image_files:
                raise ValueError(f"No image files found in directory: {image_path}")

            # Process each image
            all_results = []
            for img_file in image_files:
                result = self._process_single_image(img_file, output_dir, scale)
                all_results.append(result)

            return all_results
        else:
            # Process single image
            return [self._process_single_image(image_path, output_dir, scale)]

    def _process_single_image(
        self, image_path: Path, output_dir: Path, scale: int
    ) -> Path:
        """Process a single image with AdcSR.

        Args:
            image_path: Path to the input image
            output_dir: Directory to save the processed image
            scale: Scale factor for super-resolution

        Returns:
            Path to the generated super-resolution image
        """
        # Note: AdcSR is trained for 4x upscaling, so we'll use that
        # If a different scale is requested, we'll need to handle it differently
        if scale != 4:
            # For now, we'll use 4x and then resize if needed
            # This is a limitation of AdcSR - it's trained for 4x specifically
            pass

        # Load the image
        image = load_image(image_path)

        # Convert numpy array to PIL Image
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        pil_image = Image.fromarray(image).convert("RGB")
        original_w, original_h = pil_image.size

        # Convert to tensor and normalize to [-1, 1]
        transform = transforms.ToTensor()
        lr_tensor = transform(pil_image).to(self.device).unsqueeze(0) * 2 - 1

        # Calculate padding to multiple of 16
        pad_w = (16 - original_w % 16) % 16
        pad_h = (16 - original_h % 16) % 16

        # Apply padding if needed
        if pad_w > 0 or pad_h > 0:
            lr_tensor_padded = torch.nn.functional.pad(
                lr_tensor, (0, pad_w, 0, pad_h), mode="reflect"
            )
        else:
            lr_tensor_padded = lr_tensor

        # Run inference
        with torch.no_grad():
            sr_tensor = self.model(lr_tensor_padded)

            # Crop back to original size
            if pad_w > 0 or pad_h > 0:
                sr_tensor = sr_tensor[:, :, :original_h, :original_w]

            # Apply statistics matching (from AdcSR test.py)
            # Note: We use the original (unpadded) lr_tensor for statistics matching
            sr_tensor = (
                sr_tensor - sr_tensor.mean(dim=[2, 3], keepdim=True)
            ) / sr_tensor.std(dim=[2, 3], keepdim=True) * lr_tensor.std(
                dim=[2, 3], keepdim=True
            ) + lr_tensor.mean(dim=[2, 3], keepdim=True)

            # Convert back to PIL Image
            sr_tensor = (sr_tensor[0] / 2 + 0.5).clamp(0, 1).cpu()
            sr_pil = transforms.ToPILImage()(sr_tensor)

        # If a different scale was requested, resize
        if scale != 4:
            target_size = (
                pil_image.width * scale,
                pil_image.height * scale,
            )
            sr_pil = sr_pil.resize(target_size, Image.Resampling.LANCZOS)

        # Save result
        final_output_path = output_dir / f"sr_adcsr_{image_path.stem}.png"
        sr_pil.save(final_output_path)

        return final_output_path

    def get_description(self) -> str:
        """Get model description."""
        return (
            f"AdcSR (Adversarial Diffusion Compression) super-resolution model for upscaling images. "
            f"Epoch: {self.epoch}, default scale factor: {self.scale}x. "
            f"Optimized for 4x upscaling. Can process single images or directories of images. "
            f"Key parameters: scale (can override default, but model is trained for 4x)."
        )
