"""
AdcSR model wrapper for super-resolution.
"""

import copy
import sys
from pathlib import Path
from typing import List, Optional, Union

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
        model_dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        super().__init__(device, **kwargs)
        self.scale: int = scale
        self.device: str = device
        self.epoch: int = epoch
        self.model_dir: Path = Path(model_dir) if model_dir else ADCSR_PATH / "weight"
        self._model: Optional[torch.nn.Module] = None  # Lazy initialization

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
        image_path: Union[str, Path],
        output_dir: Union[str, Path],
        scale: Optional[int] = None,
        **kwargs,
    ) -> List[Path]:
        """Apply super-resolution to the image using AdcSR."""
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        scale = scale or self.scale

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

        # Convert to tensor and normalize to [-1, 1]
        transform = transforms.ToTensor()
        lr_tensor = transform(pil_image).to(self.device).unsqueeze(0) * 2 - 1

        # Run inference
        with torch.no_grad():
            sr_tensor = self.model(lr_tensor)

            # Apply statistics matching (from AdcSR test.py)
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

        return [final_output_path]

    def get_description(self) -> str:
        """Get model description."""
        return f"AdcSR super-resolution model (epoch: {self.epoch}, scale factor: {self.scale}x)"
