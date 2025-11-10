"""
Stable Virtual Camera model wrapper for generating novel views.
"""

import glob
import os.path as osp
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from seva.eval import create_transforms_simple, infer_prior_stats, run_one_scene
from seva.geometry import (
    get_default_intrinsics,
    get_preset_pose_fov,
)
from seva.model import SGMWrapper
from seva.modules.autoencoder import AutoEncoder
from seva.modules.conditioner import CLIPConditioner
from seva.sampling import DiscreteDenoiser
from seva.utils import load_model

from .base import BaseProcessingModel


class StableVirtualCameraModel(BaseProcessingModel):
    """Stable Virtual Camera model for generating novel views."""

    def __init__(self, device: str = "cuda", model_version: float = 1.1, **kwargs):
        super().__init__(device, **kwargs)
        self.model_version = model_version
        self._model = None  # Lazy initialization
        self._ae = None
        self._conditioner = None
        self._denoiser = None

        self._run_one_scene = run_one_scene
        self._get_preset_pose_fov = get_preset_pose_fov
        self._get_default_intrinsics = get_default_intrinsics
        self._infer_prior_stats = infer_prior_stats

    @property
    def model(self):
        """Lazy load the model when first accessed."""
        if self._model is None:
            base_model = load_model(
                model_version=self.model_version, device=self.device, verbose=False
            )
            self._model = SGMWrapper(base_model.eval()).to(self.device)
        return self._model

    @property
    def ae(self):
        """Lazy load the autoencoder when first accessed."""
        if self._ae is None:
            self._ae = AutoEncoder(chunk_size=1).to(self.device)
        return self._ae

    @property
    def conditioner(self):
        """Lazy load the conditioner when first accessed."""
        if self._conditioner is None:
            self._conditioner = CLIPConditioner().to(self.device)
        return self._conditioner

    @property
    def denoiser(self):
        """Lazy load the denoiser when first accessed."""
        if self._denoiser is None:
            self._denoiser = DiscreteDenoiser(num_idx=1000, device=self.device)
        return self._denoiser

    def process(
        self,
        image_path: str | Path,
        output_dir: str | Path,
        num_views: int = 24,
        trajectory: str = "orbit",
        # Advanced parameters with sensible defaults
        T: int = 21,
        H: int = 576,
        W: int = 576,
        guider_types: list[int] | None = None,
        cfg: list[float] | None = None,
        chunk_strategy: str = "interp",
        camera_scale: float = 2.0,
        num_steps: int = 50,
        cfg_min: float = 1.2,
        video_save_fps: float = 30.0,
        use_traj_prior: bool = True,
        replace_or_include_input: bool = True,
        seed: int = 23,
    ) -> list[Path]:
        """
        Generate novel views from the input image using img2trajvid_s-prob pipeline.

        Args:
            image_path: Path to the input image
            output_dir: Directory to save generated views
            num_views: Number of views to generate (default: 24)
            trajectory: Camera trajectory type - "orbit", "pan-left", "pan-right", etc. (default: "orbit")
            T: Temporal dimension for the model (default: 21)
            H: Image height (default: 576)
            W: Image width (default: 576)
            guider_types: List of guider types [1, 2] for two-pass generation (default: [1, 2])
            cfg: Classifier-free guidance scale, can be single value or list for multi-pass (default: [4.0, 2.0])
            chunk_strategy: Chunking strategy for processing (default: "interp")
            camera_scale: Scale factor for camera motion (default: 2.0)
            num_steps: Number of denoising steps (default: 50)
            cfg_min: Minimum CFG value (default: 1.2)
            video_save_fps: FPS for saved video (default: 30.0)
            use_traj_prior: Whether to use trajectory prior (default: True)
            replace_or_include_input: Whether to include input image in output (default: True)
            seed: Random seed (default: 23)
        """
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get image dimensions
        with Image.open(image_path) as img:
            img_W, img_H = img.size
            aspect_ratio = img_W / img_H

        # Set defaults for optional list parameters
        if guider_types is None:
            guider_types = [1, 2]
        if cfg is None:
            cfg = [4.0, 2.0]

        # Build version_dict with explicit parameters
        version_dict = {
            "H": H,
            "W": W,
            "T": T,
            "C": 4,
            "f": 8,
            "options": {
                "chunk_strategy": chunk_strategy,
                "video_save_fps": video_save_fps,
                "beta_linear_start": 5e-6,
                "log_snr_shift": 2.4,
                "guider_types": guider_types,
                "cfg": cfg,
                "camera_scale": camera_scale,
                "num_steps": num_steps,
                "cfg_min": cfg_min,
                "encoding_t": 1,
                "decoding_t": 1,
                "num_targets": num_views,
                "traj_prior": trajectory,
                "use_traj_prior": use_traj_prior,
                "replace_or_include_input": replace_or_include_input,
            },
        }

        num_inputs = 1  # img2trajvid_s-prob only supports single-view
        num_targets = version_dict["options"]["num_targets"]

        # Infer anchor stats (modifies T in-place)
        num_anchors = self._infer_prior_stats(
            version_dict["T"],
            num_inputs,
            num_total_frames=num_targets,
            version_dict=version_dict,
        )

        # Set up input and anchor indices
        input_indices = [0]
        anchor_indices = np.linspace(1, num_targets, num_anchors).tolist()

        # Generate camera poses and intrinsics
        c2ws, fovs = self._get_preset_pose_fov(
            option=version_dict["options"]["traj_prior"],
            num_frames=num_targets + 1,
            start_w2c=torch.eye(4),
            look_at=torch.Tensor([0, 0, 10]),
        )

        Ks = self._get_default_intrinsics(
            fovs, aspect_ratio=aspect_ratio
        )  # unnormalized
        Ks[:, :2] *= (
            torch.tensor([img_W, img_H]).reshape(1, -1, 1).repeat(Ks.shape[0], 1, 1)
        )  # normalized
        Ks = Ks.numpy()

        # Get anchor poses and intrinsics
        anchor_c2ws = c2ws[[round(ind) for ind in anchor_indices]]
        anchor_Ks = Ks[[round(ind) for ind in anchor_indices]]

        # Prepare image conditioning
        all_imgs_path = [str(image_path)] + [None] * num_targets
        image_cond = {
            "img": all_imgs_path,
            "input_indices": input_indices,
            "prior_indices": anchor_indices,
        }

        # Prepare camera conditioning (store c2ws for later use in postprocessing)
        c2ws_tensor = torch.tensor(c2ws[:, :3]).float()
        Ks_tensor = torch.tensor(Ks).float()
        camera_cond = {
            "c2w": c2ws_tensor,
            "K": Ks_tensor,
            "input_indices": list(range(num_inputs + num_targets)),
        }

        # Run the img2trajvid_s-prob pipeline
        video_path_generator = self._run_one_scene(
            task="img2trajvid_s-prob",
            version_dict=version_dict,
            model=self.model,
            ae=self.ae,
            conditioner=self.conditioner,
            denoiser=self.denoiser,
            image_cond=image_cond,
            camera_cond=camera_cond,
            save_path=str(output_dir),
            use_traj_prior=version_dict["options"]["use_traj_prior"],
            traj_prior_Ks=torch.tensor(anchor_Ks).float()
            if anchor_Ks is not None
            else None,
            traj_prior_c2ws=torch.tensor(anchor_c2ws[:, :3]).float()
            if anchor_c2ws is not None
            else None,
            seed=seed,
        )

        # Consume the generator (it yields video paths during generation)
        for _ in video_path_generator:
            pass

        # Convert from OpenCV to OpenGL camera format (same as demo.py)
        c2ws = c2ws_tensor @ torch.tensor(np.diag([1, -1, -1, 1])).float()

        # Collect image paths (same logic as demo.py)
        img_paths = sorted(glob.glob(osp.join(str(output_dir), "samples-rgb", "*.png")))
        if len(img_paths) != len(c2ws):
            input_img_paths = sorted(
                glob.glob(osp.join(str(output_dir), "input", "*.png"))
            )
            assert len(img_paths) == num_targets
            assert len(input_img_paths) == num_inputs
            assert c2ws.shape[0] == num_inputs + num_targets
            target_indices = [i for i in range(c2ws.shape[0]) if i not in input_indices]
            img_paths = [
                input_img_paths[input_indices.index(i)]
                if i in input_indices
                else img_paths[target_indices.index(i)]
                for i in range(c2ws.shape[0])
            ]

        # Create transforms.json (same as demo.py)
        create_transforms_simple(
            save_path=str(output_dir),
            img_paths=img_paths,
            img_whs=np.array([version_dict["W"], version_dict["H"]])[None].repeat(
                num_inputs + num_targets, 0
            ),
            c2ws=c2ws,
            Ks=Ks_tensor,
        )

        # Copy all images to the root of output_dir for the pipeline to find them easily
        # (pipeline._create_final_dataset uses glob("*") which only looks at root level)
        final_paths = []
        for i, img_path in enumerate(img_paths):
            final_path = output_dir / f"view_{i:04d}.png"
            shutil.copy2(img_path, final_path)
            final_paths.append(final_path)

        return final_paths if final_paths else []

    def get_description(self) -> str:
        """Get model description."""
        return "Stable Virtual Camera model for generating novel views from a single input image"
