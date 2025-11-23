"""
Stable Virtual Camera model wrapper for generating novel views.
"""

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_project_root = Path(__file__).parent.parent.parent.parent
_stable_virtual_camera_path = _project_root / "stable-virtual-camera"
if str(_stable_virtual_camera_path) not in sys.path:
    sys.path.insert(0, str(_stable_virtual_camera_path))


from demo import parse_task
from seva.eval import run_one_scene
from seva.model import SGMWrapper
from seva.modules.autoencoder import AutoEncoder
from seva.modules.conditioner import CLIPConditioner
from seva.sampling import DiscreteDenoiser
from seva.utils import load_model

from ..base import BaseProcessingModel


def create_transforms_simple(save_path, img_paths, img_whs, c2ws, Ks):
    import os.path as osp

    out_frames = []
    for img_path, img_wh, c2w, K in zip(img_paths, img_whs, c2ws, Ks):
        # Ensure transform_matrix is 4x4 for nerfstudio compatibility
        transform_matrix = c2w.tolist()
        if len(transform_matrix) == 3:
            # Add bottom row [0, 0, 0, 1] to make it a proper 4x4 homogeneous matrix
            transform_matrix.append([0.0, 0.0, 0.0, 1.0])

        out_frame = {
            "fl_x": K[0][0].item(),
            "fl_y": K[1][1].item(),
            "cx": K[0][2].item(),
            "cy": K[1][2].item(),
            "w": img_wh[0].item(),
            "h": img_wh[1].item(),
            "file_path": f"./{osp.relpath(img_path, start=save_path)}"
            if img_path is not None
            else None,
            "transform_matrix": transform_matrix,
        }
        out_frames.append(out_frame)
    out = {
        # "camera_model": "PINHOLE",
        "orientation_override": "none",
        "frames": out_frames,
    }
    with open(osp.join(save_path, "transforms.json"), "w") as of:
        json.dump(out, of, indent=5)


def process_scene(
    task: str,
    scene_path: str,
    output_dir: str,
    model: SGMWrapper,
    ae: AutoEncoder,
    conditioner: CLIPConditioner,
    denoiser: DiscreteDenoiser,
    version_dict: dict,
    num_inputs: int | None = None,
    use_traj_prior: bool = False,
    seed: int = 23,
    return_paths: bool = False,
):
    """
    Core function to process a single scene for any task type.

    This function extracts the common logic from main() and can be reused
    by both the CLI demo and programmatic interfaces.

    Args:
        task: Task type ("img2trajvid_s-prob", "img2img", "img2vid", "img2trajvid")
        scene_path: Path to input image or dataset directory
        output_dir: Directory to save outputs
        model: SGMWrapper model instance
        ae: AutoEncoder instance
        conditioner: CLIPConditioner instance
        denoiser: DiscreteDenoiser instance
        version_dict: Version configuration dict (will be modified in-place)
        num_inputs: Number of input views (None for auto-detect)
        use_traj_prior: Whether to use trajectory prior
        seed: Random seed
        return_paths: If True, return list of output image paths

    Returns:
        List of Path objects if return_paths=True, else None
    """
    from pathlib import Path

    # Parse task to get all setup
    (
        all_imgs_path,
        num_inputs,
        num_targets,
        input_indices,
        anchor_indices,
        c2ws,
        Ks,
        anchor_c2ws,
        anchor_Ks,
    ) = parse_task(
        task,
        scene_path,
        num_inputs,
        version_dict["T"],
        version_dict,
    )
    assert num_inputs is not None

    # Create image conditioning
    image_cond = {
        "img": all_imgs_path,
        "input_indices": input_indices,
        "prior_indices": anchor_indices,
    }

    # Create camera conditioning
    camera_cond = {
        "c2w": c2ws.clone(),
        "K": Ks.clone(),
        "input_indices": list(range(num_inputs + num_targets)),
    }

    # Run the scene generation
    video_path_generator = run_one_scene(
        task,
        version_dict,  # H, W may be updated in-place
        model=model,
        ae=ae,
        conditioner=conditioner,
        denoiser=denoiser,
        image_cond=image_cond,
        camera_cond=camera_cond,
        save_path=output_dir,
        use_traj_prior=use_traj_prior,
        traj_prior_Ks=anchor_Ks,
        traj_prior_c2ws=anchor_c2ws,
        seed=seed,
    )

    # Consume generator
    for _ in video_path_generator:
        pass

    # Post-process: convert camera format
    c2ws = c2ws @ torch.tensor(np.diag([1, -1, -1, 1])).float()

    # Collect image paths
    img_paths = sorted(glob.glob(os.path.join(output_dir, "samples-rgb", "*.png")))
    if len(img_paths) != len(c2ws):
        input_img_paths = sorted(glob.glob(os.path.join(output_dir, "input", "*.png")))
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

    # Create transforms.json
    create_transforms_simple(
        save_path=output_dir,
        img_paths=img_paths,
        img_whs=np.array([version_dict["W"], version_dict["H"]])[None].repeat(
            num_inputs + num_targets, 0
        ),
        c2ws=c2ws,
        Ks=Ks,
    )

    if return_paths:
        return [Path(p) for p in img_paths]
    return None


class StableVirtualCameraModel(BaseProcessingModel):
    """Stable Virtual Camera model for generating novel views."""

    def __init__(self, device: str = "cuda", model_version: float = 1.1, **kwargs):
        super().__init__(device, **kwargs)
        self.model_version = model_version
        self._model = None  # Lazy initialization
        self._ae = None
        self._conditioner = None
        self._denoiser = None

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
        image_path: str
        | Path,  # Keep image_path for backward compatibility with base class
        output_dir: str | Path,
        task: str = "img2trajvid_s-prob",
        # Parameters for img2trajvid_s-prob (single image with preset trajectory)
        num_views: int = 24,
        trajectory: str = "orbit",
        # Advanced parameters with sensible defaults
        T: int = 21,
        H: int = 576,
        W: int = 576,
        guider_types: list[int] | None = [1, 2],
        cfg: list[float] | None = [4.0, 2.0],
        chunk_strategy: str = "interp",
        camera_scale: float = 2.0,
        num_steps: int = 50,
        cfg_min: float = 1.2,
        video_save_fps: float = 30.0,
        use_traj_prior: bool = True,
        num_inputs: int | None = None,
        seed: int = 23,
        # Additional options that can be passed through
        **extra_options,
    ) -> list[Path]:
        """
        Generate novel views using Stable Virtual Camera.

        Supports all task types:
        - "img2trajvid_s-prob": Single image with preset trajectory (default)
        - "img2img": Dataset with train/test splits, generate novel views
        - "img2vid": Dataset, generate video frames
        - "img2trajvid": Dataset with train/test splits, generate trajectory video

        Args:
            image_path: Path to input image (for img2trajvid_s-prob) or dataset directory (for other tasks)
            output_dir: Directory to save generated views
            task: Task type - "img2trajvid_s-prob", "img2img", "img2vid", or "img2trajvid" (default: "img2trajvid_s-prob")
            num_views: Number of views to generate for img2trajvid_s-prob (default: 24)
            trajectory: Camera trajectory type for img2trajvid_s-prob - "orbit", "spiral", etc. (default: "orbit")
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
            num_inputs: Number of input views (None for auto-detect, only for dataset tasks)
            seed: Random seed (default: 23)
            **extra_options: Additional options to pass to version_dict["options"]

        Returns:
            List of Path objects for generated images
        """
        # Support both image_path (base class) and scene_path (new name) for backward compatibility
        scene_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build version_dict with explicit parameters
        options = {
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
        }

        # Add task-specific options
        if task == "img2trajvid_s-prob":
            options["num_targets"] = num_views
            options["traj_prior"] = trajectory
            options["use_traj_prior"] = use_traj_prior

        # Merge in any extra options
        options.update(extra_options)

        version_dict = {
            "H": H,
            "W": W,
            "T": T,
            "C": 4,
            "f": 8,
            "options": options,
        }

        img_paths = process_scene(
            task=task,
            scene_path=str(scene_path),
            output_dir=str(output_dir),
            model=self.model,
            ae=self.ae,
            conditioner=self.conditioner,
            denoiser=self.denoiser,
            version_dict=version_dict,
            num_inputs=num_inputs if task != "img2trajvid_s-prob" else 1,
            use_traj_prior=use_traj_prior,
            seed=seed,
            return_paths=True,
        )

        return img_paths if img_paths else []

    def get_description(self) -> str:
        """Get model description."""
        return (
            "Stable Virtual Camera model for generating novel views from a single image. "
            "Generates 25 frames (1 condition + 24 novel views) in an orbit trajectory. "
            "Output resolution: 576x576."
        )
