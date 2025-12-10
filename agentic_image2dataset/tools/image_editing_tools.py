import json
import math
import sys
from pathlib import Path
from typing import override

import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel, Field

from agentic_image2dataset.models.base import ModelRegistry
from agentic_image2dataset.tools.base import FileOperationTool


class GenerateSurroundingViewsToolArgs(BaseModel):
    """Arguments for the generate surrounding views tool."""

    image_path: str = Field(
        description="Required: The input image path to process.",
    )
    output_dir: str = Field(
        description="Required: The output directory path to save the results.",
    )
    views: list[dict[str, object]] = Field(
        default=[
            {"angle": 90, "direction": "left"},
            {"angle": 90, "direction": "right"},
            {"angle": 180, "direction": "left"},
        ],
        description='Optional: List of views to generate. Each item should be a dict with "angle" (int) and "direction" (str, "left" or "right"). Default generates 90L, 90R, and 180.',
    )
    model_name: str = Field(
        default="qwen_image_edit",
        description="Optional: The name of the model to use. Defaults to 'qwen_image_edit'.",
    )


class GenerateSurroundingViewsTool(FileOperationTool):
    """Tool for generating multiple views of the subject."""

    name: str = "generate_surrounding_views"
    description: str = (
        "Generates multiple views of the subject by rotating the camera around it."
    )
    args_schema = GenerateSurroundingViewsToolArgs
    model_registry: ModelRegistry

    def __init__(
        self,
        model_registry: ModelRegistry,
        workspace_root: Path,
        **kwargs: object,
    ):
        super().__init__(
            model_registry=model_registry,
            workspace_root=workspace_root,
            **kwargs,
        )

    @override
    def _run(
        self,
        image_path: str,
        output_dir: str,
        views: list[dict[str, object]] | None = None,
        model_name: str = "qwen_image_edit",
    ) -> str:
        """Execute the view generation task."""
        if views is None:
            views = [
                {"angle": 90, "direction": "left"},
                {"angle": 90, "direction": "right"},
                {"angle": 180, "direction": "left"},
            ]

        model = self.model_registry.get(model_name)
        if model is None:
            return f"Model '{model_name}' not found. Available models: {self.model_registry.list_available()}"

        actual_input_path = self._validate_path(image_path)
        if not actual_input_path.exists():
            return f"Error: Input path '{image_path}' does not exist."

        actual_output_path = self._validate_path(output_dir)
        all_results = []

        for view in views:
            angle = view.get("angle")
            direction = view.get("direction", "")

            # Construct prompt
            if angle == 180:
                prompt = "Rotate the camera 180 degrees."
            else:
                prompt = f"Rotate the camera {angle} degrees to the {direction}."

            try:
                results = model.process(
                    actual_input_path,
                    actual_output_path,
                    prompt=prompt,
                )
                all_results.extend(results)
            except Exception as e:
                return f"Failed to generate view {view} using '{model_name}': {str(e)}"

        # Generate transforms.json and dataset split
        try:
            # Prepare info list for metadata generation
            generated_info = []
            for res_path, view in zip(all_results, views):
                generated_info.append(
                    {
                        "path": res_path,
                        "angle": view.get("angle"),
                        "direction": view.get("direction", ""),
                    }
                )

            self._generate_transforms_json(
                actual_output_path, actual_input_path, generated_info
            )
        except Exception as e:
            return f"Successfully generated views, but failed to generate transforms.json: {str(e)}. Outputs: {[str(p) for p in all_results]}"

        return f"Successfully generated {len(all_results)} views. Metadata saved to transforms.json and train_test_split.json. Outputs: {[str(p) for p in all_results]}"

    def _generate_transforms_json(
        self,
        output_dir: Path,
        original_image_path: Path,
        generated_images_info: list[dict],
    ):
        """Generates transforms.json using seva.eval.create_transforms_simple."""
        # Add stable-virtual-camera to sys.path if needed
        svc_path = self.workspace_root / "stable-virtual-camera"
        if str(svc_path) not in sys.path:
            sys.path.append(str(svc_path))

        try:
            from seva.eval import create_transforms_simple
            from seva.geometry import get_default_intrinsics, viewmatrix
        except ImportError:
            return "Error: Could not import create_transforms_simple from seva.eval. Please ensure stable-virtual-camera is installed or in the workspace."

        # Define camera parameters
        distance = 4.0
        fov_deg = 54
        target = np.array([0.0, 0.0, 0.0])
        up = np.array([0, -1, 0])
        # Collect all image paths and their corresponding camera info
        # Start with original image (0 degrees)
        img_paths = [str(original_image_path)]

        # Helper function to calculate camera position from angle
        def get_camera_pos(angle_deg, direction):
            theta = math.radians(angle_deg)
            if direction == "left":
                theta = -theta

            x = distance * math.sin(theta)
            y = 0.0
            z = -distance * math.cos(theta)
            return np.array([x, y, z])

        # Original camera at 0 degrees
        cam_pos_0 = get_camera_pos(0, "")
        c2ws_list = [viewmatrix(target, up, cam_pos_0, subtract_position=True)]

        # Add generated images
        for info in generated_images_info:
            path = info["path"]
            angle = info["angle"]
            direction = info["direction"]

            cam_pos = get_camera_pos(angle, direction)
            img_paths.append(str(path))
            c2ws_list.append(viewmatrix(target, up, cam_pos, subtract_position=True))

        c2ws = torch.from_numpy(np.stack(c2ws_list)).float()

        # Get image dimensions from the first image
        with Image.open(original_image_path) as img:
            w, h = img.size
            aspect_ratio = w / h

        num_images = len(img_paths)
        # Prepare tensors for create_transforms_simple
        img_whs = torch.tensor([[w, h]] * num_images).float()
        fovs = np.full((num_images,), fov_deg)

        # Create Intrinsics (K)
        Ks = get_default_intrinsics(fovs, aspect_ratio=aspect_ratio)
        Ks[:, :2] *= (
            torch.tensor([w, h]).reshape(1, -1, 1).repeat(Ks.shape[0], 1, 1)
        )  # normalized
        Ks = Ks.numpy()

        # Call the function
        create_transforms_simple(
            save_path=str(output_dir),
            img_paths=img_paths,
            img_whs=img_whs,
            c2ws=c2ws,
            Ks=Ks,
        )

        # Also generate the split file required by SVC
        self._generate_dataset_split(output_dir, num_images)

    def _generate_dataset_split(self, output_dir: Path, num_images: int):
        """Generates train_test_split_{N}.json."""
        split_data = {
            "train_ids": list(range(num_images)),
            "test_ids": list(range(num_images)),
        }
        split_path = output_dir / f"train_test_split_{num_images}.json"
        with open(split_path, "w") as f:
            json.dump(split_data, f, indent=4)
