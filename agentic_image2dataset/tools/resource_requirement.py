import json
from pathlib import Path

from pydantic import BaseModel, Field

from agentic_image2dataset.tools.base import FileOperationTool
from agentic_image2dataset.utils import analyze_image_quality, get_system_resources


class ResourceRequirementToolArgs(BaseModel):
    """Arguments for the resource requirement tool."""

    model_name: str = Field(
        description="The name of the model to check requirements for"
    )
    input_path: str = Field(description="The path to the input image or directory")


class ResourceRequirementTool(FileOperationTool):
    """Tool for checking if a model can run with available resources."""

    name: str = "check_resource_requirements"
    description: str = "Check if the system has enough VRAM to run a specific model on the given input. ALWAYS call this before executing a model."
    args_schema = ResourceRequirementToolArgs

    def __init__(self, workspace_root: Path, **kwargs: object):
        super().__init__(workspace_root=workspace_root, **kwargs)

    def _run(self, model_name: str, input_path: str) -> str:
        """Check resource requirements."""

        # Load VRAM profile
        # Assuming vram_profile.json is in the project root, two levels up from this file
        profile_path = Path(__file__).parents[2] / "vram_profile.json"
        if not profile_path.exists():
            return "Error: vram_profile.json not found. Cannot check requirements."

        try:
            with open(profile_path, "r") as f:
                profiles = json.load(f)
        except Exception as e:
            return f"Error loading vram_profile.json: {str(e)}"

        if model_name not in profiles:
            return f"Model '{model_name}' not found in VRAM profile. Available models: {list(profiles.keys())}"

        model_profile = profiles[model_name]

        # Resolve input path
        try:
            path_obj = self._validate_path(input_path)
        except Exception as e:
            return f"Error validating path: {str(e)}"

        if not path_obj.exists():
            return f"Error: Input path '{input_path}' does not exist."

        # Determine dimensions
        width, height = 0, 0
        if path_obj.is_dir():
            # Check first image in directory
            images = (
                list(path_obj.glob("*.png"))
                + list(path_obj.glob("*.jpg"))
                + list(path_obj.glob("*.jpeg"))
            )
            if not images:
                return f"Error: No images found in directory '{input_path}'."
            # Use the first image to estimate requirements
            analysis = analyze_image_quality(images[0])
            width = analysis.get("width", 0)
            height = analysis.get("height", 0)
        else:
            analysis = analyze_image_quality(path_obj)
            width = analysis.get("width", 0)
            height = analysis.get("height", 0)

        if width == 0 or height == 0:
            return "Error: Could not determine image dimensions."

        num_pixels = width * height
        estimated_vram_mb = 0.0

        # Calculate VRAM usage
        if "regression" in model_profile:
            reg = model_profile["regression"]
            slope = reg.get("slope_mb_per_pixel", 0)
            intercept = reg.get("intercept_mb", 0)
            estimated_vram_mb = intercept + (slope * num_pixels)
        else:
            # Fallback to finding nearest resolution or max
            # This is a simple heuristic if regression is missing
            max_vram = 0.0
            for key, data in model_profile.items():
                if key == "regression":
                    continue
                if isinstance(data, dict) and "peak_vram_mb" in data:
                    max_vram = max(max_vram, data["peak_vram_mb"])
            estimated_vram_mb = max_vram

        # Check system resources
        resources = get_system_resources()
        available_vram_gb = resources.get("gpu_vram_available_gb", 0.0)
        total_vram_gb = resources.get("gpu_vram_total_gb", 0.0)

        estimated_vram_gb = estimated_vram_mb / 1024

        status_msg = (
            f"Model: {model_name}\n"
            f"Input: {input_path} ({width}x{height})\n"
            f"Estimated VRAM: {estimated_vram_gb:.2f} GB\n"
            f"Available VRAM: {available_vram_gb:.2f} GB (Total: {total_vram_gb:.2f} GB)\n"
        )

        if available_vram_gb >= estimated_vram_gb:
            return f"✅ Resources Sufficient.\n{status_msg}\nYou can proceed with execution."
        else:
            return f"❌ Insufficient Resources.\n{status_msg}\nWARNING: Execution may fail with OOM."
