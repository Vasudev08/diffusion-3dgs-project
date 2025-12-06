from pathlib import Path
from typing import override

from pydantic import BaseModel, Field

from agentic_image2dataset.models.base import ModelRegistry
from agentic_image2dataset.tools.base import FileOperationTool


class CenterImageToolArgs(BaseModel):
    """Arguments for the center image tool."""

    image_path: str = Field(
        description="Required: The input image path to process.",
    )
    output_dir: str = Field(
        description="Required: The output directory path to save the results.",
    )
    model_name: str = Field(
        default="qwen_image_edit",
        description="Optional: The name of the model to use. Defaults to 'qwen_image_edit'.",
    )


class CenterImageTool(FileOperationTool):
    """Tool for centering the main subject in an image."""

    name: str = "center_subject"
    description: str = (
        "Centers the main subject in the image using an image editing model."
    )
    args_schema = CenterImageToolArgs
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
        model_name: str = "qwen_image_edit",
    ) -> str:
        """Execute the centering task."""
        model = self.model_registry.get(model_name)
        if model is None:
            return f"Model '{model_name}' not found. Available models: {self.model_registry.list_available()}"

        actual_input_path = self._validate_path(image_path)
        if not actual_input_path.exists():
            return f"Error: Input path '{image_path}' does not exist."

        actual_output_path = self._validate_path(output_dir)

        try:
            results = model.process(
                actual_input_path,
                actual_output_path,
                prompt="Center the image on the main subject.",
            )
        except Exception as e:
            return f"Failed to center subject using '{model_name}': {str(e)}"

        return f"Successfully centered subject. Generated output: {[str(p) for p in results]}"


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

        return f"Successfully generated {len(all_results)} views. Outputs: {[str(p) for p in all_results]}"
