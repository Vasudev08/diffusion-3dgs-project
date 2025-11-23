import json
from pathlib import Path
from typing import override

from pydantic import BaseModel, Field

from agentic_image2dataset.models.base import ModelRegistry
from agentic_image2dataset.tools.base import FileOperationTool


class ModelExecutionToolArgs(BaseModel):
    """Arguments for the model execution tool."""

    model_name: str = Field(description="The name of the model to execute")
    parameters: str = Field(
        default="{}",
        description='Model parameters in JSON format. Must be a valid JSON string. Example: \'{"scale": 4, "batch_size": 1}\'',
    )
    input_path: str = Field(
        description="Required: The input image or directory path to process. You must explicitly provide the path.",
    )
    output_dir: str = Field(
        description="Required: The output directory path to save the results. You must explicitly provide the path.",
    )


class ModelExecutionTool(FileOperationTool):
    """Tool for executing processing models."""

    name: str = "execute_model"
    description: str = "Execute a processing model on an image or directory. You MUST provide the 'input_path' parameter. The 'parameters' argument must be provided as a JSON string."
    args_schema = ModelExecutionToolArgs
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
        self, model_name: str, input_path: str, output_dir: str, parameters: str = "{}"
    ) -> str:
        """Execute a model with given parameters."""
        model = self.model_registry.get(model_name)
        if model is None:
            return f"Model '{model_name}' not found. Available models: {self.model_registry.list_available()}"

        # Parse parameters
        try:
            params: dict[str, object] = json.loads(parameters)
        except json.JSONDecodeError:
            return "Error: parameters must be a valid JSON string."

        # Use provided input_path
        actual_input_path = self._validate_path(input_path)
        if not actual_input_path.exists():
            return f"Error: Input path '{input_path}' does not exist."

        actual_output_path = self._validate_path(output_dir)

        # Execute the model
        try:
            results = model.process(actual_input_path, actual_output_path, **params)
        except Exception as e:
            return f"Failed to execute model '{model_name}': {str(e)}"

        return f"Model '{model_name}' executed successfully. Generated {len(results)} outputs: {[str(p) for p in results]}"
