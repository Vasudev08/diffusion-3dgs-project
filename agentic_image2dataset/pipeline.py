"""
Main agentic pipeline orchestrator.
"""

import shutil
from pathlib import Path

from agentic_image2dataset.agent import AgenticImageProcessor
from agentic_image2dataset.config import LLMConfig, PipelineConfig
from agentic_image2dataset.models.base import ModelRegistry
from agentic_image2dataset.models.image_edit import QwenImageEditModel
from agentic_image2dataset.models.super_resolution import AdcSRModel, DiffBIRModel
from agentic_image2dataset.models.view_generation import StableVirtualCameraModel
from agentic_image2dataset.utils import (
    analyze_image_quality,
    detect_image_issues,
    fix_transforms,
)


class AgenticPipeline:
    """Main agentic pipeline for processing images into 3DGS datasets."""

    def __init__(self, config: PipelineConfig):
        self.config: PipelineConfig = config
        self.model_registry: ModelRegistry = ModelRegistry()

        # Initialize models
        self._initialize_models()

        # Initialize agent (will be re-initialized per process call with specific workspace)
        self.agent_config: LLMConfig = config.llm

    def _initialize_models(self):
        """Register model factories for lazy initialization."""

        # Register Stable Virtual Camera model factory
        def create_stable_virtual_camera():
            return StableVirtualCameraModel(
                device=self.config.model.device, model_version=1.1
            )

        self.model_registry.register(
            "stable_virtual_camera",
            create_stable_virtual_camera,
            role="view_generation",
        )

        # Register DiffBIR model factory
        def create_diffbir():
            return DiffBIRModel(
                device=self.config.model.device,
                scale=self.config.model.super_resolution_factor,
            )

        self.model_registry.register("diffbir", create_diffbir, role="super_resolution")

        # Register AdcSR model factory
        def create_adcsr():
            return AdcSRModel(
                device=self.config.model.device,
                scale=self.config.model.super_resolution_factor,
            )

        self.model_registry.register("adcsr", create_adcsr, role="super_resolution")

        # Register Qwen Image Edit model factory
        def create_qwen_edit():
            return QwenImageEditModel(device=self.config.model.device)

        self.model_registry.register(
            "qwen_image_edit", create_qwen_edit, role="image_editing"
        )

    def process(
        self,
        input_image: Path,
        output_dir: Path,
    ) -> dict[str, object]:
        """
        Process a single input image into a 3DGS dataset.

        Args:
            input_image: Path to the input image
            output_dir: Directory for output dataset

        Returns:
            Dictionary with processing results
        """
        input_image = Path(input_image)
        output_dir = Path(output_dir)

        if not input_image.exists():
            return {"success": False, "error": f"Input image not found: {input_image}"}

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create a temporary workspace for the agent
        # We use a subdirectory in output_dir to keep it contained, but it could be anywhere
        workspace_dir = output_dir / "agent_workspace"
        workspace_dir.mkdir(exist_ok=True)

        try:
            # Copy input image to workspace
            workspace_input_image = workspace_dir / input_image.name
            shutil.copy2(input_image, workspace_input_image)

            # Initialize agent for this specific workspace
            agent = AgenticImageProcessor(
                self.agent_config, self.model_registry, workspace_root=workspace_dir
            )

            # Step 1: Analyze input image
            if self.config.verbose:
                print("Analyzing input image...")

            analysis = analyze_image_quality(workspace_input_image)
            issues = detect_image_issues(workspace_input_image)

            if self.config.verbose:
                print(f"Image analysis: {analysis}")
                if issues:
                    print(f"Detected issues: {issues}")

            # Step 2: Plan processing with agent
            if self.config.verbose:
                print("Planning processing pipeline...")

            # Agent decides its own output structure, but we instructed it to use 'output' dir
            agent_output_dir = workspace_dir / "output"

            plan_result = agent.plan_processing(workspace_input_image)
            if not plan_result["success"]:
                return {
                    "success": False,
                    "error": f"Planning failed: {plan_result['plan']}",
                }

            if self.config.verbose:
                print(f"Processing plan: {plan_result['plan']}")

            # Step 3: Execute processing
            if self.config.verbose:
                print("Executing processing pipeline...")

            plan_description = plan_result["plan"]
            if not isinstance(plan_description, str):
                plan_description = str(plan_description)
            execution_result = agent.execute_plan(plan_description)
            if not execution_result["success"]:
                return {
                    "success": False,
                    "error": f"Execution failed: {execution_result['result']}",
                }

            # Step 4: Post-processing and cleanup
            # Check for transforms.json in the agent's output
            transforms_json = agent_output_dir / "transforms.json"
            if transforms_json.exists():
                if self.config.verbose:
                    print("Found transforms.json, applying fixes...")
                try:
                    fix_transforms(transforms_json)
                except Exception as e:
                    print(f"Warning: Failed to fix transforms.json: {e}")

            # Move results to final output directory
            # We expect the agent to have put everything in 'agent_output_dir'
            if agent_output_dir.exists():
                if self.config.verbose:
                    print(f"Moving results to {output_dir}...")

                # Move contents of agent_output_dir to output_dir
                for item in agent_output_dir.iterdir():
                    dest = output_dir / item.name
                    if dest.exists():
                        if dest.is_dir():
                            shutil.rmtree(dest)
                        else:
                            dest.unlink()
                    shutil.move(str(item), str(dest))
            else:
                print("Warning: Agent did not create an 'output' directory.")

            return {
                "success": True,
                "output_dir": output_dir,
                "analysis": analysis,
                "issues": issues,
                "plan": plan_result["plan"],
                "execution_result": execution_result["result"],
            }

        finally:
            # Cleanup workspace
            if workspace_dir.exists():
                shutil.rmtree(workspace_dir, ignore_errors=True)

    def get_available_models(self) -> list[str]:
        """Get list of available models."""
        return self.model_registry.list_available()

    def get_model_info(self, model_name: str) -> dict[str, object] | None:
        """Get information about a specific model."""
        model = self.model_registry.get(model_name)
        if model is None:
            return None

        return {
            "name": model_name,
            "description": model.get_description(),
        }
