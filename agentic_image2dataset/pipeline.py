"""
Main agentic pipeline orchestrator.
"""

import json
import shutil
from pathlib import Path

from agentic_image2dataset.agent import AgenticImageProcessor
from agentic_image2dataset.colmap_processor import COLMAPProcessor
from agentic_image2dataset.config import PipelineConfig
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
        self.colmap_processor: COLMAPProcessor = COLMAPProcessor(
            quality=config.model.colmap_quality, device=config.model.device
        )

        # Initialize models
        self._initialize_models()

        # Initialize agent
        self.agent: AgenticImageProcessor = AgenticImageProcessor(
            config.llm, self.model_registry
        )

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
        num_views: int | None = None,
        skip_colmap: bool = False,
    ) -> dict[str, object]:
        """
        Process a single input image into a 3DGS dataset.

        Args:
            input_image: Path to the input image
            output_dir: Directory for output dataset
            num_views: Number of views to generate (if None, agent decides)
            skip_colmap: Skip COLMAP preprocessing

        Returns:
            Dictionary with processing results
        """
        input_image = Path(input_image)
        output_dir = Path(output_dir)

        if not input_image.exists():
            return {"success": False, "error": f"Input image not found: {input_image}"}

        # Create output directory structure
        output_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = output_dir / "temp"
        temp_dir.mkdir(exist_ok=True)

        # Step 1: Analyze input image
        if self.config.verbose:
            print("Analyzing input image...")

        analysis = analyze_image_quality(input_image)
        issues = detect_image_issues(input_image)

        if self.config.verbose:
            print(f"Image analysis: {analysis}")
            if issues:
                print(f"Detected issues: {issues}")

        # Step 2: Plan processing with agent
        if self.config.verbose:
            print("Planning processing pipeline...")

        plan_result = self.agent.plan_processing(input_image, temp_dir)
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
        execution_result = self.agent.execute_plan(plan_description)
        if not execution_result["success"]:
            return {
                "success": False,
                "error": f"Execution failed: {execution_result['result']}",
            }

        # Step 4: Collect generated images
        generated_images = self._collect_generated_images(temp_dir)

        if not generated_images:
            return {
                "success": False,
                "error": "No images were generated by the processing pipeline",
            }

        # Step 5: Check for camera information (transforms.json from Stable Virtual Camera)
        # or run COLMAP preprocessing (if not skipped)
        transforms_json = temp_dir / "transforms.json"
        has_transforms = transforms_json.exists()

        if has_transforms:
            if self.config.verbose:
                print("Found transforms.json from Stable Virtual Camera")
                print(
                    "Note: transforms.json is in NeRF format and compatible with gsplat"
                )
                print(
                    "For gsplat's COLMAP parser, you may need to convert it or use a NeRF-style loader"
                )
            # Use the transforms.json directly - no need for COLMAP
            self._create_final_dataset(
                images_dir=temp_dir, colmap_dir=None, output_dir=output_dir
            )
        elif not skip_colmap and not self.config.skip_colmap:
            if self.config.verbose:
                print("Running COLMAP preprocessing...")

            colmap_result = self.colmap_processor.process_images(
                image_dir=temp_dir, output_dir=output_dir / "colmap_temp"
            )

            if not colmap_result["success"]:
                if self.config.verbose:
                    print(
                        f"COLMAP failed: {colmap_result.get('error', 'Unknown error')}"
                    )
                    print("Continuing without COLMAP preprocessing...")
            else:
                # Create final dataset structure
                self._create_final_dataset(
                    images_dir=temp_dir,
                    colmap_dir=colmap_result["sparse_dir"],
                    output_dir=output_dir,
                )
        else:
            # Just copy images to final structure
            self._create_final_dataset(
                images_dir=temp_dir, colmap_dir=None, output_dir=output_dir
            )

        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)

        return {
            "success": True,
            "output_dir": output_dir,
            "generated_images": len(generated_images),
            "analysis": analysis,
            "issues": issues,
            "plan": plan_result["plan"],
            "execution_result": execution_result["result"],
        }

    def _collect_generated_images(self, temp_dir: Path) -> list[Path]:
        """Collect all generated images from the temp directory."""
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        generated_images = []

        for file_path in temp_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                generated_images.append(file_path)

        return generated_images

    def _create_final_dataset(
        self, images_dir: Path, colmap_dir: Path | None, output_dir: Path
    ):
        """Create the final dataset structure."""
        # Create images directory
        final_images_dir = output_dir / "images"
        final_images_dir.mkdir(exist_ok=True)

        # Copy all images
        for img_file in images_dir.glob("*"):
            if img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
                shutil.copy2(img_file, final_images_dir / img_file.name)

        # Copy transforms.json if it exists (from Stable Virtual Camera)
        transforms_json = images_dir / "transforms.json"
        if transforms_json.exists():
            shutil.copy2(transforms_json, output_dir / "transforms.json")

            # Fix the transforms.json in the output directory
            try:
                fix_transforms(output_dir / "transforms.json")
            except Exception as e:
                print(f"Warning: Failed to fix transforms.json: {e}")

            if self.config.verbose:
                print(
                    f"Copied and fixed transforms.json to {output_dir / 'transforms.json'}"
                )

        # Copy COLMAP results if available
        if colmap_dir and colmap_dir.exists():
            final_sparse_dir = output_dir / "sparse" / "0"
            final_sparse_dir.mkdir(parents=True, exist_ok=True)

            for file in colmap_dir.glob("*.bin"):
                shutil.copy2(file, final_sparse_dir / file.name)

        # Create metadata file
        metadata = {
            "dataset_type": "3dgs_ready",
            "num_images": len(list(final_images_dir.glob("*"))),
            "has_colmap": colmap_dir is not None and colmap_dir.exists(),
            "has_transforms_json": transforms_json.exists(),
            "created_by": "agentic_image2dataset",
        }

        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

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
