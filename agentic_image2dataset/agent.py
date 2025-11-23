"""
LangChain agent for orchestrating the image processing pipeline.
"""

import json
import os
import shutil
from abc import ABCMeta
from dataclasses import dataclass, field
from pathlib import Path
from typing import override

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from agentic_image2dataset.config import LLMConfig
from agentic_image2dataset.models.base import ModelRegistry
from agentic_image2dataset.utils import (
    analyze_image_quality,
    detect_image_issues,
)


def create_llm(config: LLMConfig) -> BaseChatModel:
    """
    Factory function to create the appropriate LLM based on provider.

    All LangChain models automatically read API keys from environment variables:
    - Google: GOOGLE_API_KEY
    - OpenAI: OPENAI_API_KEY
    - Anthropic: ANTHROPIC_API_KEY

    Users must set these environment variables before running the code.

    Args:
        config: LLM configuration containing provider, model_name, and other settings

    Returns:
        BaseChatModel instance for the specified provider

    Raises:
        ValueError: If provider is not supported or required dependencies are missing
    """
    provider = config.provider

    if provider == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError(
                "langchain-google-genai is required for Google provider. Install it with: pip install langchain-google-genai"
            )

        # ChatGoogleGenerativeAI reads GOOGLE_API_KEY from environment automatically
        return ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
        )

    elif provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "langchain-openai is required for OpenAI provider. Install it with: pip install langchain-openai"
            )

        # ChatOpenAI reads OPENAI_API_KEY from environment automatically
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            max_completion_tokens=config.max_tokens,
        )

    elif provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(
                "langchain-anthropic is required for Anthropic provider. Install it with: pip install langchain-anthropic"
            )

        # ChatAnthropic reads ANTHROPIC_API_KEY from environment automatically
        return ChatAnthropic(
            model_name=config.model_name,
            temperature=config.temperature,
            max_tokens_to_sample=config.max_tokens,
            timeout=None,
            stop=None,
        )

    else:
        raise ValueError(
            f"Unsupported provider: {provider}. Supported providers are: google, openai, anthropic"
        )


@dataclass
class ModelGuidance:
    """Guidance for choosing between models in a role."""

    model_name: str
    guidance_text: str


@dataclass
class RoleConfiguration:
    """Configuration for a model role in the prompt."""

    display_name: str
    general_guidance: str | None = None
    model_guidance: list[ModelGuidance] = field(default_factory=list)
    shared_notes: str | None = None


class FileOperationTool(BaseTool, metaclass=ABCMeta):
    """Base class for file operation tools with safety checks."""

    workspace_root: Path

    def __init__(self, workspace_root: Path, **kwargs: object):
        super().__init__(workspace_root=workspace_root, **kwargs)

    def _validate_path(self, path_str: str) -> Path:
        """Validate that the path is within the workspace."""
        try:
            path = (self.workspace_root / path_str).resolve()
            if not str(path).startswith(str(self.workspace_root.resolve())):
                raise ValueError(
                    f"Access denied: Path '{path_str}' is outside the workspace."
                )
            return path
        except Exception as e:
            raise ValueError(f"Invalid path '{path_str}': {str(e)}")


class ImageAnalysisTool(FileOperationTool):
    """Tool for analyzing input images."""

    name: str = "analyze_image"
    description: str = "Analyze an image to determine its quality, characteristics, and processing needs"

    def __init__(self, workspace_root: Path, **kwargs: object):
        super().__init__(workspace_root=workspace_root, **kwargs)

    @override
    def _run(self, image_path: str, query: str = "") -> str:
        """Analyze the input image."""
        try:
            actual_path = self._validate_path(image_path)
            if not actual_path.exists():
                return f"Error: Image path '{image_path}' does not exist."

            analysis = analyze_image_quality(actual_path)
            issues = detect_image_issues(actual_path)

            result = {
                "analysis": analysis,
                "issues": issues,
            }

            return json.dumps(result, indent=2)
        except Exception as e:
            return f"Error analyzing image: {str(e)}"


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
        from agentic_image2dataset.utils import get_system_resources

        # Load VRAM profile
        # Assuming vram_profile.json is in the project root, one level up from this package
        profile_path = Path(__file__).parents[1] / "vram_profile.json"
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


class ListDirectoryTool(FileOperationTool):
    """Tool for listing files in a directory."""

    name: str = "list_directory"
    description: str = "List files in a directory within the workspace. Use this to check generated outputs."

    class Args(BaseModel):
        directory: str = Field(
            default=".", description="Directory to list (relative to workspace root)"
        )

    args_schema = Args

    @override
    def _run(self, directory: str = ".") -> str:
        try:
            target_dir = self._validate_path(directory)
            if not target_dir.exists():
                return f"Directory '{directory}' does not exist."
            if not target_dir.is_dir():
                return f"Path '{directory}' is not a directory."

            files = sorted(os.listdir(target_dir))
            if not files:
                return "Directory is empty."

            return "\n".join(files)
        except Exception as e:
            return f"Error listing directory: {str(e)}"


class CopyFileTool(FileOperationTool):
    """Tool for copying files."""

    name: str = "copy_file"
    description: str = "Copy a file or directory within the workspace."

    class Args(BaseModel):
        source: str = Field(description="Source path (relative to workspace root)")
        destination: str = Field(
            description="Destination path (relative to workspace root)"
        )

    args_schema = Args

    @override
    def _run(self, source: str, destination: str) -> str:
        try:
            src_path = self._validate_path(source)
            dst_path = self._validate_path(destination)

            if not src_path.exists():
                return f"Source '{source}' does not exist."

            if src_path.is_dir():
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            else:
                # Ensure parent directory exists
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_path)

            return f"Successfully copied '{source}' to '{destination}'."
        except Exception as e:
            return f"Error copying file: {str(e)}"


class MoveFileTool(FileOperationTool):
    """Tool for moving files."""

    name: str = "move_file"
    description: str = "Move a file or directory within the workspace."

    class Args(BaseModel):
        source: str = Field(description="Source path (relative to workspace root)")
        destination: str = Field(
            description="Destination path (relative to workspace root)"
        )

    args_schema = Args

    @override
    def _run(self, source: str, destination: str) -> str:
        try:
            src_path = self._validate_path(source)
            dst_path = self._validate_path(destination)

            if not src_path.exists():
                return f"Source '{source}' does not exist."

            # Ensure parent directory exists
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_path), str(dst_path))

            return f"Successfully moved '{source}' to '{destination}'."
        except Exception as e:
            return f"Error moving file: {str(e)}"


class DeleteFileTool(FileOperationTool):
    """Tool for deleting files."""

    name: str = "delete_file"
    description: str = (
        "Delete a file or directory within the workspace. USE WITH CAUTION."
    )

    class Args(BaseModel):
        path: str = Field(description="Path to delete (relative to workspace root)")

    args_schema = Args

    @override
    def _run(self, path: str) -> str:
        try:
            target_path = self._validate_path(path)

            if not target_path.exists():
                return f"Path '{path}' does not exist."

            if target_path.is_dir():
                shutil.rmtree(target_path)
            else:
                os.remove(target_path)

            return f"Successfully deleted '{path}'."
        except Exception as e:
            return f"Error deleting file: {str(e)}"


class AgenticImageProcessor:
    """LangChain-based agent for orchestrating image processing."""

    def __init__(
        self,
        config: LLMConfig,
        model_registry: ModelRegistry,
        workspace_root: Path | None = None,
    ):
        self.config: LLMConfig = config
        self.model_registry: ModelRegistry = model_registry

        # Set workspace root, default to current directory if not provided
        workspace_root = workspace_root if workspace_root else Path.cwd()
        workspace_root.mkdir(parents=True, exist_ok=True)

        # Initialize LLM using factory function
        self.llm: BaseChatModel = create_llm(config)

        # Create tools
        self.tools: list[BaseTool] = [
            ImageAnalysisTool(workspace_root=workspace_root),
            ResourceRequirementTool(workspace_root=workspace_root),
            ModelExecutionTool(self.model_registry, workspace_root=workspace_root),
            ListDirectoryTool(workspace_root=workspace_root),
            CopyFileTool(workspace_root=workspace_root),
            MoveFileTool(workspace_root=workspace_root),
            DeleteFileTool(workspace_root=workspace_root),
        ]

        # Create memory
        self.memory: ConversationBufferWindowMemory = ConversationBufferWindowMemory(
            memory_key="chat_history", return_messages=True, k=10
        )

        # Create agent
        self.agent: AgentExecutor = self._create_agent()

    def _get_role_configurations(self) -> dict[str, RoleConfiguration]:
        """Get structured configurations for each role."""
        return {
            "view_generation": RoleConfiguration(
                display_name="View Generation Models",
                general_guidance=None,
                model_guidance=[
                    ModelGuidance(
                        model_name="stable_virtual_camera",
                        guidance_text="Works best when the input image is aligned to make generating views in an orbit trajectory easier (e.g., object centered and upright). Output image dimensions are 576x576",
                    ),
                ],
            ),
            "super_resolution": RoleConfiguration(
                display_name="Super-Resolution Models",
                general_guidance="When choosing between super-resolution models:",
                model_guidance=[
                    ModelGuidance(
                        model_name="diffbir",
                        guidance_text="Primary choice for maximum quality. High VRAM usage.",
                    ),
                    ModelGuidance(
                        model_name="adcsr",
                        guidance_text="Fallback choice. Lower VRAM usage, faster processing, competitive quality.",
                    ),
                ],
                shared_notes="Both models support 4x upscaling. If 'diffbir' fails the resource check, automatically use 'adcsr'. IMPORTANT: These models can process either a single image OR a directory containing multiple images. When applying super-resolution AFTER view generation, you should process the directory containing all generated views to upscale them all at once.",
            ),
            "image_editing": RoleConfiguration(
                display_name="Image Editing Models",
                general_guidance="Use for generating alternative views or modifying image perspectives:",
                model_guidance=[
                    ModelGuidance(
                        model_name="qwen_image_edit",
                        guidance_text="Use 'qwen_image_edit' to generate views parallel to the horizontal axis or modify image content based on text prompts",
                    ),
                ],
                shared_notes="Useful for creating additional camera angles when view generation models are insufficient",
            ),
        }

    def _build_model_descriptions(self, roles: dict[str, list[str]]) -> str:
        """Build model descriptions from role configurations."""
        role_configs = self._get_role_configurations()
        model_descriptions: list[str] = []

        for role_key, model_names in roles.items():
            config = role_configs.get(role_key)
            if not config:
                # Fallback for roles without explicit configuration
                config = RoleConfiguration(
                    display_name=role_key.replace("_", " ").title() + " Models"
                )

            # Add role header
            model_descriptions.append(f"{config.display_name}:")

            # Add individual model descriptions
            for model_name in model_names:
                model = self.model_registry.get(model_name)
                if model:
                    description = model.get_description()
                    model_descriptions.append(f"  - {model_name}: {description}")

            # Add general guidance if present
            if config.general_guidance:
                model_descriptions.append(f"\n{config.general_guidance}")

            # Add model-specific guidance
            if config.model_guidance:
                for guidance in config.model_guidance:
                    if guidance.model_name in model_names:
                        model_descriptions.append(f"  - {guidance.guidance_text}")

            # Add shared notes if present
            if config.shared_notes:
                model_descriptions.append(f"  - {config.shared_notes}")

            # Add spacing between roles
            model_descriptions.append("")

        return (
            "\n".join(model_descriptions).strip()
            if model_descriptions
            else "No models available."
        )

    def _create_agent(self) -> AgentExecutor:
        """Create the LangChain agent."""

        # Get available models and organize by role
        roles = self.model_registry.get_all_roles()

        # Build model descriptions using structured configurations
        models_text = self._build_model_descriptions(roles)

        system_prompt = f"""You are an expert image processing agent that specializes in creating 3D Gaussian Splatting datasets from single input images.

Your task is to:
1. Analyze the input image to understand its characteristics and quality
2. Decide on the optimal processing pipeline based on the image analysis
3. Execute the necessary processing steps in the correct order
4. Ensure the final output is suitable for 3DGS training

Available processing models:
{models_text}


Processing decisions should consider:
- Image resolution and quality
- Scene complexity and content
- Blur, brightness, and contrast issues
- Optimal number of views for 3DGS training
- Available computational resources and time constraints

CRITICAL RESOURCE MANAGEMENT:
1. Before executing ANY model, you MUST check if the system has enough VRAM using the 'check_resource_requirements' tool.
2. If the check returns "Insufficient Resources":
   - You MUST automatically switch to a less resource-intensive model if one is available for the same task (e.g., switch from 'diffbir' to 'adcsr').
   - Do NOT ask the user for permission to switch.
   - Only stop and warn the user if NO alternative model is available.

File Management Instructions:
- You are responsible for managing your files within the workspace.
- The 'input_path' for models must be a valid path within the workspace.
- ALWAYS use the 'list_directory' tool after running a model to verify the outputs.
- Decide which images to keep. You should keep the final dataset images and potentially useful intermediate steps.
- Create a directory named 'output' for your final results.
- Inside 'output', create a subdirectory named 'images' and place your final images there.
- Also place the 'transforms.json' file (generated by the view generation model) in the 'output' directory.
- You may create backup directories for intermediate steps if needed.
- Use 'copy_file', 'move_file', and 'delete_file' to organize your workspace.
- BE CAREFUL with 'delete_file'. Only delete files you are sure are no longer needed.

Always explain your reasoning and provide clear feedback on the processing steps."""

        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        # Create the agent
        agent = create_openai_functions_agent(
            llm=self.llm, tools=self.tools, prompt=prompt
        )

        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10,
        )

    def plan_processing(self, image_path: Path) -> dict[str, str | bool]:
        """Plan the processing pipeline for the given image."""
        # Create planning prompt
        planning_prompt = f"""
Analyze the input image at {image_path} and create a processing plan.

First, use the analyze_image tool to understand the image characteristics.
Then, based on the analysis, decide which models to use and in what order.

Consider:
1. Should we apply super-resolution before, after view generation, or both?
2. How many views should we generate?
3. What parameters should we use for each model?

Provide a detailed plan with reasoning for each decision.
"""

        response = self.agent.invoke({"input": planning_prompt})
        return {"plan": response["output"], "success": True}

    def execute_plan(self, plan_description: str) -> dict[str, str | bool]:
        """Execute the processing plan."""
        execution_prompt = f"""
Execute the following processing plan:

{plan_description}

Use the available tools to:
1. Analyze the image if not already done
2. Execute the necessary models in the correct order
3. Provide feedback on the results

Be systematic and check the results of each step before proceeding.
"""

        response = self.agent.invoke({"input": execution_prompt})
        return {"result": response["output"], "success": True}
