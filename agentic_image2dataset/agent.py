"""
LangChain agent for orchestrating the image processing pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path

from langchain.agents import create_agent
from langchain.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable

from agentic_image2dataset.config import LLMConfig
from agentic_image2dataset.models.base import ModelRegistry
from agentic_image2dataset.tools import (
    CopyFileTool,
    DeleteFileTool,
    ImageAnalysisTool,
    ListDirectoryTool,
    ModelExecutionTool,
    MoveFileTool,
    ResourceRequirementTool,
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

        # Create agent
        self.agent: Runnable = self._create_agent()

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
                        model_name="hypir",
                        guidance_text="Primary choice for best quality results with efficient VRAM usage. Uses Stable Diffusion prior for superior image enhancement.",
                    ),
                    ModelGuidance(
                        model_name="adcsr",
                        guidance_text="Fallback choice. Lower VRAM usage, faster processing, competitive quality.",
                    ),
                    ModelGuidance(
                        model_name="diffbir",
                        guidance_text="Alternative high-quality choice. High VRAM usage, slower processing.",
                    ),
                ],
                shared_notes="All models support 4x upscaling. If 'hypir' fails the resource check, automatically use 'adcsr'. 'diffbir' is available as an alternative if needed. IMPORTANT: These models can process either a single image OR a directory containing multiple images. When applying super-resolution AFTER view generation, you should process the directory containing all generated views to upscale them all at once.",
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

    def _create_agent(self) -> Runnable:
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
   - You MUST automatically switch to a less resource-intensive model if one is available for the same task (e.g., switch from 'hypir' to 'adcsr').
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

        # Create the agent using the new create_agent API
        return create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=system_prompt,
        )

    def plan_processing(self, image_path: Path) -> dict[str, str | bool]:
        """Plan the processing pipeline for the given image."""
        # Create planning prompt
        planning_prompt = f"""
Analyze the input image at {image_path} and create a processing plan.

CRITICAL PLANNING CONSTRAINTS:
1. DO NOT execute ANY models during this planning phase. Only use analysis tools (analyze_image, check_resource_requirements, list_directory).
2. You MUST reason step-by-step before describing your plan. Walk through your thought process explicitly.

STEP-BY-STEP REASONING PROCESS:
1. First, use the analyze_image tool to understand the image characteristics.
2. Then, reason through the following questions one by one:
   a. What are the image's current resolution, quality, and characteristics?
   b. What processing transformations are needed to create a good 3DGS dataset?
   c. Should we apply super-resolution before view generation, after, or both? Why?
   d. Which specific models should we use, considering resource requirements?
   e. How many views should we generate and why?
   f. What parameters should we use for each model and why?
   g. What is the complete sequence of operations?

3. After reasoning through each question, provide a detailed plan that includes:
   - The complete processing pipeline (ordered list of operations)
   - Model choices with justifications
   - Parameter selections with rationales
   - Expected intermediate and final outputs

Remember: This is PLANNING ONLY. Do not execute any models. Save execution for the execute_plan phase.
"""

        response: AIMessage = self.agent.invoke(
            {"messages": [HumanMessage(planning_prompt)]}
        )
        return {"plan": response["messages"][-1].text, "success": True}

    def summarize_plan(self, verbose_plan: str) -> dict[str, str | bool]:
        """Summarize a verbose plan into concise, actionable execution steps.

        Args:
            verbose_plan: The detailed plan with reasoning from plan_processing

        Returns:
            Dictionary containing the summarized plan and success status
        """
        summarization_prompt = f"""
Given the following detailed processing plan, create a concise, structured summary that contains ONLY the actionable steps needed for execution.

DETAILED PLAN:
{verbose_plan}

Please provide a numbered list of execution steps that includes:
1. Which models to use (exact model names)
2. The order of operations
3. Required parameters for each model
4. Input/output file paths
5. Any resource checks needed

Remove all reasoning, explanations, and analysis. Keep only the concrete actions to execute.
Format as a clear, numbered list of steps.
"""

        response: AIMessage = self.agent.invoke(
            {"messages": [HumanMessage(summarization_prompt)]}
        )
        return {"summary": response["messages"][-1].text, "success": True}

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

        response: AIMessage = self.agent.invoke(
            {"messages": [HumanMessage(execution_prompt)]}
        )
        return {"result": response["messages"][-1].text, "success": True}
