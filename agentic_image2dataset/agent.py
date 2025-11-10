"""
LangChain agent for orchestrating the image processing pipeline.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from .config import LLMConfig
from .models.base import ModelRegistry
from .utils import (
    analyze_image_quality,
    detect_image_issues,
    get_optimal_view_count,
    suggest_processing_order,
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
    general_guidance: Optional[str] = None
    model_guidance: List[ModelGuidance] = field(default_factory=list)
    shared_notes: Optional[str] = None


class ImageAnalysisTool(BaseTool):
    """Tool for analyzing input images."""

    name: str = "analyze_image"
    description: str = "Analyze an image to determine its quality, characteristics, and processing needs"
    image_path: Path

    def __init__(self, image_path: Path, **kwargs: object):
        super().__init__(image_path=image_path, **kwargs)

    def _run(self, query: str = "") -> str:
        """Analyze the input image."""
        analysis = analyze_image_quality(self.image_path)
        issues = detect_image_issues(self.image_path)

        result = {
            "analysis": analysis,
            "issues": issues,
            "suggested_processing_order": suggest_processing_order(analysis),
            "optimal_view_count": get_optimal_view_count(analysis),
        }

        return json.dumps(result, indent=2)


class ModelExecutionToolArgs(BaseModel):
    """Arguments for the model execution tool."""

    model_name: str = Field(description="The name of the model to execute")
    parameters: str = Field(
        default="{}",
        description='Model parameters in JSON format. Must be a valid JSON string. Example: \'{"scale": 4, "batch_size": 1}\'',
    )


class ModelExecutionTool(BaseTool):
    """Tool for executing processing models."""

    name: str = "execute_model"
    description: str = "Execute a processing model on the input image. The 'parameters' argument must be provided as a JSON string."
    args_schema = ModelExecutionToolArgs
    model_registry: ModelRegistry
    input_path: Path
    output_dir: Path

    def __init__(
        self,
        model_registry: ModelRegistry,
        input_path: Path,
        output_dir: Path,
        **kwargs: object,
    ):
        super().__init__(
            model_registry=model_registry,
            input_path=input_path,
            output_dir=output_dir,
            **kwargs,
        )

    def _run(self, model_name: str, parameters: str = "{}") -> str:
        """Execute a model with given parameters."""
        model = self.model_registry.get(model_name)
        if model is None:
            return f"Model '{model_name}' not found. Available models: {self.model_registry.list_available()}"

        # Parse parameters
        params: dict[str, object] = json.loads(parameters)

        # Execute the model
        results = model.process(self.input_path, self.output_dir, **params)

        return f"Model '{model_name}' executed successfully. Generated {len(results)} outputs: {[str(p) for p in results]}"


class AgenticImageProcessor:
    """LangChain-based agent for orchestrating image processing."""

    def __init__(self, config: LLMConfig, model_registry: ModelRegistry):
        self.config: LLMConfig = config
        self.model_registry: ModelRegistry = model_registry

        # Initialize LLM
        self.llm: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
            api_key=config.api_key,
        )

        # Create tools
        self.tools: list[BaseTool] = []

        # Create memory
        self.memory: ConversationBufferWindowMemory = ConversationBufferWindowMemory(
            memory_key="chat_history", return_messages=True, k=10
        )

        # Create agent
        self.agent: AgentExecutor = self._create_agent()

    def _get_role_configurations(self) -> Dict[str, RoleConfiguration]:
        """Get structured configurations for each role."""
        return {
            "view_generation": RoleConfiguration(
                display_name="View Generation Models",
                general_guidance=None,
                model_guidance=[],
            ),
            "super_resolution": RoleConfiguration(
                display_name="Super-Resolution Models",
                general_guidance="When choosing between super-resolution models:",
                model_guidance=[
                    ModelGuidance(
                        model_name="diffbir",
                        guidance_text="Use 'diffbir' for maximum quality when computational resources allow",
                    ),
                    ModelGuidance(
                        model_name="adcsr",
                        guidance_text="Use 'adcsr' for faster processing with competitive quality",
                    ),
                ],
                shared_notes="Both models support 4x upscaling by default",
            ),
        }

    def _build_model_descriptions(self, roles: Dict[str, List[str]]) -> str:
        """Build model descriptions from role configurations."""
        role_configs = self._get_role_configurations()
        model_descriptions = []

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
                    model_descriptions.append(
                        f"  - {model_name}: {model.get_description()}"
                    )

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

    def add_tools(self, image_path: Path, output_dir: Path):
        """Add tools to the agent."""
        self.tools = [
            ImageAnalysisTool(image_path),
            ModelExecutionTool(self.model_registry, image_path, output_dir),
        ]

        # Recreate agent with new tools
        self.agent = self._create_agent()

    def plan_processing(
        self, image_path: Path, output_dir: Path
    ) -> dict[str, str | bool]:
        """Plan the processing pipeline for the given image."""
        self.add_tools(image_path, output_dir)

        # Create planning prompt
        planning_prompt = f"""
Analyze the input image at {image_path} and create a processing plan.

First, use the analyze_image tool to understand the image characteristics.
Then, based on the analysis, decide which models to use and in what order.

Consider:
1. Should we apply super-resolution first or after view generation?
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
