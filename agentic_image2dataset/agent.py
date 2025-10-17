"""
LangChain agent for orchestrating the image processing pipeline.
"""

import json
from pathlib import Path

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI

from .config import LLMConfig
from .models.base import ModelRegistry
from .utils import (
    analyze_image_quality,
    detect_image_issues,
    get_optimal_view_count,
    suggest_processing_order,
)


class ImageAnalysisTool(BaseTool):
    """Tool for analyzing input images."""

    name: str = "analyze_image"
    description: str = "Analyze an image to determine its quality, characteristics, and processing needs"
    image_path: Path

    def __init__(self, image_path: Path, **kwargs: object):
        super().__init__(image_path=image_path, **kwargs)

    def _run(self, query: str = "") -> str:
        """Analyze the input image."""
        try:
            analysis = analyze_image_quality(self.image_path)
            issues = detect_image_issues(self.image_path)

            result = {
                "analysis": analysis,
                "issues": issues,
                "suggested_processing_order": suggest_processing_order(analysis),
                "optimal_view_count": get_optimal_view_count(analysis),
            }

            return json.dumps(result, indent=2)
        except Exception as e:
            return f"Error analyzing image: {str(e)}"

    async def _arun(self, query: str = "") -> str:
        return self._run(query)


class ModelExecutionTool(BaseTool):
    """Tool for executing processing models."""

    name: str = "execute_model"
    description: str = "Execute a processing model on the input image"
    model_registry: ModelRegistry
    output_dir: Path

    def __init__(
        self, model_registry: ModelRegistry, output_dir: Path, **kwargs: object
    ):
        super().__init__(model_registry=model_registry, output_dir=output_dir, **kwargs)

    def _run(self, model_name: str, parameters: str = "{}") -> str:
        """Execute a model with given parameters."""
        try:
            model = self.model_registry.get(model_name)
            if model is None:
                return f"Model '{model_name}' not found. Available models: {self.model_registry.list_available()}"

            if not model.is_available():
                return f"Model '{model_name}' is not available (missing dependencies)"

            # Parse parameters
            params: dict[str, object] = json.loads(parameters)

            # Execute the model
            input_path = (
                self.output_dir.parent / "input_image.jpg"
            )  # Assume input is saved here
            results = model.process(input_path, self.output_dir, **params)

            return f"Model '{model_name}' executed successfully. Generated {len(results)} outputs: {[str(p) for p in results]}"

        except Exception as e:
            return f"Error executing model '{model_name}': {str(e)}"

    async def _arun(self, model_name: str, parameters: str = "{}") -> str:
        return self._run(model_name, parameters)


class AgenticImageProcessor:
    """LangChain-based agent for orchestrating image processing."""

    def __init__(self, config: LLMConfig, model_registry: ModelRegistry):
        self.config: LLMConfig = config
        self.model_registry: ModelRegistry = model_registry

        # Initialize LLM
        self.llm: ChatOpenAI = ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            max_completion_tokens=config.max_tokens,
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

    def _create_agent(self) -> AgentExecutor:
        """Create the LangChain agent."""

        # Define the system prompt
        system_prompt = """You are an expert image processing agent that specializes in creating 3D Gaussian Splatting datasets from single input images.

Your task is to:
1. Analyze the input image to understand its characteristics and quality
2. Decide on the optimal processing pipeline based on the image analysis
3. Execute the necessary processing steps in the correct order
4. Ensure the final output is suitable for 3DGS training

Available processing models:
- view_generation: Generate multiple novel views from a single image using Stable Virtual Camera
- super_resolution: Enhance image quality using Real-ESRGAN

Processing decisions should consider:
- Image resolution and quality
- Scene complexity and content
- Blur, brightness, and contrast issues
- Optimal number of views for 3DGS training

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
            ModelExecutionTool(self.model_registry, output_dir),
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

        try:
            response = self.agent.invoke({"input": planning_prompt})
            return {"plan": response["output"], "success": True}
        except Exception as e:
            return {"plan": f"Planning failed: {str(e)}", "success": False}

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

        try:
            response = self.agent.invoke({"input": execution_prompt})
            return {"result": response["output"], "success": True}
        except Exception as e:
            return {"result": f"Execution failed: {str(e)}", "success": False}
