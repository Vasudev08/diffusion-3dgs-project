"""
Configuration classes for the agentic pipeline.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class LLMConfig:
    """
    Configuration for LLM agent.

    API keys must be set via environment variables:
    - Google: GOOGLE_API_KEY
    - OpenAI: OPENAI_API_KEY
    - Anthropic: ANTHROPIC_API_KEY

    All LangChain models automatically read from these environment variables.
    """

    provider: str = "google"  # google, openai, anthropic
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.1
    max_tokens: int = 2000


@dataclass
class ModelConfig:
    """Configuration for processing models."""

    device: str = "cuda"
    batch_size: int = 1
    num_views: int = 24
    super_resolution_factor: int = 4
    view_generation_trajectory: str = "orbit"  # orbit, spiral, arc
    colmap_quality: str = "high"  # high, medium, low


@dataclass
class PipelineConfig:
    """Main pipeline configuration."""

    llm: LLMConfig
    model: ModelConfig
    output_dir: Path
    input_image: Path
    skip_colmap: bool = False
    verbose: bool = True

    @classmethod
    def from_dict(cls, config_dict: dict[str, object]) -> "PipelineConfig":
        """Create config from dictionary."""
        llm_dict = config_dict.get("llm", {})
        model_dict = config_dict.get("model", {})

        # Type assertions for better type safety
        assert isinstance(llm_dict, dict)
        assert isinstance(model_dict, dict)

        llm_config = LLMConfig(**llm_dict)
        model_config = ModelConfig(**model_dict)

        return cls(
            llm=llm_config,
            model=model_config,
            output_dir=Path(str(config_dict["output_dir"])),
            input_image=Path(str(config_dict["input_image"])),
            skip_colmap=bool(config_dict.get("skip_colmap", False)),
            verbose=bool(config_dict.get("verbose", True)),
        )
