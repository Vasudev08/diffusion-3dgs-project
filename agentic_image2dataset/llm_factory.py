"""
Factory for creating LLM instances.
"""

from langchain_core.language_models import BaseChatModel

from agentic_image2dataset.config import LLMConfig


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
