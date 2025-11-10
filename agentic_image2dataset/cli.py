"""
Command-line interface for the agentic pipeline.
"""

import argparse
import json
import sys
from pathlib import Path

from .config import LLMConfig, ModelConfig, PipelineConfig
from .pipeline import AgenticPipeline


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Agentic Image to 3DGS Dataset Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (requires GOOGLE_API_KEY environment variable)
  export GOOGLE_API_KEY=your-api-key
  python -m agentic_image2dataset.cli --input photo.jpg --output dataset/

  # With custom settings
  python -m agentic_image2dataset.cli --input photo.jpg --output dataset/ \\
    --num-views 36 --device cuda --llm-provider google --llm-model gemini-2.5-flash

  # Using OpenAI (requires OPENAI_API_KEY environment variable)
  export OPENAI_API_KEY=your-api-key
  python -m agentic_image2dataset.cli --input photo.jpg --output dataset/ \\
    --llm-provider openai --llm-model gpt-4-turbo

  # Using Anthropic (requires ANTHROPIC_API_KEY environment variable)
  export ANTHROPIC_API_KEY=your-api-key
  python -m agentic_image2dataset.cli --input photo.jpg --output dataset/ \\
    --llm-provider anthropic --llm-model claude-3-5-sonnet-20241022

  # Skip COLMAP preprocessing
  python -m agentic_image2dataset.cli --input photo.jpg --output dataset/ \\
    --skip-colmap
        """,
    )

    # Required arguments
    parser.add_argument(
        "--input", "-i", type=Path, required=True, help="Path to input image"
    )

    parser.add_argument(
        "--output", "-o", type=Path, required=True, help="Output directory for dataset"
    )

    # LLM configuration
    parser.add_argument(
        "--llm-provider",
        default="google",
        choices=["google", "openai", "anthropic"],
        help="LLM provider to use (google, openai, or anthropic)",
    )

    parser.add_argument(
        "--llm-model",
        default="gemini-2.5-flash",
        help="LLM model to use for planning (e.g., gemini-2.5-flash, gemini-pro, gpt-4, gpt-4-turbo, claude-3-opus, claude-3-5-sonnet)",
    )

    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=0.1,
        help="LLM temperature for planning",
    )

    # Model configuration
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for processing",
    )

    parser.add_argument(
        "--num-views",
        type=int,
        help="Number of views to generate (if not specified, agent decides)",
    )

    parser.add_argument(
        "--super-resolution-factor",
        type=int,
        default=4,
        help="Super-resolution scale factor",
    )

    parser.add_argument(
        "--colmap-quality",
        default="high",
        choices=["high", "medium", "low"],
        help="COLMAP reconstruction quality",
    )

    # Processing options
    parser.add_argument(
        "--skip-colmap", action="store_true", help="Skip COLMAP preprocessing"
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    # Utility options
    parser.add_argument(
        "--list-models", action="store_true", help="List available models and exit"
    )

    parser.add_argument("--model-info", help="Get information about a specific model")

    args = parser.parse_args()

    # Handle utility commands
    if args.list_models:
        _list_models()
        return

    if args.model_info:
        _show_model_info(args.model_info)
        return

    # Validate input
    if not args.input.exists():
        print(f"Error: Input image not found: {args.input}")
        sys.exit(1)

    # Create configuration
    config = _create_config(args)

    # Run pipeline
    try:
        pipeline = AgenticPipeline(config)

        print(f"Processing image: {args.input}")
        print(f"Output directory: {args.output}")
        print(f"Device: {config.model.device}")
        print(f"LLM Provider: {config.llm.provider}")
        print(f"LLM Model: {config.llm.model_name}")

        result = pipeline.process(
            input_image=args.input,
            output_dir=args.output,
            num_views=args.num_views,
            skip_colmap=args.skip_colmap,
        )

        if result["success"]:
            print("\n✅ Processing completed successfully!")
            print(f"Generated {result['generated_images']} images")
            print(f"Output directory: {result['output_dir']}")

            if result.get("issues"):
                issues = result["issues"]
                if isinstance(issues, list):
                    issues_str = ", ".join(str(issue) for issue in issues)
                else:
                    issues_str = str(issues)
                print(f"\n⚠️  Detected issues: {issues_str}")
        else:
            print(f"\n❌ Processing failed: {result['error']}")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


def _create_config(args) -> PipelineConfig:
    """Create configuration from command line arguments."""
    # Create LLM config (API keys must be set via environment variables)
    llm_config = LLMConfig(
        provider=args.llm_provider,
        model_name=args.llm_model,
        temperature=args.llm_temperature,
    )

    model_config = ModelConfig(
        device=args.device,
        num_views=args.num_views or 24,
        super_resolution_factor=args.super_resolution_factor,
        colmap_quality=args.colmap_quality,
    )

    return PipelineConfig(
        llm=llm_config,
        model=model_config,
        output_dir=args.output,
        input_image=args.input,
        skip_colmap=args.skip_colmap,
        verbose=args.verbose,
    )


def _list_models():
    """List available models."""
    # Create a minimal config for model initialization
    config = PipelineConfig(
        llm=LLMConfig(),
        model=ModelConfig(),
        output_dir=Path("."),
        input_image=Path("."),
        skip_colmap=True,
        verbose=False,
    )

    pipeline = AgenticPipeline(config)
    available_models = pipeline.get_available_models()

    print("Available models:")
    for model_name in available_models:
        model_info = pipeline.get_model_info(model_name)
        if model_info:
            status = "✅ Available" if model_info["available"] else "❌ Not available"
            print(f"  {model_name}: {status}")
            print(f"    Description: {model_info['description']}")


def _show_model_info(model_name: str):
    """Show detailed information about a model."""
    config = PipelineConfig(
        llm=LLMConfig(),
        model=ModelConfig(),
        output_dir=Path("."),
        input_image=Path("."),
        skip_colmap=True,
        verbose=False,
    )

    pipeline = AgenticPipeline(config)
    model_info = pipeline.get_model_info(model_name)

    if model_info:
        print(f"Model: {model_name}")
        print(f"Description: {model_info['description']}")
        print(f"Available: {'Yes' if model_info['available'] else 'No'}")
        print(f"Requirements: {json.dumps(model_info['requirements'], indent=2)}")
    else:
        print(f"Model '{model_name}' not found")


if __name__ == "__main__":
    main()
