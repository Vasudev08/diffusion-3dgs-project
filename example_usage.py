#!/usr/bin/env python3
"""
Example usage of the agentic image processing pipeline.

Set the GOOGLE_API_KEY environment variable to your Google API key.
```bash
export GOOGLE_API_KEY=your-api-key
```
or load the .env file in the root directory.
"""

from pathlib import Path

from dotenv import load_dotenv

from agentic_image2dataset import (
    AgenticPipeline,
    LLMConfig,
    ModelConfig,
    PipelineConfig,
)

# Set up API key GOOGLE_API_KEY environment variable from .env file
load_dotenv()


def main():
    """Example usage of the agentic pipeline."""

    # Create configuration
    config = PipelineConfig(
        llm=LLMConfig(model_name="gemini-2.5-flash", temperature=0.1),
        model=ModelConfig(
            device="cuda",
            num_views=24,
            super_resolution_factor=4,
        ),
        output_dir=Path("example_output"),
        input_image=Path("example_input.png"),
        verbose=True,
    )

    # Initialize pipeline
    print("Initializing agentic pipeline...")
    pipeline = AgenticPipeline(config)

    # Check available models
    available_models = pipeline.get_available_models()
    print(f"Available models: {available_models}")

    # Process the image
    print("Processing image...")
    result = pipeline.process(
        input_image=config.input_image,
        output_dir=config.output_dir,
    )

    if result["success"]:
        print("✅ Processing completed successfully!")
        print(f"Output directory: {result['output_dir']}")
    else:
        print(f"❌ Processing failed: {result['error']}")


if __name__ == "__main__":
    main()
