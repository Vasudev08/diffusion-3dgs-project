#!/usr/bin/env python3
"""
Example usage of the agentic image processing pipeline.
"""

import os
from pathlib import Path

# Set up API key (replace with your actual key)
os.environ["GOOGLE_API_KEY"] = "your-api-key"


from agentic_image2dataset import (
    AgenticPipeline,
    LLMConfig,
    ModelConfig,
    PipelineConfig,
)


def main():
    """Example usage of the agentic pipeline."""

    # Create configuration
    config = PipelineConfig(
        llm=LLMConfig(model_name="gemini-2.5-flash", temperature=0.1),
        model=ModelConfig(
            device="cuda",
            num_views=24,
            super_resolution_factor=4,
            colmap_quality="high",
        ),
        output_dir=Path("example_output"),
        input_image=Path("example_input.png"),
        skip_colmap=False,
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
        num_views=config.model.num_views,
    )

    if result["success"]:
        print("✅ Processing completed successfully!")
        print(f"Generated {result['generated_images']} images")
        print(f"Output directory: {result['output_dir']}")

        if result.get("issues"):
            print(f"Issues detected: {', '.join(result['issues'])}")
    else:
        print(f"❌ Processing failed: {result['error']}")


if __name__ == "__main__":
    main()
