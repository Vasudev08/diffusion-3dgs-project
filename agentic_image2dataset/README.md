# Agentic Image to 3DGS Dataset Pipeline

A LangChain-based agentic pipeline for processing single input images into 3DGS-ready datasets using Stable Virtual Camera, DiffBIR, and PyCOLMAP.

## Features

- **Intelligent Processing**: LLM-based planning that analyzes image characteristics and decides optimal processing order
- **Extensible Architecture**: Easy to add new processing models
- **Automated COLMAP**: Automatic PyCOLMAP preprocessing for 3DGS training
- **Quality Assessment**: Comprehensive image analysis for informed decision making
- **Standard Output**: COLMAP-format compatible with gsplat examples

## Quick Start

### Installation

```bash
# Install dependencies
pip install -e .

# Set up API keys (choose one)
export OPENAI_API_KEY="your-openai-key"      # For GPT models
export ANTHROPIC_API_KEY="your-anthropic-key"  # For Claude models
```

### DiffBIR Setup

DiffBIR requires additional dependencies and model downloads:

```bash
# Install DiffBIR dependencies (included in project requirements)
pip install omegaconf accelerate xformers timm

# DiffBIR models will be automatically downloaded on first use
# For manual download, see DiffBIR/README.md
```

### Basic Usage

```bash
# Process a single image
python -m agentic_image2dataset.cli --input photo.jpg --output dataset/

# With custom settings
python -m agentic_image2dataset.cli --input photo.jpg --output dataset/ \
  --num-views 36 --device cuda --llm-model gpt-4
```

### Python API

```python
from agentic_image2dataset import AgenticPipeline, PipelineConfig, LLMConfig, ModelConfig

# Create configuration
config = PipelineConfig(
    llm=LLMConfig(model_name="gpt-4"),
    model=ModelConfig(device="cuda", num_views=24),
    output_dir=Path("dataset/"),
    input_image=Path("photo.jpg")
)

# Run pipeline
pipeline = AgenticPipeline(config)
result = pipeline.process(
    input_image="photo.jpg",
    output_dir="dataset/",
    num_views=24
)

print(f"Generated {result['generated_images']} images")
```

## Architecture

### Core Components

- **`AgenticPipeline`**: Main orchestrator that coordinates all components
- **`AgenticImageProcessor`**: LangChain agent for intelligent planning
- **`ModelRegistry`**: Manages available processing models
- **`COLMAPProcessor`**: Automated PyCOLMAP preprocessing

### Processing Models

- **Stable Virtual Camera**: Generates novel views from single input
- **DiffBIR**: Advanced super-resolution enhancement using diffusion models
  - Superior quality compared to traditional methods like Real-ESRGAN
  - Better handling of complex textures and details
  - Multiple model versions (v2.1 recommended)
- **Extensible**: Easy to add new models by inheriting `BaseProcessingModel`

### Workflow

1. **Image Analysis**: Comprehensive quality and characteristic assessment
2. **LLM Planning**: Agent decides optimal processing order and parameters
3. **Model Execution**: Runs selected models in optimal sequence
4. **COLMAP Processing**: Automated 3D reconstruction
5. **Dataset Creation**: Standard COLMAP format output

## Configuration

### LLM Configuration

```python
llm_config = LLMConfig(
    model_name="gpt-4",           # or "gpt-3.5-turbo", "claude-3-opus"
    api_key="your-api-key",       # or set via environment variable
    temperature=0.1,              # Lower for more consistent planning
    max_tokens=2000
)
```

### Model Configuration

```python
model_config = ModelConfig(
    device="cuda",                # or "cpu"
    num_views=24,                 # Number of views to generate
    super_resolution_factor=4,    # SR scale factor
    colmap_quality="high",        # "high", "medium", "low"
    view_generation_trajectory="orbit"  # "orbit", "spiral", "arc"
)
```

## CLI Options

```bash
python -m agentic_image2dataset.cli --help

# Required
--input, -i          Path to input image
--output, -o         Output directory

# LLM Configuration
--llm-model          LLM model (gpt-4, gpt-3.5-turbo, claude-3-opus)
--llm-api-key        API key (or set via environment)
--llm-temperature    LLM temperature (default: 0.1)

# Model Configuration
--device             Device (cuda/cpu, default: cuda)
--num-views          Number of views to generate
--super-resolution-factor  SR scale factor (default: 4)
--colmap-quality     COLMAP quality (high/medium/low)

# Processing Options
--skip-colmap        Skip COLMAP preprocessing
--verbose, -v        Enable verbose output

# Utility
--list-models        List available models
--model-info MODEL   Get model information
```

## Output Structure

```
dataset/
├── images/              # Generated images
│   ├── view_000.png
│   ├── view_001.png
│   └── ...
├── sparse/              # COLMAP reconstruction
│   └── 0/
│       ├── cameras.bin
│       ├── images.bin
│       └── points3D.bin
└── metadata.json        # Dataset metadata
```

## Integration with gsplat

The output is compatible with gsplat training:

```bash
# Train 3DGS model
python gsplat/examples/simple_trainer.py --data_path dataset/
```

## Adding New Models

To add a new processing model:

```python
from agentic_image2dataset.models.base import BaseProcessingModel

class MyCustomModel(BaseProcessingModel):
    def analyze(self, image_path):
        # Analyze image characteristics
        return {"custom_analysis": "..."}

    def process(self, image_path, output_dir, **kwargs):
        # Process the image
        return [output_path]

    def get_description(self):
        return "My custom processing model"

    def get_requirements(self):
        return {"dependencies": ["my_package"]}

# Register the model
pipeline.model_registry.register("my_model", MyCustomModel())
```

## Troubleshooting

### Common Issues

1. **Missing API Keys**: Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` environment variables
2. **CUDA Out of Memory**: Use `--device cpu` or reduce `--num-views`
3. **DiffBIR Model Loading**: Ensure DiffBIR dependencies are installed and models are downloaded
4. **COLMAP Fails**: Try `--skip-colmap` or lower `--colmap-quality`
5. **Model Not Available**: Check dependencies with `--list-models`

### Debug Mode

```bash
python -m agentic_image2dataset.cli --input photo.jpg --output dataset/ --verbose
```

## Dependencies

- **LangChain**: Agent orchestration
- **Stable Virtual Camera**: Novel view generation
- **DiffBIR**: Advanced super-resolution using diffusion models
- **PyCOLMAP**: 3D reconstruction
- **OpenCV**: Image processing
- **PyTorch**: Deep learning framework

## License

This project is part of the diffusion-3dgs-project and follows the same license terms.
