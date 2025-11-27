# Setup Script for Dual Environment

This project uses two separate virtual environments to avoid dependency conflicts:

## Setup Instructions

### 1. Main Project Environment
```bash
# In project root
uv sync --extra google
```

### 2. Nano Banana Service Environment
```bash
# In nano_banana_service subdirectory
cd nano_banana_service
uv sync
cd ..
```

### 3. Run the Pipeline
```bash
# Back in project root
uv run colab_pipeline.py --input_image Mango.jpg
```

## Structure
```
diffusion-3dgs-project/
├── .venv/                    # Main environment (LangChain + pipeline)
├── nano_banana_service/
│   ├── .venv/               # Nano Banana environment (google-generativeai)
│   ├── pyproject.toml
│   └── nano_banana_standalone.py
```

The pipeline automatically calls the Nano Banana service in its separate environment via subprocess!
