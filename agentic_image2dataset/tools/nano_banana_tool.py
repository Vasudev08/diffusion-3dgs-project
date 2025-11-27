from pathlib import Path
from typing import Type
import subprocess

from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import os

class NanoBananaToolInput(BaseModel):
    image_path: str = Field(description="Path to the input image file to be processed")

class NanoBananaTool(BaseTool):
    name: str = "nano_banana_preprocess"
    description: str = "Uses Nano Banana Pro (Gemini 3 Pro Image) to preprocess an image for 3D Splatting. It isolates the object on a white background, sharpens details, and ensures a 1:1 aspect ratio."
    args_schema: Type[BaseModel] = NanoBananaToolInput
    workspace_root: Path = Field(default_factory=Path.cwd)

    def _run(self, image_path: str) -> str:
        """Use the tool by calling standalone script in separate venv."""
        try:
            # 1. Check API Key
            if "NANO_API_KEY" not in os.environ:
                return "Error: NANO_API_KEY environment variable not found."

            # 2. Resolve paths
            input_path = self.workspace_root / image_path
            if not input_path.exists():
                return f"Error: Input file not found at {input_path}"
            
            output_filename = f"processed_{input_path.stem}.png"
            output_path = self.workspace_root / output_filename
            
            # 3. Find the standalone script and venv in sub-project
            project_root = Path(__file__).parent.parent.parent
            nano_service_dir = project_root / "nano_banana_service"
            standalone_script = nano_service_dir / "nano_banana_standalone.py"
            nano_venv_python = nano_service_dir / ".venv" / "bin" / "python"
            
            if not standalone_script.exists():
                return f"Error: Standalone script not found at {standalone_script}"
            
            if not nano_venv_python.exists():
                return f"Error: Nano Banana venv not found. Run: cd nano_banana_service && uv sync"
            
            # 4. Call the standalone script in the separate venv
            print(f"Calling Nano Banana in separate venv...")
            result = subprocess.run(
                [str(nano_venv_python), str(standalone_script), str(input_path), str(output_path)],
                capture_output=True,
                text=True,
                env=os.environ.copy()
            )
            
            if result.returncode == 0:
                return f"Successfully processed image. Saved to: {output_filename}"
            else:
                return f"Error: Nano Banana script failed:\n{result.stdout}\n{result.stderr}"

        except Exception as e:
            return f"Error processing image with Nano Banana: {str(e)}"

    def _arun(self, image_path: str):
        raise NotImplementedError("This tool does not support async")
