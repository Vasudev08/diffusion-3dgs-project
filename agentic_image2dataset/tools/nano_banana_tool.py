from pathlib import Path
from typing import Type
import os

from google import genai
from google.genai import types

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

class NanoBananaToolInput(BaseModel):
    image_path: str = Field(description="Path to the input image file to be processed")

class NanoBananaTool(BaseTool):
    name: str = "nano_banana_preprocess"
    description: str = "Uses Nano Banana Pro (Gemini 3 Pro Image) to preprocess an image for 3D Splatting. It isolates the object on a white background, sharpens details, and ensures a 1:1 aspect ratio."
    args_schema: Type[BaseModel] = NanoBananaToolInput
    workspace_root: Path = Field(default_factory=Path.cwd)

    def _run(self, image_path: str) -> str:
        """Use the tool via google-genai SDK with inline image data."""
        try:
            # 1. Check API Key
            api_key = os.environ.get("NANO_API_KEY")
            if not api_key:
                return "Error: NANO_API_KEY environment variable not found."

            # 2. Resolve paths
            input_path = self.workspace_root / image_path
            if not input_path.exists():
                return f"Error: Input file not found at {input_path}"
            
            output_filename = f"processed_{input_path.stem}.png"
            output_path = self.workspace_root / output_filename
            
            print(f"Processing {input_path.name} with Nano Banana (SDK/Inline)...")

            # 3. Initialize Client
            client = genai.Client(api_key=api_key)
            
            # 4. Prepare Inline Image Data
            try:
                image_bytes = input_path.read_bytes()
                # Simple mime type detection based on extension, default to jpeg
                mime_type = "image/png" if input_path.suffix.lower() == ".png" else "image/jpeg"
                image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            except Exception as e:
                return f"Error reading image file: {e}"

            # 5. Generate Content
            prompt = "Isolate this object on a purely solid white background. Sharpen details. High quality texture."
            
            try:
                response = client.models.generate_content(
                    model='gemini-3-pro-image-preview',
                    contents=[prompt, image_part]
                )
            except Exception as e:
                return f"Error generating content: {e}"
            
            # 6. Save Result
            if response.candidates:
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        with open(output_path, "wb") as f:
                            f.write(part.inline_data.data)
                        return f"Successfully processed image. Saved to: {output_filename}"
            
            return f"Model returned no image. Response: {response}"

        except Exception as e:
            return f"Error processing image with Nano Banana: {str(e)}"

    def _arun(self, image_path: str):
        raise NotImplementedError("This tool does not support async")
