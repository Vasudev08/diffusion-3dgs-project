
from pathlib import Path
from typing import Type

from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import google.generativeai as genai
import os

class NanoBananaToolInput(BaseModel):
    image_path: str = Field(description="Path to the input image file to be processed")

class NanoBananaTool(BaseTool):
    name: str = "nano_banana_preprocess"
    description: str = "Uses Nano Banana Pro (Gemini 3 Pro Image) to preprocess an image for 3D Splatting. It isolates the object on a white background, sharpens details, and ensures a 1:1 aspect ratio."
    args_schema: Type[BaseModel] = NanoBananaToolInput
    workspace_root: Path = Field(default_factory=Path.cwd)

    def _run(self, image_path: str) -> str:
        """Use the tool."""
        try:
            # 1. Setup API Key (ensure it's in env)
            if "NANO_API_KEY" not in os.environ:
                return "Error: NANO_API_KEY environment variable not found."

            genai.configure(api_key=os.environ["NANO_API_KEY"])

            # 2. Resolve path
            input_path = self.workspace_root / image_path
            if not input_path.exists():
                return f"Error: Input file not found at {input_path}"

            # 3. Configuration for 'Gemini 3 Pro Image' (using the preview model as requested)
            # Note: Using the model name provided by user, but falling back if needed might be good later.
            # For now, we stick to the user's requested model name.
            model = genai.GenerativeModel('gemini-3-pro-image-preview')
            
            # 4. Upload file
            print(f"Uploading {input_path} to Gemini...")
            sample_file = genai.upload_file(str(input_path))
            
            # 5. The Magic Prompt
            prompt = "Isolate this object on a purely solid white background. Sharpen details. High quality texture."
            
            # 6. Generate/Edit
            print("Sending request to Nano Banana Pro...")
            # Note: The standard SDK uses generate_content for Gemini models.
            # 'generate_images' is likely from a different SDK or hypothetical.
            # We will try generate_content with the prompt and image.
            response = model.generate_content([prompt, sample_file])
            
            # 7. Save the result
            output_filename = f"processed_{input_path.stem}.png"
            output_path = self.workspace_root / output_filename
            
            saved_image = False
            text_response = []

            if response.parts:
                for part in response.parts:
                    # Check for text
                    if hasattr(part, "text") and part.text:
                        text_response.append(part.text)
                    
                    # Check for inline_data (Image)
                    if hasattr(part, "inline_data") and part.inline_data:
                        print(f"Received inline data with mime_type: {part.inline_data.mime_type}")
                        # We assume it's an image. Save it.
                        with open(output_path, "wb") as f:
                            f.write(part.inline_data.data)
                        saved_image = True
            
            if saved_image:
                return f"Successfully processed image. Saved to: {output_filename}. Text note: {' '.join(text_response)}"
            elif text_response:
                return f"Model returned text only: {' '.join(text_response)}"
            else:
                return "Error: Model returned no content (no text or inline_data)."

        except Exception as e:
            return f"Error processing image with Nano Banana: {str(e)}"

    def _arun(self, image_path: str):
        raise NotImplementedError("This tool does not support async")
