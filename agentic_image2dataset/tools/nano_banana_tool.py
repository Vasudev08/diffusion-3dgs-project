from pathlib import Path
from typing import Type
import requests
import json
import base64
import os

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
        """Use the tool via direct REST API (No SDK required)."""
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
            
            print(f"Processing {input_path.name} with Nano Banana (REST API)...")

            # 3. Prepare API Request
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro-image-preview:generateContent?key={api_key}"
            
            # Read and Encode Image
            with open(input_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            
            # Construct JSON Payload
            payload = {
                "contents": [{
                    "parts": [
                        {"text": "Isolate this object on a purely solid white background. Sharpen details. High quality texture."},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg", # Assuming JPEG/PNG input, API is flexible
                                "data": image_data
                            }
                        }
                    ]
                }]
            }
            
            # 4. Send Request
            headers = {'Content-Type': 'application/json'}
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            
            if response.status_code != 200:
                return f"Error from Gemini API: {response.text}"
                
            # 5. Parse Response (Extract Image)
            try:
                result = response.json()
                candidates = result.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    for part in parts:
                        if "inline_data" in part:
                            b64_data = part["inline_data"]["data"]
                            with open(output_path, "wb") as f:
                                f.write(base64.b64decode(b64_data))
                            return f"Successfully processed image. Saved to: {output_filename}"
                            
                # Fallback if no image returned
                return f"Model returned no image. Full response: {result}"
                
            except Exception as e:
                return f"Error parsing API response: {str(e)}"

        except Exception as e:
            return f"Error processing image with Nano Banana: {str(e)}"

    def _arun(self, image_path: str):
        raise NotImplementedError("This tool does not support async")
