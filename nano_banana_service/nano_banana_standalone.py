#!/usr/bin/env python
"""
Standalone Nano Banana preprocessing script.
Run this in a separate venv with google-genai installed.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

def preprocess_image(input_path: str, output_path: str, api_key: str) -> bool:
    """Preprocess an image using Nano Banana (Gemini 3 Pro Image)."""
    print(f"DEBUG: Starting preprocess_image with {input_path}")
    try:
        client = genai.Client(api_key=api_key)
        
        # Prepare Inline Image Data
        print(f"DEBUG: Reading file {input_path}...")
        try:
            path_obj = Path(input_path)
            image_bytes = path_obj.read_bytes()
            # Simple mime type detection
            mime_type = "image/png" if path_obj.suffix.lower() == ".png" else "image/jpeg"
            print(f"DEBUG: Read {len(image_bytes)} bytes, mime_type={mime_type}")
            image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        except Exception as e:
            print(f"DEBUG: Error reading file: {e}")
            return False
        
        # Generate/Edit
        prompt = "Isolate this object on a purely solid white background. Sharpen details. High quality texture."
        print("DEBUG: Sending request to Gemini...")
        
        try:
            response = client.models.generate_content(
                model='gemini-3-pro-image-preview', # Using flash-exp for testing, change to gemini-3-pro-image-preview if needed
                contents=[prompt, image_part]
            )
            print(f"DEBUG: Response received. Candidates: {len(response.candidates) if response.candidates else 0}")
        except Exception as e:
            print(f"DEBUG: Error generating content: {e}")
            return False
        
        # Save result
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if part.inline_data:
                    print(f"DEBUG: Saving inline data to {output_path}...")
                    with open(output_path, "wb") as f:
                        f.write(part.inline_data.data)
                    return True
        
        print(f"DEBUG: No inline data found. Response: {response}")
        return False
        
    except Exception as e:
        print(f"DEBUG: Unexpected error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python nano_banana_standalone.py <input_image> <output_image>")
        sys.exit(1)
    
    load_dotenv()
    api_key = os.getenv("NANO_API_KEY")
    if not api_key:
        print("Error: NANO_API_KEY not found in environment")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    success = preprocess_image(input_path, output_path, api_key)
    sys.exit(0 if success else 1)
