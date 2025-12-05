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
    try:
        client = genai.Client(api_key=api_key)
        
        # Upload file
        print(f"Uploading {input_path}...")
        try:
            file_ref = client.files.upload(file=str(input_path))
        except Exception as e:
            print(f"Error uploading file: {e}")
            return False
        
        # Generate/Edit
        prompt = "Isolate this object on a purely solid white background. Sharpen details. High quality texture."
        print("Processing with Nano Banana Pro...")
        
        try:
            response = client.models.generate_content(
                model='gemini-3-pro-image-preview',
                contents=[prompt, file_ref]
            )
        except Exception as e:
            print(f"Error generating content: {e}")
            return False
        
        # Save result
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if part.inline_data:
                    print(f"Saving to {output_path}...")
                    with open(output_path, "wb") as f:
                        f.write(part.inline_data.data)
                    return True
        
        print("Error: No image data in response")
        return False
        
    except Exception as e:
        print(f"Error: {e}")
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
