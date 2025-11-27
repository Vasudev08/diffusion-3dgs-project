#!/usr/bin/env python
"""
Standalone Nano Banana preprocessing script.
Run this in a separate venv with google-generativeai installed.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

def preprocess_image(input_path: str, output_path: str, api_key: str) -> bool:
    """Preprocess an image using Nano Banana (Gemini 3 Pro Image)."""
    try:
        genai.configure(api_key=api_key)
        
        # Upload file
        print(f"Uploading {input_path}...")
        sample_file = genai.upload_file(str(input_path))
        
        # Create model
        model = genai.GenerativeModel('gemini-3-pro-image-preview')
        
        # Generate/Edit
        prompt = "Isolate this object on a purely solid white background. Sharpen details. High quality texture."
        print("Processing with Nano Banana Pro...")
        response = model.generate_content([prompt, sample_file])
        
        # Save result
        if response.parts:
            for part in response.parts:
                if hasattr(part, "inline_data") and part.inline_data:
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
