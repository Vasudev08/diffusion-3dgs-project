# colab_pipeline.py
# Instructions for Google Colab:
# 1. Copy this script into a cell or upload it to your Colab environment.
# 2. Before running this script, you MUST run the following installation commands in a separate cell:
#
# !pip install -q nerfstudio gsplat==1.4.0 tinycudann
# !pip install -q google-generativeai langchain-google-genai langchain
# !pip install -q tyro diffusers transformers accelerate
# !git clone https://github.com/Vasudev08/diffusion-3dgs-project.git
# %cd diffusion-3dgs-project
# !pip install -e .
#
# 3. Make sure you have set your GOOGLE_API_KEY in the Colab secrets or environment.

import os
import sys
import shutil
import subprocess
from pathlib import Path

try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
    print("Warning: Not running in Google Colab. Some features (upload/download) will be disabled.")

# Add project root to python path
sys.path.append(str(Path(__file__).parent))

from agentic_image2dataset import (
    AgenticPipeline,
    PipelineConfig,
    LLMConfig,
    ModelConfig,
)

def get_input_image_colab():
    """Upload image using Colab's file upload."""
    if not IN_COLAB:
        return Path("example_input.png") # Fallback
    
    print("\nPlease upload your input image:")
    uploaded = files.upload()
    
    if not uploaded:
        print("No file uploaded.")
        return None
        
    # Get the first uploaded filename
    filename = next(iter(uploaded))
    return Path(filename)

def zip_and_download(directory: Path):
    """Zip the output directory and trigger download."""
    if not IN_COLAB:
        print(f"Output available at: {directory}")
        return

    print(f"\nZipping output directory: {directory}")
    shutil.make_archive(str(directory), 'zip', directory)
    zip_path = f"{directory}.zip"
    
    print(f"Downloading {zip_path}...")
    files.download(zip_path)

def main():
    print("=== Google Colab Agentic 3DGS Pipeline ===")
    
    # 1. Setup API Key
    if "GOOGLE_API_KEY" not in os.environ:
        print("\nGoogle API Key not found. Please enter it below:")
        try:
            from google.colab import userdata
            os.environ["GOOGLE_API_KEY"] = userdata.get('GOOGLE_API_KEY')
        except:
            api_key = input("Enter GOOGLE_API_KEY: ").strip()
            if api_key:
                os.environ["GOOGLE_API_KEY"] = api_key
            else:
                print("Error: API Key required.")
                return

    # 2. Get Input
    input_image = get_input_image_colab()
    if not input_image:
        return
    
    # 3. Configure Pipeline
    output_dir = Path("output_dataset_colab")
    
    # Colab usually has a T4 GPU, so we can use decent settings
    print("\nConfiguration:")
    print("Using Standard Quality settings for Colab T4...")
    
    config = PipelineConfig(
        llm=LLMConfig(model_name="gemini-2.5-flash", temperature=0.1),
        model=ModelConfig(
            device="cuda", # Assume CUDA in Colab
            num_views=32,  # Good balance
            super_resolution_factor=2, # Save some VRAM
        ),
        output_dir=output_dir,
        input_image=input_image,
        verbose=True,
    )
    
    # Manually add skip_colmap
    config.skip_colmap = True

    # 4. Initialize pipeline
    print("\nInitializing agentic pipeline...")
    pipeline = AgenticPipeline(config)
    
    # 5. Process the image
    print(f"Processing image: {input_image}...")
    result = pipeline.process(
        input_image=config.input_image,
        output_dir=config.output_dir,
        num_views=config.model.num_views,
        skip_colmap=config.skip_colmap,
    )

    if result["success"]:
        print("\nProcessing completed successfully!")
        
        # Run Splatfacto Training
        if getattr(config, 'skip_colmap', False):
            print("\n[Extra] Running Splatfacto training (since COLMAP was skipped)...")
            try:
                dataset_dir = result['output_dir']
                results_dir = dataset_dir / "results"
                exports_dir = dataset_dir / "exports"
                
                print(f"Training Splatfacto -> {results_dir}")
                
                # 1. Train Splatfacto
                # Reduced iterations for Colab speed, but enough for quality
                train_cmd = [
                    "ns-train", "splatfacto",
                    "--data", str(dataset_dir),
                    "--output-dir", str(results_dir),
                    "--max-num-iterations", "7000", 
                    "--steps-per-save", "1000"
                ]
                
                print(f"Running command: {' '.join(train_cmd)}")
                subprocess.run(train_cmd, check=True)
                
                # 2. Find the config file
                run_dirs = list(results_dir.glob("splatfacto/*"))
                if not run_dirs:
                    print("Error: Could not find training output directory.")
                    return

                latest_run_dir = max(run_dirs, key=os.path.getmtime)
                config_path = latest_run_dir / "config.yml"
                
                if not config_path.exists():
                     print(f"Error: Config file not found at {config_path}")
                     return

                print(f"Exporting Gaussian Splats from {config_path} -> {exports_dir}")
                
                # 3. Export to PLY
                export_cmd = [
                    "ns-export", "gaussian-splat",
                    "--load-config", str(config_path),
                    "--output-dir", str(exports_dir)
                ]
                
                print(f"Running command: {' '.join(export_cmd)}")
                subprocess.run(export_cmd, check=True)
                
                print(f"\n3DGS Generation Complete! Output: {exports_dir}")
                
                # 4. Download Results
                zip_and_download(output_dir)
                
            except subprocess.CalledProcessError as e:
                print(f"Error during Nerfstudio execution: {e}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
    else:
        print(f"\nProcessing failed: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
