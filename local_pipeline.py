import os
import sys
from pathlib import Path
import subprocess
from dotenv import load_dotenv

# Add project root to python path
sys.path.append(str(Path(__file__).parent))

from agentic_image2dataset import (
    AgenticPipeline,
    PipelineConfig,
    LLMConfig,
    ModelConfig,
)
from agentic_image2dataset.utils import get_system_resources

# Load .env file if present
load_dotenv()

def check_capabilities():
    """Check if the local machine has sufficient capabilities."""
    print("\nChecking system capabilities...")
    resources = get_system_resources()
    
    print(f"System RAM: {resources['system_ram_available_gb']:.2f} GB available (Total: {resources['system_ram_total_gb']:.2f} GB)")
    
    if resources['gpu_name'] != "None":
        print(f"GPU: {resources['gpu_name']}")
        print(f"VRAM: {resources['gpu_vram_available_gb']:.2f} GB available (Total: {resources['gpu_vram_total_gb']:.2f} GB)")
        
        if resources['gpu_vram_total_gb'] < 4:
            print("Warning: Low VRAM detected (< 4GB). 3DGS training might fail or be very slow.")
        elif resources['gpu_vram_total_gb'] < 8:
            print("ℹ Note: Moderate VRAM (4-8GB). You might need to reduce batch size or image resolution.")
        else:
            print("VRAM looks good for standard workloads.")
            
        return True
    else:
        print(" No CUDA GPU detected. This pipeline requires a CUDA-enabled GPU for 3DGS and diffusion models.")
        return False

def get_input_image():
    """Get input image path from user."""
    while True:
        path_str = input("\nEnter path to input image: ").strip()
        path_str = path_str.strip('"').strip("'")
        
        path = Path(path_str)
        if path.exists() and path.is_file():
            return path
        print(f"Error: File not found at {path}")

def main():
    print("=== Local Agentic 3DGS Pipeline ===")
    
    # 1. Check Capabilities
    if not check_capabilities():
        choice = input("\nContinue anyway with CPU (very slow/might fail)? (y/n): ").lower()
        if choice != 'y':
            return

    # 2. Setup API Key
    if "GOOGLE_API_KEY" not in os.environ:
        print("\nGoogle API Key not found in environment variables.")
        api_key = input("Please enter your Google API Key: ").strip()
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
        else:
            print("Error: API Key required for the agent.")
            return

    # 3. Get Input
    input_image = get_input_image()
    
    # 4. Configure Pipeline
    output_dir = Path("output_dataset_local")
    
    print("\nConfiguration:")
    use_high_quality = input("Use high quality settings (slower)? (y/n) [default: y]: ").lower() != 'n'
    
    # Create configuration (matching example_usage.py structure)
    config = PipelineConfig(
        llm=LLMConfig(model_name="gemini-2.5-flash", temperature=0.1),
        model=ModelConfig(
            device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" and subprocess.run("nvidia-smi", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0 else "cpu",
            num_views=32 if use_high_quality else 16,
            super_resolution_factor=4 if use_high_quality else 2,
        ),
        output_dir=output_dir,
        input_image=input_image,
        verbose=True,
    )
    
    # Manually add skip_colmap since it's not in the dataclass definition but used in logic
    config.skip_colmap = True

    # 5. Initialize pipeline
    print("\nInitializing agentic pipeline...")
    pipeline = AgenticPipeline(config)

    # 6. Process the image
    print(f"Processing image: {input_image}...")
    # Note: We pass skip_colmap here if the process method accepts it, 
    # but based on previous edits, we rely on config.skip_colmap or passing it explicitly.
    # The process method signature in pipeline.py accepts skip_colmap.
    result = pipeline.process(
        input_image=config.input_image,
        output_dir=config.output_dir,
    )

    if result["success"]:
        print("\nProcessing completed successfully!")
        # Handle generated_images key safely as in example_usage.py
        if "generated_images" in result:
            print(f"Generated {result['generated_images']} images")
        elif "execution_result" in result:
            print(f"Execution Result: {result['execution_result']}")
        
        print(f"Output directory: {result['output_dir']}")
        
        if result.get("issues"):
            print(f"Issues detected: {', '.join(result['issues'])}")
            
        # Run Splatfacto Training
        if getattr(config, 'skip_colmap', False):
            print("\n[Extra] Running Splatfacto training (since COLMAP was skipped)...")
            try:
                dataset_dir = result['output_dir']
                results_dir = dataset_dir / "results"
                exports_dir = dataset_dir / "exports"
                
                print(f"Training Splatfacto -> {results_dir}")
                
                # 1. Train Splatfacto
                train_cmd = [
                    sys.executable, "-m", "nerfstudio.scripts.train", "splatfacto",
                    "--data", str(dataset_dir),
                    "--output-dir", str(results_dir),
                    "--max-num-iterations", "10000",
                    "--steps-per-save", "1000"
                ]
                
                print(f"Running command: {' '.join(train_cmd)}")
                subprocess.run(train_cmd, check=True)
                
                # 2. Find the config file (it's in a timestamped subdirectory)
                # We need to find the latest run directory
                run_dirs = list(results_dir.glob("splatfacto/*"))
                if not run_dirs:
                    print("Error: Could not find training output directory.")
                    return

                # Sort by modification time to get the latest run
                latest_run_dir = max(run_dirs, key=os.path.getmtime)
                config_path = latest_run_dir / "config.yml"
                
                if not config_path.exists():
                     print(f"Error: Config file not found at {config_path}")
                     return

                print(f"Exporting Gaussian Splats from {config_path} -> {exports_dir}")
                
                # 3. Export to PLY
                export_cmd = [
                    sys.executable, "-m", "nerfstudio.scripts.export", "gaussian-splat",
                    "--load-config", str(config_path),
                    "--output-dir", str(exports_dir)
                ]
                
                print(f"Running command: {' '.join(export_cmd)}")
                subprocess.run(export_cmd, check=True)
                
                print(f"\n3DGS Generation Complete! Output: {exports_dir}")
                
            except subprocess.CalledProcessError as e:
                print(f"Error during Nerfstudio execution: {e}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
    else:
        print(f"\nProcessing failed: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
