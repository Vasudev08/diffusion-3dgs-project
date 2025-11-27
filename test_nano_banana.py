
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from agentic_image2dataset.tools.nano_banana_tool import NanoBananaTool

def test_tool():
    load_dotenv()
    
    if "NANO_API_KEY" not in os.environ:
        print("Error: NANO_API_KEY not found in environment.")
        return

    # Check for example input
    input_image = Path("Mango.jpg")
    if not input_image.exists():
        print(f"Error: {input_image} not found. Please provide an image to test.")
        return

    print(f"Testing Nano Banana Tool with {input_image}...")
    
    tool = NanoBananaTool()
    result = tool._run(str(input_image))
    
    print("\nTool Output:")
    print(result)

if __name__ == "__main__":
    test_tool()
