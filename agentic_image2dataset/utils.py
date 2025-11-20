"""
Utility functions for image analysis and processing.
"""

from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np


def analyze_image_quality(image_path: Path) -> Dict[str, Any]:
    """
    Analyze image quality and characteristics.

    Args:
        image_path: Path to the input image

    Returns:
        Dictionary containing analysis results
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Convert to grayscale for analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # Basic image properties
    analysis = {
        "width": width,
        "height": height,
        "aspect_ratio": width / height,
        "total_pixels": width * height,
        "channels": image.shape[2] if len(image.shape) == 3 else 1,
    }

    # Blur detection using Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    analysis["blur_score"] = float(laplacian_var)
    analysis["is_blurry"] = bool(laplacian_var < 100)  # Threshold for blur detection

    # Brightness analysis
    mean_brightness = np.mean(gray)
    analysis["brightness"] = float(mean_brightness)
    analysis["is_dark"] = bool(mean_brightness < 50)
    analysis["is_bright"] = bool(mean_brightness > 200)

    # Contrast analysis
    contrast = np.std(gray)
    analysis["contrast"] = float(contrast)
    analysis["low_contrast"] = bool(contrast < 30)

    # Edge density (scene complexity)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (width * height)
    analysis["edge_density"] = float(edge_density)
    analysis["scene_complexity"] = (
        "high" if edge_density > 0.1 else "medium" if edge_density > 0.05 else "low"
    )

    # Color analysis
    if len(image.shape) == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Saturation analysis
        mean_saturation = np.mean(s)
        analysis["saturation"] = float(mean_saturation)
        analysis["is_desaturated"] = bool(mean_saturation < 30)

        # Color diversity
        unique_colors = len(np.unique(h.reshape(-1)))
        analysis["color_diversity"] = unique_colors
        analysis["monochromatic"] = unique_colors < 50

    # Resolution quality assessment
    if width < 512 or height < 512:
        analysis["resolution_quality"] = "low"
    elif width < 1024 or height < 1024:
        analysis["resolution_quality"] = "medium"
    else:
        analysis["resolution_quality"] = "high"

    # Overall quality score (0-1)
    quality_score = 0.0

    # Blur penalty
    if not analysis["is_blurry"]:
        quality_score += 0.3

    # Contrast bonus
    if not analysis["low_contrast"]:
        quality_score += 0.2

    # Resolution bonus
    if analysis["resolution_quality"] == "high":
        quality_score += 0.3
    elif analysis["resolution_quality"] == "medium":
        quality_score += 0.2

    # Brightness bonus (not too dark or too bright)
    if not analysis["is_dark"] and not analysis["is_bright"]:
        quality_score += 0.2

    analysis["quality_score"] = min(quality_score, 1.0)

    return analysis


def detect_image_issues(image_path: Path) -> List[str]:
    """
    Detect potential issues with the input image.

    Args:
        image_path: Path to the input image

    Returns:
        List of detected issues
    """
    issues = []
    analysis = analyze_image_quality(image_path)

    if analysis["is_blurry"]:
        issues.append("Image appears blurry")

    if analysis["is_dark"]:
        issues.append("Image is too dark")

    if analysis["is_bright"]:
        issues.append("Image is overexposed")

    if analysis["low_contrast"]:
        issues.append("Image has low contrast")

    if analysis["resolution_quality"] == "low":
        issues.append("Image resolution is low")

    if analysis["is_desaturated"]:
        issues.append("Image appears desaturated")

    return issues


def suggest_processing_order(analysis: Dict[str, Any]) -> List[str]:
    """
    Suggest the optimal processing order based on image analysis.

    Args:
        analysis: Image analysis results

    Returns:
        List of processing steps in suggested order
    """
    steps = []

    # If image is blurry or low resolution, suggest super-resolution first
    if analysis["is_blurry"] or analysis["resolution_quality"] in ["low", "medium"]:
        steps.append("super_resolution")

    # Always suggest view generation
    steps.append("view_generation")

    # If super-resolution wasn't applied first, suggest it after view generation
    if "super_resolution" not in steps:
        steps.append("super_resolution")

    return steps


def get_optimal_view_count(analysis: Dict[str, Any]) -> int:
    """
    Determine optimal number of views to generate based on image characteristics.

    Args:
        analysis: Image analysis results

    Returns:
        Suggested number of views
    """
    base_views = 24

    # Adjust based on scene complexity
    if analysis["scene_complexity"] == "high":
        return min(base_views + 12, 48)  # More views for complex scenes
    elif analysis["scene_complexity"] == "low":
        return max(base_views - 8, 12)  # Fewer views for simple scenes

    return base_views


def fix_transforms(transforms_path: Path) -> None:
    """
    Fix transforms.json by resizing and scaling intrinsics based on actual image dimensions.

    Args:
        transforms_path: Path to the transforms.json file
    """
    import json
    import cv2

    print(f"Processing {transforms_path}...")

    with open(transforms_path, "r") as f:
        data = json.load(f)

    if not data.get("frames"):
        print("No frames found in transforms.json")
        return

    transforms_dir = transforms_path.parent
    new_frames = []
    modified_count = 0

    for frame in data["frames"]:
        new_frame = frame.copy()

        # Resolve image path
        file_path = frame.get("file_path")
        if not file_path:
            print("Warning: Frame missing file_path, skipping.")
            new_frames.append(new_frame)
            continue

        # Handle relative paths
        image_path = transforms_dir / file_path
        if not image_path.exists():
            # Try checking if file_path is absolute or relative to cwd (fallback)
            image_path = Path(file_path)

        if not image_path.exists():
            print(
                f"Warning: Image not found at {image_path}, skipping dimension update."
            )
            new_frames.append(new_frame)
            continue

        # Read image to get actual dimensions
        # We read only the header if possible, but cv2.imread loads the whole image.
        # For efficiency with many large images, one might use PIL, but we stick to cv2 as per import.
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Warning: Could not read image {image_path}, skipping.")
            new_frames.append(new_frame)
            continue

        actual_h, actual_w = img.shape[:2]

        # Infer original dimensions from current intrinsics
        # Assuming principal point is at center, original W = cx * 2, H = cy * 2
        # This logic assumes the current 'cx' and 'cy' in the json are from some "original" resolution
        # that might not match the actual image if it was resized but intrinsics weren't updated,
        # OR if the intrinsics are correct for the "original" sensor but we have a resized image.
        # The goal here is to make intrinsics match the ACTUAL image dimensions.

        current_cx = frame["cx"]
        current_cy = frame["cy"]

        # If we assume the current intrinsics correspond to a theoretical image size of (2*cx, 2*cy)
        orig_w = current_cx * 2
        orig_h = current_cy * 2

        if orig_w == 0 or orig_h == 0:
            print(f"Warning: Invalid inferred dimensions for {file_path}, skipping.")
            new_frames.append(new_frame)
            continue

        scale_x = actual_w / orig_w
        scale_y = actual_h / orig_h

        # Update dimensions
        new_frame["w"] = actual_w
        new_frame["h"] = actual_h

        # Scale intrinsics
        new_frame["fl_x"] = frame["fl_x"] * scale_x
        new_frame["fl_y"] = frame["fl_y"] * scale_y
        new_frame["cx"] = frame["cx"] * scale_x
        new_frame["cy"] = frame["cy"] * scale_y

        new_frames.append(new_frame)
        modified_count += 1

    data["frames"] = new_frames

    # Overwrite the file
    with open(transforms_path, "w") as f:
        json.dump(data, f, indent=4)

    print(
        f"Saved fixed transforms to {transforms_path} (Updated {modified_count} frames)"
    )
