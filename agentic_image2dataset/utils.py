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
    analysis["is_blurry"] = laplacian_var < 100  # Threshold for blur detection

    # Brightness analysis
    mean_brightness = np.mean(gray)
    analysis["brightness"] = float(mean_brightness)
    analysis["is_dark"] = mean_brightness < 50
    analysis["is_bright"] = mean_brightness > 200

    # Contrast analysis
    contrast = np.std(gray)
    analysis["contrast"] = float(contrast)
    analysis["low_contrast"] = contrast < 30

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
        analysis["is_desaturated"] = mean_saturation < 30

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
