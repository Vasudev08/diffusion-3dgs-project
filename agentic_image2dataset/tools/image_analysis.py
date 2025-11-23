import json
from pathlib import Path
from typing import override

from agentic_image2dataset.tools.base import FileOperationTool
from agentic_image2dataset.utils import (
    analyze_image_quality,
    detect_image_issues,
)


class ImageAnalysisTool(FileOperationTool):
    """Tool for analyzing input images."""

    name: str = "analyze_image"
    description: str = "Analyze an image to determine its quality, characteristics, and processing needs"

    def __init__(self, workspace_root: Path, **kwargs: object):
        super().__init__(workspace_root=workspace_root, **kwargs)

    @override
    def _run(self, image_path: str, query: str = "") -> str:
        """Analyze the input image."""
        try:
            actual_path = self._validate_path(image_path)
            if not actual_path.exists():
                return f"Error: Image path '{image_path}' does not exist."

            analysis = analyze_image_quality(actual_path)
            issues = detect_image_issues(actual_path)

            result = {
                "analysis": analysis,
                "issues": issues,
            }

            return json.dumps(result, indent=2)
        except Exception as e:
            return f"Error analyzing image: {str(e)}"
