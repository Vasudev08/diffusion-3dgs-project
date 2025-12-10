"""
Tool for analyzing images using an LLM/VLM.
"""

import base64
import mimetypes
from pathlib import Path
from typing import override

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from agentic_image2dataset.tools.base import FileOperationTool


class VisualAnalysisToolArgs(BaseModel):
    """Arguments for the visual analysis tool."""

    image_paths: list[str] = Field(
        description="List of paths to the images to analyze."
    )
    query: str = Field(
        description="The question or prompt to ask about the images.",
        default="Describe these images in detail.",
    )


class VisualAnalysisTool(FileOperationTool):
    """Tool for performing visual analysis on images using a VLM."""

    name: str = "visual_analysis"
    description: str = (
        "Analyze one or more images using a Visual Language Model (VLM). "
        "Use this tool to ask questions about the visual content of images, "
        "such as 'Describe the main subject', 'Is the image blurry?', or 'Compare these two images'. "
        "This tool acts as a subagent that can 'see' the images."
    )
    args_schema = VisualAnalysisToolArgs
    llm: BaseChatModel

    def __init__(
        self,
        llm: BaseChatModel,
        workspace_root: Path,
        **kwargs: object,
    ):
        super().__init__(
            llm=llm,
            workspace_root=workspace_root,
            **kwargs,
        )

    @override
    def _run(
        self, image_paths: list[str], query: str = "Describe these images in detail."
    ) -> str:
        """Analyze the input images using the VLM."""
        try:
            content = [{"type": "text", "text": query}]

            for image_path in image_paths:
                actual_path = self._validate_path(image_path)
                if not actual_path.exists():
                    return f"Error: Image path '{image_path}' does not exist."

                # Encode image
                mime_type, _ = mimetypes.guess_type(actual_path)
                if not mime_type:
                    mime_type = "image/jpeg"

                with open(actual_path, "rb") as image_file:
                    image_data = base64.b64encode(image_file.read()).decode("utf-8")

                image_url = f"data:{mime_type};base64,{image_data}"
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    }
                )

            # Create message with images and query
            message = HumanMessage(content=content)

            # Invoke LLM
            response = self.llm.invoke([message])
            return str(response.content)

        except Exception as e:
            return f"Error performing visual analysis: {str(e)}"
