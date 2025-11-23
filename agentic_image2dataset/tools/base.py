from abc import ABCMeta
from pathlib import Path

from langchain.tools import BaseTool


class FileOperationTool(BaseTool, metaclass=ABCMeta):
    """Base class for file operation tools with safety checks."""

    workspace_root: Path

    def __init__(self, workspace_root: Path, **kwargs: object):
        super().__init__(workspace_root=workspace_root, **kwargs)

    def _validate_path(self, path_str: str) -> Path:
        """Validate that the path is within the workspace."""
        try:
            path = (self.workspace_root / path_str).resolve()
            if not str(path).startswith(str(self.workspace_root.resolve())):
                raise ValueError(
                    f"Access denied: Path '{path_str}' is outside the workspace."
                )
            return path
        except Exception as e:
            raise ValueError(f"Invalid path '{path_str}': {str(e)}")
