import os
import shutil
from typing import override

from pydantic import BaseModel, Field

from agentic_image2dataset.tools.base import FileOperationTool


class ListDirectoryTool(FileOperationTool):
    """Tool for listing files in a directory."""

    name: str = "list_directory"
    description: str = "List files in a directory within the workspace. Use this to check generated outputs."

    class Args(BaseModel):
        directory: str = Field(
            default=".", description="Directory to list (relative to workspace root)"
        )

    args_schema = Args

    @override
    def _run(self, directory: str = ".") -> str:
        try:
            target_dir = self._validate_path(directory)
            if not target_dir.exists():
                return f"Directory '{directory}' does not exist."
            if not target_dir.is_dir():
                return f"Path '{directory}' is not a directory."

            files = sorted(os.listdir(target_dir))
            if not files:
                return "Directory is empty."

            return "\n".join(files)
        except Exception as e:
            return f"Error listing directory: {str(e)}"


class CopyFileTool(FileOperationTool):
    """Tool for copying files."""

    name: str = "copy_file"
    description: str = "Copy a file or directory within the workspace."

    class Args(BaseModel):
        source: str = Field(description="Source path (relative to workspace root)")
        destination: str = Field(
            description="Destination path (relative to workspace root)"
        )

    args_schema = Args

    @override
    def _run(self, source: str, destination: str) -> str:
        try:
            src_path = self._validate_path(source)
            dst_path = self._validate_path(destination)

            if not src_path.exists():
                return f"Source '{source}' does not exist."

            if src_path.is_dir():
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            else:
                # Ensure parent directory exists
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_path)

            return f"Successfully copied '{source}' to '{destination}'."
        except Exception as e:
            return f"Error copying file: {str(e)}"


class MoveFileTool(FileOperationTool):
    """Tool for moving files."""

    name: str = "move_file"
    description: str = "Move a file or directory within the workspace."

    class Args(BaseModel):
        source: str = Field(description="Source path (relative to workspace root)")
        destination: str = Field(
            description="Destination path (relative to workspace root)"
        )

    args_schema = Args

    @override
    def _run(self, source: str, destination: str) -> str:
        try:
            src_path = self._validate_path(source)
            dst_path = self._validate_path(destination)

            if not src_path.exists():
                return f"Source '{source}' does not exist."

            # Ensure parent directory exists
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_path), str(dst_path))

            return f"Successfully moved '{source}' to '{destination}'."
        except Exception as e:
            return f"Error moving file: {str(e)}"


class DeleteFileTool(FileOperationTool):
    """Tool for deleting files."""

    name: str = "delete_file"
    description: str = (
        "Delete a file or directory within the workspace. USE WITH CAUTION."
    )

    class Args(BaseModel):
        path: str = Field(description="Path to delete (relative to workspace root)")

    args_schema = Args

    @override
    def _run(self, path: str) -> str:
        try:
            target_path = self._validate_path(path)

            if not target_path.exists():
                return f"Path '{path}' does not exist."

            if target_path.is_dir():
                shutil.rmtree(target_path)
            else:
                os.remove(target_path)

            return f"Successfully deleted '{path}'."
        except Exception as e:
            return f"Error deleting file: {str(e)}"
