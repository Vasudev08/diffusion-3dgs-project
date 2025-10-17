"""
Base classes for processing models in the agentic pipeline.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image


class BaseProcessingModel(ABC):
    """Abstract base class for all processing models."""

    def __init__(self, device: str = "cuda", **kwargs):
        self.device = device
        self.config = kwargs

    @abstractmethod
    def analyze(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Analyze an image and return metadata about its characteristics.

        Args:
            image_path: Path to the input image

        Returns:
            Dictionary containing analysis results
        """
        pass

    @abstractmethod
    def process(
        self, image_path: Union[str, Path], output_dir: Union[str, Path], **kwargs
    ) -> List[Path]:
        """
        Process an image and save results to output directory.

        Args:
            image_path: Path to the input image
            output_dir: Directory to save processed images
            **kwargs: Additional processing parameters

        Returns:
            List of paths to generated images
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Get a description of what this model does."""
        pass

    @abstractmethod
    def get_requirements(self) -> Dict[str, Any]:
        """Get requirements for this model to run."""
        pass

    def is_available(self) -> bool:
        """Check if this model is available (dependencies installed, etc.)."""
        try:
            requirements = self.get_requirements()
            return all(self._check_requirement(req) for req in requirements)
        except Exception:
            return False

    def _check_requirement(self, requirement: str) -> bool:
        """Check if a specific requirement is met."""
        try:
            __import__(requirement)
            return True
        except ImportError:
            return False


class ModelRegistry:
    """Registry for managing available processing models."""

    def __init__(self):
        self._models: Dict[str, BaseProcessingModel] = {}

    def register(self, name: str, model: BaseProcessingModel):
        """Register a model with the given name."""
        self._models[name] = model

    def get(self, name: str) -> Optional[BaseProcessingModel]:
        """Get a model by name."""
        return self._models.get(name)

    def list_available(self) -> List[str]:
        """List names of available models."""
        return [name for name, model in self._models.items() if model.is_available()]

    def get_all(self) -> Dict[str, BaseProcessingModel]:
        """Get all registered models."""
        return self._models.copy()


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """Load an image and return as numpy array."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path)
    return np.array(image)


def save_image(image: np.ndarray, output_path: Union[str, Path]) -> Path:
    """Save a numpy array as an image."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    Image.fromarray(image).save(output_path)
    return output_path
