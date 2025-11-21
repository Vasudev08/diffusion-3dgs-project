"""
Base classes for processing models in the agentic pipeline.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image


class BaseProcessingModel(ABC):
    """Abstract base class for all processing models."""

    def __init__(self, device: str = "cuda", **kwargs):
        self.device = device
        self.config = kwargs

    @abstractmethod
    def process(
        self, image_path: str | Path, output_dir: str | Path, **kwargs
    ) -> list[Path]:
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


class ModelRegistry:
    """Registry for managing available processing models with lazy initialization and role-based organization."""

    def __init__(self):
        self._model_factories: dict[str, Callable[[], BaseProcessingModel]] = {}
        self._model_roles: dict[str, str] = {}  # Maps model name to role
        self._role_registries: dict[
            str, list[str]
        ] = {}  # Maps role to list of model names

    def register(
        self,
        name: str,
        model_factory: Callable[[], BaseProcessingModel],
        role: str | None = None,
    ):
        """
        Register a model factory function.

        Args:
            name: Name to register the model under (should be the actual model name)
            model_factory: Callable that returns a BaseProcessingModel instance
            role: Optional role/category for the model (e.g., "super_resolution", "view_generation")
        """
        self._model_factories[name] = model_factory
        if role:
            self._model_roles[name] = role
            if role not in self._role_registries:
                self._role_registries[role] = []
            self._role_registries[role].append(name)

    def get(self, name: str) -> BaseProcessingModel | None:
        """
        Get a model by name, creating a new instance each time.

        Args:
            name: Name of the model to get

        Returns:
            A new model instance, or None if not found
        """
        if name not in self._model_factories:
            return None

        # Create a new model instance each time
        return self._model_factories[name]()

    def list_available(self) -> list[str]:
        """List names of all available models."""
        return list(self._model_factories.keys())

    def list_by_role(self, role: str) -> list[str]:
        """List names of models in a specific role/category."""
        return self._role_registries.get(role, [])

    def get_role(self, model_name: str) -> str | None:
        """Get the role/category of a model."""
        return self._model_roles.get(model_name)

    def get_all_roles(self) -> dict[str, list[str]]:
        """Get all roles and their associated models."""
        return dict(self._role_registries)

    def get_all(self) -> dict[str, BaseProcessingModel]:
        """Get all registered models, creating new instances."""
        result = {}
        for name in self._model_factories.keys():
            model = self.get(name)
            if model is not None:
                result[name] = model
        return result


def load_image(image_path: str | Path) -> np.ndarray:
    """Load an image and return as numpy array."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path)
    return np.array(image)


def save_image(image: np.ndarray, output_path: str | Path) -> Path:
    """Save a numpy array as an image."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    Image.fromarray(image).save(output_path)
    return output_path
