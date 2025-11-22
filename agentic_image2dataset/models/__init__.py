"""
Processing models for the agentic pipeline.
"""

from .base import BaseProcessingModel, ModelRegistry
from .image_edit import QwenImageEditModel
from .super_resolution import AdcSRModel, DiffBIRModel
from .view_generation import StableVirtualCameraModel

__all__ = [
    "BaseProcessingModel",
    "ModelRegistry",
    "StableVirtualCameraModel",
    "DiffBIRModel",
    "AdcSRModel",
    "QwenImageEditModel",
]
