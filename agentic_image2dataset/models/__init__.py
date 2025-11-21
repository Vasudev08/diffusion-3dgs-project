"""
Processing models for the agentic pipeline.
"""

from .base import BaseProcessingModel, ModelRegistry
from .super_resolution import AdcSRModel, DiffBIRModel
from .view_generator import StableVirtualCameraModel
from .qwen_edit import QwenImageEditModel

__all__ = [
    "BaseProcessingModel",
    "ModelRegistry",
    "StableVirtualCameraModel",
    "DiffBIRModel",
    "AdcSRModel",
    "QwenImageEditModel",
]
