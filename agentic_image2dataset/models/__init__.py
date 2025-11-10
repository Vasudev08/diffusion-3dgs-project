"""
Processing models for the agentic pipeline.
"""

from .base import BaseProcessingModel, ModelRegistry
from .super_resolution import AdcSRModel, DiffBIRModel
from .view_generator import StableVirtualCameraModel

__all__ = [
    "BaseProcessingModel",
    "ModelRegistry",
    "StableVirtualCameraModel",
    "DiffBIRModel",
    "AdcSRModel",
]
