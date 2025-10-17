"""
Agentic Image to 3DGS Dataset Pipeline

A LangChain-based agentic pipeline for processing single input images into
3DGS-ready datasets using Stable Virtual Camera, Real-ESRGAN, and PyCOLMAP.
"""

from .config import LLMConfig, ModelConfig, PipelineConfig
from .pipeline import AgenticPipeline

__version__ = "0.1.0"
__all__ = ["AgenticPipeline", "PipelineConfig", "ModelConfig", "LLMConfig"]
