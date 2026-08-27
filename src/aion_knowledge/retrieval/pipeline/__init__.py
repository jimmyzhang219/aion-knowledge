"""检索管线 — 显式 Stage 编排。"""

from .base import Stage
from .context import PipelineContext
from .pipeline import RetrievalPipeline

__all__ = [
    "Stage",
    "PipelineContext",
    "RetrievalPipeline",
]
