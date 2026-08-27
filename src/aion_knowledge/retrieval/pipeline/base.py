"""检索管线 Stage 协议。"""
from __future__ import annotations

from abc import ABC, abstractmethod

from .context import PipelineContext


class Stage(ABC):
    """管线阶段：接收 PipelineContext，修改 ctx.results 或 ctx.path_results。"""

    @abstractmethod
    async def run(self, ctx: PipelineContext) -> None:
        """执行当前阶段，通过修改 ctx 传递数据。"""
