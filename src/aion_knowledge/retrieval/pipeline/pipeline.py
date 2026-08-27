"""RetrievalPipeline — 顺序编排 Stage。"""
from __future__ import annotations

import logging
import time

from .base import Stage
from .context import PipelineContext

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """检索管线：按注册顺序依次执行 Stage。"""

    def __init__(self, stages: list[Stage]) -> None:
        self._stages = stages

    async def run(self, ctx: PipelineContext) -> None:
        """按顺序执行所有 Stage，每个 Stage 通过 ctx 传递数据。"""
        for stage in self._stages:
            start = time.perf_counter()
            await stage.run(ctx)
            logger.info(
                "Stage[%s] 执行耗时 %.2fms",
                stage.__class__.__name__,
                (time.perf_counter() - start) * 1000,
            )
