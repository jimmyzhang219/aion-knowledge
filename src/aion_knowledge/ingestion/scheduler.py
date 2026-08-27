"""应用启动时创建 Worker1/Worker2 的后台任务。"""

from __future__ import annotations

import asyncio
import logging

from aion_knowledge.indexing.checkpoint import CheckpointManager
from aion_knowledge.infrastructure.workers import pipeline_worker, postproc_worker

logger = logging.getLogger(__name__)


async def start_workers() -> list[asyncio.Task[None]]:
    """启动 Worker1 和 Worker2 的后台任务。

    启动前检查 stale checkpoint，记录需要恢复的文档。
    """
    # 启动前检查 stale checkpoint
    mgr = CheckpointManager()
    stale_ids = await mgr.list_stale()
    if stale_ids:
        logger.warning("Found %d stale checkpoints: %s", len(stale_ids), stale_ids)

    task1 = asyncio.create_task(pipeline_worker(), name="pipeline_worker")
    task2 = asyncio.create_task(postproc_worker(), name="postproc_worker")
    logger.info("Workers started: pipeline_worker (ctx_queue consumer), postproc_worker (postproc_queue consumer)")
    return [task1, task2]


async def stop_workers(tasks: list[asyncio.Task[None]]) -> None:
    """取消 Worker 后台任务。"""
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    logger.info("Workers stopped.")
