"""全局 asyncio.Queue 实例 — 跨层消息通道。"""

from __future__ import annotations

import asyncio

from aion_knowledge.infrastructure.models import PostProcTask, UnifiedContext

ctx_queue: asyncio.Queue[UnifiedContext] = asyncio.Queue(maxsize=128)
"""ctx_queue：多源数据接入队列。发送方入队后立即返回，pipeline_worker 全局串行消费。"""

postproc_queue: asyncio.Queue[PostProcTask] = asyncio.Queue(maxsize=128)
"""postproc_queue：后处理任务队列。pipeline_worker 在双写完成后入队，postproc_worker 逐个消费。"""
