#!/usr/bin/env python
"""Pipeline 端到端运行脚本。

模拟完整链路：
  上传 Markdown 文件
    → LocalFileStore 存储 + 元数据入库
    → UnifiedContext → Queue1
    → Worker1: 文档解析 → 分块 → 双写 (InMemoryChunkStore + EmbeddingDispatcher)
    → PostProcTask → Queue2
    → Worker2: 后处理（全部禁用，仅打日志）
    → 完成

用法：
  python scripts/run_pipeline_e2e.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
import textwrap

from aion_knowledge.ingestion.file_upload import enqueue_upload

from aion_knowledge.infrastructure.queues import ctx_queue, postproc_queue
from aion_knowledge.infrastructure.workers import pipeline_worker, postproc_worker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("pipeline_e2e")


async def main() -> None:
    logger.info("=" * 60)
    logger.info("Pipeline E2E 测试开始")
    logger.info("=" * 60)

    # ── 1. 创建测试 Markdown 文档 ────────────────────────────────
    md_content = textwrap.dedent("""\
    # Pipeline 端到端测试文档

    本文档用于验证 Aion Knowledge 异步队列管道的完整链路。

    ## 第一章：概述

    Aion Knowledge 是一个知识库系统，支持多源数据接入、文档解析、分块、
    向量化存储、以及多路检索。

    核心特性包括：
    - 双层异步队列解耦（Queue1 核心处理 / Queue2 后处理）
    - 自适应分块策略链（Heading → Heuristic → Recursive）
    - 双写持久化（关系库 + 向量库）
    - 配置驱动的后处理管道

    ## 第二章：技术架构

    系统采用模块化设计，各组件职责清晰、可独立测试。

    ### 2.1 数据接入层

    支持 file_upload、url_import、manual_entry、connector、FAQ、API 等多种数据源。
    所有数据源统一经过"落盘 → 封装 UnifiedContext → 入队 Queue1"的流程。

    ### 2.2 核心处理层

    Worker1 串行执行三条流水线阶段：
    1. Parser：根据文档类型选择解析引擎，统一输出 Markdown
    2. Chunker：自适应分块策略链，输出 ChunkResult 列表
    3. DualWrite：关系库 chunks 表 + 向量库嵌入

    ## 第三章：验证点

    以下是需要验证的关键节点：

    | 步骤 | 验证内容 |
    |------|----------|
    | enqueue_upload | 返回 {"status": "queued", "context_id": ...} |
    | Queue1 消费 | UnifiedContext 字段完整 |
    | Parser | 解析成功，输出 Markdown 文件 |
    | Chunker | chunk_count > 0，内容完整 |
    | DualWrite | chunks 写入 + 嵌入调度 |
    | Queue2 消费 | PostProcTask 字段完整 |
    | 日志输出 | 包含各阶段关键日志 |

    ## 第四章：后续

    后处理阶段当前所有子任务均已禁用。
    启用后 Worker2 将并发执行：关键词提取、问题生成、摘要、图谱提取等。
    """).encode("utf-8")

    # ── 2. 启动 Worker 后台任务 ────────────────────────────────
    logger.info("启动 Worker1 (Queue1 消费者) 和 Worker2 (Queue2 消费者)...")
    worker1_task = asyncio.create_task(pipeline_worker(), name="pipeline_worker")
    worker2_task = asyncio.create_task(postproc_worker(), name="postproc_worker")
    logger.info("Worker 已启动，等待处理...")
    print()

    # ── 3. 上传文件到管道 ──────────────────────────────────────
    logger.info(">>> Step 1: enqueue_upload 上传测试文档")
    result = await enqueue_upload(
        kb_id="00000000-0000-0000-0000-000000000001",
        file_content=md_content,
        file_name="pipeline_e2e_test.md",
        suffix="md",
        chunk_strategy="auto",
    )
    logger.info("<<< enqueue_upload 返回: %s", result)
    print()

    # ── 4. 等待队列消费完成 ────────────────────────────────────
    logger.info("等待 Queue1 消费完成...")
    await ctx_queue.join()
    logger.info("ctx_queue 消费完成，等待 postproc_queue 消费完成...")
    await postproc_queue.join()
    logger.info("postproc_queue 已完成")
    print()

    # ── 5. 停止 Worker ─────────────────────────────────────────
    worker1_task.cancel()
    worker2_task.cancel()
    await asyncio.gather(worker1_task, worker2_task, return_exceptions=True)

    logger.info("=" * 60)
    logger.info("Pipeline E2E 测试完成 ✅")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
