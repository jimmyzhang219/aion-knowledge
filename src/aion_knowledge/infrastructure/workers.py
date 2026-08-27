"""Worker 编排函数 — ctx_queue / postproc_queue 的消费者循环。"""

from __future__ import annotations

import logging
from typing import Any

from aion_knowledge.common.trace import reset_trace_id, set_trace_id
from aion_knowledge.infrastructure.models import PostProcTask, UnifiedContext
from aion_knowledge.infrastructure.queues import ctx_queue, postproc_queue

logger = logging.getLogger(__name__)


# ── Pipeline Worker：核心处理管道 ────────────────────────────

async def pipeline_worker() -> None:
    """ctx_queue 消费者：全局单协程，逐个消息串行处理。"""
    logger.info("流水线 worker 已启动，等待 ctx_queue 任务...")
    while True:
        ctx: UnifiedContext = await ctx_queue.get()
        token = set_trace_id(ctx.trace_id)
        try:
            logger.info("流水线开始处理：文档=%s，来源=%s", ctx.doc_name, ctx.source)
            from aion_knowledge.indexing.executor import IndexingExecutor

            executor = IndexingExecutor()
            postproc_task = await executor.run(ctx)
            # 入队 postproc_queue：二批后处理由 postproc_worker 异步消费，此处不等待其完成
            await postproc_queue.put(postproc_task)
            logger.info("流水线处理完成：文档=%s → postproc_queue，chunks=%d",
                        ctx.doc_name, postproc_task.chunk_count)
        except Exception as exc:
            logger.error("流水线处理失败：文档=%s，错误=%s", ctx.doc_name, exc, exc_info=True)
        finally:
            ctx_queue.task_done()
            reset_trace_id(token)


# ── PostProc Worker：后处理管道 ──────────────────────────────

async def _run_postproc_subtasks(task: PostProcTask) -> None:
    """使用调度器执行已启用的后处理模块。"""
    from aion_knowledge.common.config import settings
    from aion_knowledge.infrastructure.db import get_session
    from aion_knowledge.ingestion.document_repo import get_document_by_id
    from aion_knowledge.pipeline.postproc.base import PostProcContext
    from aion_knowledge.pipeline.postproc.dispatcher import PostProcDispatcher
    from aion_knowledge.storage.relational.chunk_repo import ChunkRepository

    chunks: list[dict[str, Any]] = []
    try:
        async with get_session() as session:
            # 文档已删除/不存在：跳过整个任务（逻辑删除后不再向 KB 级表写数据；
            # 仅覆盖「排队期间已删」场景——检查通过后、二批写入前被删的残余窗口由规格接受）
            if await get_document_by_id(session, task.document_id) is None:
                logger.info("后处理跳过（文档已删除或不存在）：doc=%s", task.document_id)
                return
            repo = ChunkRepository(session)
            rows = await repo.get_by_document(document_id=task.document_id)
            chunks = [
                {
                    # str() 保持与首批 text 模块回写的 chunk_uuid 契约一致（均为字符串）
                    "chunk_uuid": str(r.id),
                    "content": r.content,
                    "seq_num": r.seq_num,
                    "chunk_type": r.chunk_type,
                    "chunk_metadata": r.chunk_metadata,
                }
                for r in rows
            ]
    except Exception as exc:
        logger.warning("后处理：加载 chunk 失败，文档=%s：%s", task.document_id, exc, exc_info=True)

    cfg = task.postproc_config
    settings_dict = {
        "keyword_extract":  settings.postproc_keyword_extract and cfg.enable_keyword_extract,
        "question_gen":     settings.postproc_question_gen and cfg.enable_question_gen,
        "summarizer":       settings.postproc_summarizer and cfg.enable_summarizer,
        "raptor":           settings.postproc_raptor and cfg.enable_raptor,
        "graph_extract":    settings.postproc_graph_extract and cfg.enable_graph_extract,
        "community":        settings.postproc_community and cfg.enable_community,
        "disambiguation":   settings.postproc_disambiguation and cfg.enable_disambiguation,
        "wiki":             settings.postproc_wiki and cfg.enable_wiki,
    }

    dispatcher = PostProcDispatcher(settings_dict, only=task.modules)
    ctx = PostProcContext(
        document_id=task.document_id,
        kb_id=task.kb_id,
        doc_name=task.doc_name,
        suffix=task.suffix,
        parser_id=task.parser_id,
        parser_config=task.parser_config,
    )
    await dispatcher.run_second_batch(ctx, chunks)


async def postproc_worker() -> None:
    """postproc_queue 消费者：全局单协程，逐个消息串行拉取，内部并发执行子任务。"""
    logger.info("后处理 worker 已启动，等待 postproc_queue 任务...")
    while True:
        task: PostProcTask = await postproc_queue.get()
        token = set_trace_id(task.trace_id)
        try:
            logger.info("后处理开始处理：文档=%s，chunks=%d",
                        task.doc_name, task.chunk_count)
            await _run_postproc_subtasks(task)
            logger.info("后处理处理完成：文档=%s", task.doc_name)
        except Exception as exc:
            logger.error("后处理处理失败：文档=%s，错误=%s", task.doc_name, exc, exc_info=True)
        finally:
            postproc_queue.task_done()
            reset_trace_id(token)
