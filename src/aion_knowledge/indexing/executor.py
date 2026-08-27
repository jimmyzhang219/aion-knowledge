"""IndexingExecutor — 索引构建的核心编排器。

职责：接收 UnifiedContext → 策略 execute → first_batch → PostProcTask。
不再持有 pipeline 步骤实现，全部委托给策略和 pipeline 模块。
"""

from __future__ import annotations

import logging

from aion_knowledge.indexing.strategy.base import ChunkingStrategy
from aion_knowledge.indexing.strategy.registry import get_strategy
from aion_knowledge.infrastructure.models import PostProcTask, UnifiedContext

logger = logging.getLogger(__name__)


class IndexingExecutor:
    """编排索引构建：策略 execute → first_batch → 完成。"""

    async def run(self, ctx: UnifiedContext) -> PostProcTask:
        """执行完整索引流水线。"""
        from aion_knowledge.infrastructure.db import get_session
        from aion_knowledge.ingestion.document_repo import (
            update_ingestion_task_status,
        )

        doc_id = ctx.ext_metadata.get("document_id")
        task_id = ctx.ext_metadata.get("task_id")

        strategy = self._get_strategy(ctx)

        logger.info("索引构建启动：文档=%s，后缀=%s，策略=%s",
                     ctx.doc_name, ctx.suffix,
                     strategy.__class__.__name__)

        # Step 0: 标记 processing
        if task_id:
            async with get_session() as session:
                await update_ingestion_task_status(session, task_id, "processing")

        # Step 1-5: 策略创建 chunks — 通过 pipeline 模块直接执行
        chunks = await strategy.execute(ctx)

        # Step 6: PostProc first batch — text + img + vector（无 chunk 时跳过）
        if chunks:
            await self._run_first_batch(ctx, doc_id, chunks)
        else:
            logger.warning("索引构建：文档 %s 未产生任何 chunk", ctx.doc_name)

        # Step 7: 标记 completed
        await self._mark_completed(doc_id, task_id)

        # 8: 构建 PostProcTask
        return self._build_postproc_task(ctx, chunks)

    def _get_strategy(self, ctx: UnifiedContext) -> ChunkingStrategy:
        """根据文档 source 选择对应的索引构建策略。"""
        return get_strategy(ctx.source)

    async def _run_first_batch(
        self, ctx: UnifiedContext, doc_id: str | None, chunks: list[dict[str, object]],
    ) -> None:
        """首批 always_on 模块：Text 落库 → VLM → Vector。"""
        from aion_knowledge.common.config import settings

        if settings.index_only:
            logger.info("INDEX_ONLY 模式：跳过 first batch，文档=%s", ctx.doc_name)
            return

        from aion_knowledge.pipeline.postproc.base import PostProcContext
        from aion_knowledge.pipeline.postproc.dispatcher import PostProcDispatcher

        pp_ctx = PostProcContext(
            document_id=str(doc_id) if doc_id else "",
            kb_id=ctx.kb_id,
            doc_name=ctx.doc_name,
        )
        dispatcher = PostProcDispatcher({})
        await dispatcher.run_first_batch(pp_ctx, chunks)

        logger.info("首批模块完成：文档=%s，chunks=%d",
                     ctx.doc_name, len(chunks))

    async def _mark_completed(
        self, doc_id: str | None, task_id: str | None,
    ) -> None:
        """更新 document/ingestion_task 状态为 completed。"""
        if not doc_id and not task_id:
            return

        from aion_knowledge.infrastructure.db import get_session
        from aion_knowledge.ingestion.document_repo import (
            update_document_status,
            update_ingestion_task_status,
        )

        async with get_session() as session:
            if doc_id:
                await update_document_status(session, doc_id, "completed")
            if task_id:
                await update_ingestion_task_status(session, task_id, "completed")

    def _build_postproc_task(
        self, ctx: UnifiedContext, chunks: list[dict[str, object]],
    ) -> PostProcTask:
        """封装为后处理任务。"""
        from aion_knowledge.infrastructure.models import PostProcConfig
        return PostProcTask(
            document_id=ctx.ext_metadata.get("document_id", ctx.context_id),
            kb_id=ctx.kb_id,
            doc_name=ctx.doc_name,
            chunk_count=len(chunks),
            postproc_config=PostProcConfig(),
            suffix=ctx.suffix,
            trace_id=ctx.trace_id,  # 穿透 ctx_queue → postproc_queue 的请求链路 ID
        )

    # ╳ 以下方法已移除，由 pipeline 模块替代：
    # ╳ download_raw()      → pipeline.downloader.Downloader
    # ╳ run_parser()        → pipeline.parser.Parser
    # ╳ run_cleaner()       → pipeline.cleaner.Cleaner
    # ╳ run_chunker()       → pipeline.chunker.Chunker
    # ╳ async_upload_md()   → resolve_storage().upload()
    # ╳ _upload_images()    → 策略中直接使用 resolve_storage()
