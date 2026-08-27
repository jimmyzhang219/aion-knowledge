"""IngestionStrategy — 文档类型的接入策略基类。

模板方法模式：execute() 定义编排骨架，各子步骤拆分独立方法。
子类只需覆盖需要变更的步骤，无需重写整个 execute。
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from aion_knowledge.common.trace import get_trace_id
from aion_knowledge.infrastructure.db import get_session
from aion_knowledge.infrastructure.models import UnifiedContext
from aion_knowledge.infrastructure.queues import ctx_queue
from aion_knowledge.ingestion.document_repo import create_document
from aion_knowledge.ingestion.kb_guard import ensure_kb_exists
from aion_knowledge.ingestion.storage import (
    extract_storage_key,
    hash_file,
    save_to_storage,
)

logger = logging.getLogger(__name__)


class IngestionStrategy(ABC):
    """文档类型的接入策略基类。"""

    @property
    @abstractmethod
    def source(self) -> str:
        """UnifiedContext.source 标识。"""
        ...

    @property
    def suffix(self) -> str:
        """文档后缀标识。"""
        return ""

    async def execute(
        self,
        kb_id: str,
        content: bytes,
        file_name: str,
        creator: str = "system",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """完整接入流程：校验 KB 存在 → 前置处理 → hash → S3 → 创建 document → 构建 UnifiedContext → 入队。

        子类可覆盖任一子步骤来自定义行为，无需重写整个 execute。

        Returns:
            dict: {status, context_id, document_id}
        """
        # Step 0: 知识库存在性校验（程序化维护 kb 关系，所有写入必经）
        await self._ensure_kb_exists(kb_id)

        # Step 1: 前置处理（查重、解析等）
        if pre_result := await self._pre_process(
            kb_id=kb_id, content=content, file_name=file_name, **kwargs,
        ):
            return pre_result

        # Step 2: 哈希
        file_hash = self._compute_hash(content)

        # Step 3: S3 存储
        s3_key = self._build_s3_key(file_name, kb_id=kb_id, file_hash=file_hash)
        s3_ref = await self._save_to_storage(content, s3_key)
        file_path = extract_storage_key(s3_ref)

        # Step 4: 创建 document 记录
        async with get_session() as session:
            doc = await self._create_document_record(
                session=session,
                kb_id=kb_id,
                file_name=file_name,
                file_hash=file_hash,
                size=len(content),
                creator=creator,
                file_path=file_path,
                chunk_strategy=kwargs.get("chunk_strategy", "auto"),
            )

        # Step 5: 构建类型特定的 UnifiedContext
        ctx = await self._build_context(
            doc_id=str(doc.id),
            doc_name=file_name,
            kb_id=kb_id,
            s3_ref=s3_ref,
            suffix=self.suffix,
            content=content,
            **kwargs,
        )
        ctx.created_at = datetime.now(timezone.utc).isoformat()

        # 设置 S3 文件目录（派生自实际 key，子类自定义布局自动跟随）
        ctx.file_dir = os.path.dirname(s3_key)

        # Step 6: 入队
        await self._enqueue_context(ctx)

        logger.info(
            "%s enqueued: doc=%s ctx=%s doc_id=%s",
            self.__class__.__name__, file_name, ctx.context_id, doc.id,
        )

        return {
            "status": "queued",
            "context_id": ctx.context_id,
            "document_id": str(doc.id),
        }

    # ── 子步骤（子类按需覆盖） ──────────────────────────────

    async def _ensure_kb_exists(self, kb_id: str) -> None:
        """校验知识库存在，不存在则抛 KnowledgeBaseNotFoundError。"""
        await ensure_kb_exists(kb_id)

    async def _pre_process(
        self,
        kb_id: str,
        content: bytes,
        file_name: str,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        """前置处理：返回 dict 则短路 enqueue（如查重），返回 None 继续。"""
        return None

    def _compute_hash(self, content: bytes) -> str:
        """计算文件哈希。"""
        return hash_file(content)

    def _build_s3_key(self, file_name: str, kb_id: str, file_hash: str) -> str:
        """S3 存储路径：{kb_id}/{file_hash}/{原始文件名}，目录按 KB + hash 隔离，文件名保留原名。"""
        return f"{kb_id}/{file_hash}/{file_name}"

    async def _save_to_storage(self, content: bytes, s3_key: str) -> str:
        """保存到 S3/对象存储。"""
        return await save_to_storage(content, s3_key)

    async def _create_document_record(
        self,
        session: Any,
        kb_id: str,
        file_name: str,
        file_hash: str,
        size: int,
        creator: str,
        file_path: str,
        chunk_strategy: str,
    ) -> Any:
        """创建文档记录。"""
        return await create_document(
            session=session,
            kb_id=kb_id,
            doc_name=file_name,
            suffix=self.suffix,
            file_hash=file_hash,
            size=size,
            creator=creator,
            file_path=file_path,
            chunk_strategy=chunk_strategy,
            trace_id=get_trace_id(),  # 请求链路 ID：API 请求内为 X-Trace-ID 值，后台路径自动生成
        )

    @abstractmethod
    async def _build_context(
        self,
        doc_id: str,
        doc_name: str,
        kb_id: str,
        s3_ref: str,
        suffix: str,
        content: bytes,
        **kwargs: Any,
    ) -> UnifiedContext:
        """构建类型特定的 UnifiedContext。

        子类在此填充 source、ext_metadata 等字段。
        """
        ...

    async def _enqueue_context(self, ctx: UnifiedContext) -> None:
        """将 UnifiedContext 入队。"""
        await ctx_queue.put(ctx)
