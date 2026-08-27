"""RegularIngestionStrategy — 常规文档的接入策略。"""
from __future__ import annotations

import logging
from typing import Any

from aion_knowledge.infrastructure.db import get_session
from aion_knowledge.infrastructure.models import UnifiedContext
from aion_knowledge.ingestion.document_repo import create_ingestion_task, get_document_by_hash
from aion_knowledge.ingestion.strategy.base import IngestionStrategy
from aion_knowledge.ingestion.strategy.registry import register_source
from aion_knowledge.models.enums import StrategyName

logger = logging.getLogger(__name__)


@register_source(StrategyName.regular)
@register_source(StrategyName.url_import)
@register_source(StrategyName.manual_entry)
class RegularIngestionStrategy(IngestionStrategy):
    """常规文档的接入策略（文件上传、URL 导入、手动录入等）。"""

    def __init__(self, suffix: str, source: str = "regular"):
        self._suffix = suffix
        self._source = source

    @property
    def source(self) -> str:
        return self._source

    @property
    def suffix(self) -> str:
        return self._suffix

    async def _pre_process(
        self,
        kb_id: str,
        content: bytes,
        file_name: str,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        """前置处理：查重。返回 duplicate dict 或 None。"""
        file_hash = self._compute_hash(content)

        async with get_session() as session:
            existing = await get_document_by_hash(session, kb_id, file_hash)
            if existing:
                logger.info(
                    "Duplicate file skipped: hash=%s doc=%s", file_hash, file_name
                )
                return {"status": "duplicate", "document_id": str(existing.id)}

        return None

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
        """构建 UnifiedContext，创建 ingestion task 追踪进度。"""
        async with get_session() as session:
            task = await create_ingestion_task(session, doc_id)

        ext_meta = {"document_id": doc_id, "task_id": str(task.id)}
        if kwargs.get("creator"):
            ext_meta["creator"] = kwargs["creator"]

        return UnifiedContext(
            source=self.source,
            kb_id=kb_id,
            doc_name=doc_name,
            suffix=suffix,
            original_file_ref=s3_ref,
            content=content,
            chunk_strategy=kwargs.get("chunk_strategy", "auto"),
            ext_metadata=ext_meta,
        )
