"""FAQIngestionStrategy — FAQ 批量导入的接入策略。"""
from __future__ import annotations

import logging
from typing import Any

from aion_knowledge.infrastructure.db import get_session
from aion_knowledge.infrastructure.models import UnifiedContext
from aion_knowledge.ingestion.document_repo import get_document_by_hash
from aion_knowledge.ingestion.strategy.base import IngestionStrategy
from aion_knowledge.ingestion.strategy.registry import register_source
from aion_knowledge.models.enums import StrategyName

logger = logging.getLogger(__name__)


@register_source(StrategyName.faq)
class FAQIngestionStrategy(IngestionStrategy):
    """FAQ 批量导入的接入策略。"""

    source = "faq"

    def __init__(self, suffix: str, **kwargs: Any) -> None:
        super().__init__()
        self._suffix = suffix

    @property
    def suffix(self) -> str:
        return self._suffix

    async def _pre_process(
        self, kb_id: str, content: bytes, file_name: str, **kwargs: Any
    ) -> dict[str, Any] | None:
        """前置处理：查重。"""
        file_hash = self._compute_hash(content)

        async with get_session() as session:
            existing = await get_document_by_hash(session, kb_id, file_hash)
            if existing:
                logger.info("Duplicate FAQ skipped: hash=%s doc=%s", file_hash, file_name)
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
        return UnifiedContext(
            source=self.source,
            kb_id=kb_id,
            doc_name=doc_name,
            suffix=suffix,
            original_file_ref=s3_ref,
            content=content,
            ext_metadata={"document_id": doc_id},
        )
