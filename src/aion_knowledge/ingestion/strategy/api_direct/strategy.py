"""ApiDirectIngestionStrategy — 上游系统通过 REST API 直接推送数据的接入策略。"""
from __future__ import annotations

import logging
from typing import Any

from aion_knowledge.infrastructure.models import UnifiedContext
from aion_knowledge.ingestion.strategy.base import IngestionStrategy
from aion_knowledge.ingestion.strategy.registry import register_source
from aion_knowledge.models.enums import StrategyName

logger = logging.getLogger(__name__)


@register_source(StrategyName.api_direct)
class ApiDirectIngestionStrategy(IngestionStrategy):
    """上游系统通过 REST API 直接推送数据的接入策略。

    输入数据不涉及传统文件上传，但引擎层仍落地 S3 持久化并创建
    document 记录，保持与现有处理管线完全兼容。
    """

    source = StrategyName.api_direct.value

    @property
    def suffix(self) -> str:
        return self._suffix

    def __init__(self, suffix: str = "raw"):
        self._suffix = suffix

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
