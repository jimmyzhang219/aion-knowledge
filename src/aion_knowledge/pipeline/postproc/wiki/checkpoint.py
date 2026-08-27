"""WikiCheckpointManager — 基于 graph_metadata.checkpoints 的按文档检查点。"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from aion_knowledge.infrastructure.db import get_session
from aion_knowledge.storage.relational.graph_repo import GraphMetadataRepository

logger = logging.getLogger(__name__)


class WikiCheckpointManager:
    """Wiki 百科构建检查点：按 (kb_id, doc_id) 记录执行结果。

    用于区分「模块未执行」与「模块已执行但无产出」（无候选 / LLM 失败），
    便于排查 chunk_wiki 为空的数据问题。
    """

    async def save(
        self,
        kb_id: str,
        doc_id: str,
        status: str,
        page_count: int = 0,
        candidate_count: int = 0,
    ) -> None:
        """保存该文档的 wiki 执行检查点（graph_metadata.checkpoints.wiki.docs.<doc_id>）。"""
        async with get_session() as session:
            repo = GraphMetadataRepository(session)
            await repo.save_doc_checkpoint(kb_id, "wiki", doc_id, {
                "status": status,
                "page_count": page_count,
                "candidate_count": candidate_count,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            })
            await session.commit()

    async def load(self, kb_id: str, doc_id: str) -> dict[str, Any] | None:
        """读取该文档的 wiki 检查点，无记录返回 None。"""
        async with get_session() as session:
            repo = GraphMetadataRepository(session)
            value = await repo.get_checkpoint(kb_id, ["wiki", "docs", doc_id])
            return value if isinstance(value, dict) else None
