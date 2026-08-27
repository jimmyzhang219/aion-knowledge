"""DisambiguationMerger — Neo4j 图节点合并。"""
from __future__ import annotations

import logging
import uuid

from aion_knowledge.infrastructure.db import get_session

logger = logging.getLogger(__name__)


class DisambiguationMerger:
    """将一组同义实体合并到 canonical 名称。"""

    async def batch_merge(
        self, groups: list[tuple[str, list[str]]], kb_id: str
    ) -> None:
        """批量合并多组同义实体。"""
        for canonical, aliases in groups:
            await self.merge_entities(canonical, aliases, kb_id)

    async def merge_entities(
        self, canonical: str, aliases: list[str], kb_id: str
    ) -> None:
        """合并一组同义实体到标准名。"""
        if not aliases:
            return
        from aion_knowledge.infrastructure.graph import merge_aliases as neo4j_merge_aliases
        await neo4j_merge_aliases(kb_id, canonical, aliases)
        await self._record_merge_history(canonical, aliases, kb_id)

    async def _record_merge_history(
        self, canonical: str, aliases: list[str], kb_id: str
    ) -> None:
        """记录合并历史到 chunk_disambiguation 表。"""
        async with get_session() as session:
            from aion_knowledge.pipeline.postproc.disambiguation.orm import ChunkDisambiguation
            for alias in aliases:
                session.add(ChunkDisambiguation(
                    chunk_id=None,  # KB 级合并决策，无关联切片
                    kb_id=uuid.UUID(kb_id),
                    entity_name=alias,
                    resolved_id=canonical,
                    merged_into=canonical,
                    merge_confidence=1.0,
                    payload={"source": "disambiguation_merger"},
                ))
            await session.commit()
