"""GraphMetadataRepository — graph_metadata + chunk_community 操作。"""
from __future__ import annotations

import json
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from aion_knowledge.common.uuid7 import uuid7


class GraphMetadataRepository:
    """graph_metadata 表 + chunk_community 表的操作。"""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def upsert_stats(
        self,
        kb_id: str,
        entity_count: int,
        relation_count: int,
        doc_count: int = 0,
        community_count: int = 0,
    ) -> None:
        await self._session.execute(
            text("""
                INSERT INTO graph_metadata
                    (id, kb_id, status, doc_count, entity_count, relation_count,
                     community_count, version, checkpoints, updated_at)
                VALUES
                    (:id, :kb_id, 'active', :doc_count, :entity_count, :relation_count,
                     :community_count, 1, '{}'::jsonb, now())
                ON CONFLICT (kb_id) DO UPDATE SET
                    entity_count = :entity_count2,
                    relation_count = :relation_count2,
                    doc_count = :doc_count2,
                    community_count = :community_count2,
                    version = graph_metadata.version + 1,
                    updated_at = now()
            """),
            {
                "id": uuid7(),
                "kb_id": kb_id,
                "entity_count": entity_count,
                "relation_count": relation_count,
                "doc_count": doc_count,
                "community_count": community_count,
                "entity_count2": entity_count,
                "relation_count2": relation_count,
                "doc_count2": doc_count,
                "community_count2": community_count,
            },
        )

    async def save_checkpoint(self, kb_id: str, key: str, data: dict[str, Any]) -> None:
        """写入顶层检查点（key 为单层键，如 'disambiguation'）。

        graph_metadata 行不存在时自动创建（INSERT ON CONFLICT），避免静默丢失。
        注意：jsonb_set 不创建缺失的中间路径，嵌套键请用 save_doc_checkpoint。
        """
        await self._session.execute(
            text("""
                INSERT INTO graph_metadata
                    (id, kb_id, status, doc_count, entity_count, relation_count,
                     community_count, version, checkpoints, updated_at)
                VALUES
                    (:id, :kb_id, 'active', 0, 0, 0, 0, 1,
                     jsonb_build_object(CAST(:key AS text), CAST(:val AS jsonb)),
                     now())
                ON CONFLICT (kb_id) DO UPDATE SET
                    checkpoints = jsonb_set(
                        COALESCE(graph_metadata.checkpoints, '{}'::jsonb),
                        CAST(:path AS text[]),
                        CAST(:val AS jsonb)
                    ),
                    updated_at = now()
            """),
            {
                "id": uuid7(),
                "kb_id": kb_id,
                # jsonb_build_object 键需 text，jsonb_set 路径需 text[]。
                # 同一参数不能同时 CAST AS text 与 AS text[]：asyncpg 按 Python 值解析 codec（list→数组），
                # 会使 CAST AS text 报 DataError。故拆为 :key（str）与 :path（list）两个参数。
                "key": key,
                "path": [key],
                "val": json.dumps(data),
            },
        )

    async def save_doc_checkpoint(
        self, kb_id: str, module: str, doc_id: str, data: dict[str, Any],
    ) -> None:
        """写入按文档的检查点（checkpoints.<module>.docs.<doc_id>）。

        用 || 逐层拼接，中间路径缺失时也能正确创建（jsonb_set 做不到）。
        """
        await self._session.execute(
            text("""
                INSERT INTO graph_metadata
                    (id, kb_id, status, doc_count, entity_count, relation_count,
                     community_count, version, checkpoints, updated_at)
                VALUES
                    (:id, :kb_id, 'active', 0, 0, 0, 0, 1,
                     jsonb_build_object(
                         CAST(:module AS text),
                         jsonb_build_object(
                             'docs',
                             jsonb_build_object(CAST(:doc_id AS text), CAST(:val AS jsonb))
                         )
                     ),
                     now())
                ON CONFLICT (kb_id) DO UPDATE SET
                    checkpoints = COALESCE(graph_metadata.checkpoints, '{}'::jsonb)
                        || jsonb_build_object(
                            CAST(:module AS text),
                            COALESCE(graph_metadata.checkpoints->CAST(:module AS text), '{}'::jsonb)
                            || jsonb_build_object(
                                'docs',
                                COALESCE(graph_metadata.checkpoints->CAST(:module AS text)->'docs', '{}'::jsonb)
                                || jsonb_build_object(CAST(:doc_id AS text), CAST(:val AS jsonb))
                            )
                        ),
                    updated_at = now()
            """),
            {
                "id": uuid7(),
                "kb_id": kb_id,
                "module": module,
                "doc_id": doc_id,
                "val": json.dumps(data),
            },
        )

    async def load_checkpoint(self, kb_id: str, key: str) -> Any | None:
        row = await self._session.execute(
            text("""
                SELECT checkpoints->:key->>'graph_hash' AS val
                FROM graph_metadata WHERE kb_id = :kb_id
            """),
            {"kb_id": kb_id, "key": key},
        )
        return row.scalar()

    async def get_checkpoint(
        self, kb_id: str, key: str | list[str],
    ) -> Any | None:
        """读取检查点的完整 JSON 值；key 支持嵌套路径（如 ['wiki','docs',doc_id]）。"""
        path = [key] if isinstance(key, str) else list(key)
        row = await self._session.execute(
            text("""
                SELECT checkpoints #> CAST(:path AS text[]) AS val
                FROM graph_metadata WHERE kb_id = :kb_id
            """),
            {"kb_id": kb_id, "path": path},
        )
        return row.scalar()

    async def clear_checkpoint(self, kb_id: str, key: str | list[str]) -> None:
        path = [key] if isinstance(key, str) else list(key)
        await self._session.execute(
            text("""
                UPDATE graph_metadata SET
                    checkpoints = COALESCE(checkpoints, '{}'::jsonb) #- CAST(:path AS text[]),
                    updated_at = now()
                WHERE kb_id = :kb_id
            """),
            {"kb_id": kb_id, "path": path},
        )

    async def count_communities(self, kb_id: str) -> int:
        """查询指定知识库的社区报告数量。"""
        row = await self._session.execute(
            text("SELECT COUNT(*) FROM chunk_community WHERE kb_id = :kb_id"),
            {"kb_id": kb_id},
        )
        return row.scalar() or 0

    async def get_communities(self, kb_id: str, limit: int = 5) -> list[dict[str, Any]]:
        rows = await self._session.execute(
            text("""
                SELECT community_id, summary, findings::text, payload->>'title' AS title
                FROM chunk_community
                WHERE kb_id = :kb_id
                LIMIT :limit
            """),
            {"kb_id": kb_id, "limit": limit},
        )
        return [
            {
                "community_id": row[0],
                "summary": row[1],
                "findings": row[2],
                "title": row[3],
            }
            for row in rows
        ]
