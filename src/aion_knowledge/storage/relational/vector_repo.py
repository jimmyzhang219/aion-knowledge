"""VectorRepository — chunk_vector 表的读写封装（含 pgvector 距离检索）。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, cast
from uuid import UUID

from sqlalchemy import CursorResult, Row, text
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class VectorResult:
    chunk_id: str
    content: str
    document_id: str
    kb_id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorRepository:
    """chunk_vector 表的数据库操作（含 pgvector 余弦距离检索）。"""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def similarity_search(
        self,
        kb_id: str,
        embedding: list[float],
        top_k: int,
        *,
        embedding_column: str = "embedding",
        chunk_type: str | None = None,
    ) -> list[VectorResult]:
        """向量余弦距离检索，JOIN chunk_text 返回 content。

        embedding_column 支持: embedding / embedding_summary / embedding_questions
        """
        allowed_columns = frozenset({"embedding", "embedding_summary", "embedding_questions"})
        if embedding_column not in allowed_columns:
            raise ValueError(f"embedding_column must be one of {allowed_columns}")
        embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"
        chunk_type_filter = " AND c.chunk_type = :chunk_type" if chunk_type else ""
        not_null_filter = f" AND v.{embedding_column} IS NOT NULL" if embedding_column != "embedding" else ""
        rows = await self._session.execute(
            text(f"""
                SELECT c.id, c.content, c.document_id, c.kb_id, c.metadata,
                       (1 - (v.{embedding_column} <=> CAST(:query_embedding AS vector))) AS score
                FROM chunk_vector v
                JOIN chunk_text c ON c.id = v.chunk_id::uuid
                JOIN doc_knowledge_documents d ON d.id = c.document_id AND NOT d.deleted
                JOIN kb_knowledge_bases k ON k.id = c.kb_id AND NOT k.deleted
                WHERE c.kb_id = :kb_id{chunk_type_filter}{not_null_filter}
                ORDER BY score DESC
                LIMIT :limit
            """),
            {
                "query_embedding": embedding_str,
                "kb_id": kb_id,
                "limit": top_k,
                "chunk_type": chunk_type,
            },
        )
        return [self._row_to_vector_result(r) for r in rows]

    async def fetch_embeddings(self, chunk_ids: list[str]) -> dict[str, list[float]]:
        """按 chunk_id 批量取回 chunk_vector.embedding（content 向量）。

        Args:
            chunk_ids: 待取向量的 chunk_id 列表。

        Returns:
            {chunk_id: embedding}，未命中的 chunk_id 不包含在结果中。
            embedding 列为 pgvector 文本格式 "[v1,v2,...]"，经 json.loads 解析为 list[float]。
        """
        if not chunk_ids:
            return {}
        rows = await self._session.execute(
            text("""
                SELECT chunk_id::text AS chunk_id, embedding
                FROM chunk_vector
                WHERE chunk_id = ANY(CAST(:ids AS uuid[]))
            """),
            {"ids": chunk_ids},
        )
        return {str(r.chunk_id): json.loads(str(r.embedding)) for r in rows}

    @staticmethod
    def _row_to_vector_result(r: Row[Any]) -> VectorResult:
        metadata = dict(r.metadata) if r.metadata else {}
        return VectorResult(
            chunk_id=str(r.id),
            content=r.content,
            document_id=str(r.document_id),
            kb_id=str(r.kb_id),
            score=float(r.score),
            metadata=metadata,
        )

    async def insert(
        self,
        id: UUID,
        chunk_id: str,
        kb_id: str,
        embedding: list[float],
        payload: dict[str, Any],
    ) -> None:
        await self._session.execute(
            text("""
                INSERT INTO chunk_vector (id, chunk_id, kb_id, embedding, payload)
                VALUES (:id, CAST(:chunk_id AS uuid), CAST(:kb_id AS uuid), CAST(:embedding AS vector), CAST(:payload AS jsonb))
            """),
            {
                "id": id,
                "chunk_id": chunk_id,
                "kb_id": kb_id,
                "embedding": str(embedding),
                "payload": json.dumps(payload),
            },
        )

    async def update_questions(
        self,
        chunk_id: str,
        questions: str,
        embedding_questions: list[float] | None = None,
    ) -> int:
        """写入 questions 和可选的 embedding_questions。返回影响行数。"""
        if embedding_questions:
            result = await self._session.execute(
                text("UPDATE chunk_vector SET questions = :q, embedding_questions = CAST(:emb AS vector) WHERE chunk_id = :cid"),
                {"q": questions, "emb": str(embedding_questions), "cid": chunk_id},
            )
        else:
            result = await self._session.execute(
                text("UPDATE chunk_vector SET questions = :q WHERE chunk_id = :cid"),
                {"q": questions, "cid": chunk_id},
            )
        # session.execute 静态类型为 Result，DML 实际返回 CursorResult 才有 rowcount
        return cast(CursorResult[Any], result).rowcount

    async def update_summary_embedding(self, chunk_id: str, embedding_summary: list[float]) -> None:
        await self._session.execute(
            text("UPDATE chunk_vector SET embedding_summary = CAST(:emb AS vector) WHERE chunk_id = :cid"),
            {"emb": str(embedding_summary), "cid": chunk_id},
        )
