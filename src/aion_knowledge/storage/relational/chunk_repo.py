"""ChunkRepository — chunk_text 表的读写封装。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import bindparam, text
from sqlalchemy.engine import Row
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class ChunkRow:
    id: str
    content: str
    document_id: str
    kb_id: str
    seq_num: int
    chunk_type: str = "text"
    chunk_metadata: dict = field(default_factory=dict)  # type: ignore[type-arg]


@dataclass
class FAQRow:
    id: str
    kb_id: str
    document_id: str
    content: str
    standard_question: str = ""
    similar_questions: list[str] = field(default_factory=list)
    negative_questions: list[str] = field(default_factory=list)
    answers: list[str] = field(default_factory=list)
    category: str = ""


class ChunkRepository:
    """chunk_text 表的数据库操作。"""

    def __init__(self, session: AsyncSession):
        self._session = session

    # ------------------------------------------------------------------ #
    # 读取方法
    # ------------------------------------------------------------------ #

    async def get_by_document(self, document_id: str) -> list[ChunkRow]:
        """按文档加载全部 chunk（含 parent/image 等类型，供二批统一处理）。"""
        rows = await self._session.execute(
            text("""
                SELECT id, content, document_id, kb_id, seq_num, chunk_type, metadata
                FROM chunk_text
                WHERE document_id = :doc_id
                ORDER BY seq_num
            """),
            {"doc_id": document_id},
        )
        return [self._row_to_chunk_row(r) for r in rows]

    async def get_by_kb(self, kb_id: str, chunk_type: str = "text") -> list[ChunkRow]:
        rows = await self._session.execute(
            text("""
                SELECT id, content, document_id, kb_id, seq_num, chunk_type, metadata
                FROM chunk_text
                WHERE kb_id = :kb_id AND chunk_type = :chunk_type
                ORDER BY seq_num
            """),
            {"kb_id": kb_id, "chunk_type": chunk_type},
        )
        return [self._row_to_chunk_row(r) for r in rows]

    async def get_by_id(self, chunk_id: str) -> ChunkRow | None:
        row = await self._session.execute(
            text("""
                SELECT id, content, document_id, kb_id, seq_num, chunk_type, metadata
                FROM chunk_text WHERE id = :cid
            """),
            {"cid": chunk_id},
        )
        r = row.first()
        return self._row_to_chunk_row(r) if r else None

    async def get_by_ids(self, chunk_ids: list[str]) -> list[ChunkRow]:
        """批量按 chunk_ids 查 chunk_text。空列表返回 []。"""
        if not chunk_ids:
            return []
        rows = await self._session.execute(
            text("""
                SELECT id, content, document_id, kb_id, seq_num, chunk_type, metadata
                FROM chunk_text
                WHERE id IN :ids
                  AND NOT EXISTS (SELECT 1 FROM doc_knowledge_documents d
                                  WHERE d.id = chunk_text.document_id AND d.deleted)
            """).bindparams(bindparam("ids", expanding=True)),
            {"ids": chunk_ids},
        )
        return [self._row_to_chunk_row(r) for r in rows]

    async def count_by_kb(self, kb_id: str) -> int:
        row = await self._session.execute(
            text("SELECT COUNT(*) FROM chunk_text WHERE kb_id = :kb_id"),
            {"kb_id": kb_id},
        )
        return row.scalar() or 0

    async def count_by_document(self, document_id: str) -> int:
        row = await self._session.execute(
            text("SELECT COUNT(*) FROM chunk_text WHERE document_id = :doc_id"),
            {"doc_id": document_id},
        )
        return row.scalar() or 0

    async def search_by_keyword(self, kb_id: str, keyword: str, limit: int) -> list[ChunkRow]:
        rows = await self._session.execute(
            text("""
                SELECT c.id, c.content, c.document_id, c.kb_id, c.seq_num, c.chunk_type, c.metadata
                FROM chunk_text c
                WHERE c.kb_id = :kb_id
                  AND EXISTS (
                      SELECT 1 FROM unnest(c.keywords) kw WHERE kw ILIKE :pattern
                  )
                LIMIT :limit
            """),
            {"kb_id": kb_id, "pattern": f"%{keyword}%", "limit": limit},
        )
        return [self._row_to_chunk_row(r) for r in rows]

    async def search_faq(self, kb_id: str, query: str) -> list[FAQRow]:
        query_lower = query.lower().strip()
        rows = await self._session.execute(
            text("""
                SELECT c.id, c.kb_id, c.document_id, c.content,
                       c.metadata::jsonb->>'standard_question' AS standard_question,
                       c.metadata::jsonb->>'similar_questions' AS similar_questions,
                       c.metadata::jsonb->>'negative_questions' AS negative_questions,
                       c.metadata::jsonb->>'answers' AS answers,
                       c.metadata::jsonb->>'category' AS category
                FROM chunk_text c
                JOIN doc_knowledge_documents d ON d.id = c.document_id AND NOT d.deleted
                JOIN kb_knowledge_bases k ON k.id = c.kb_id AND NOT k.deleted
                WHERE c.kb_id = :kb_id
                  AND c.chunk_type = 'faq'
                  AND (c.metadata::jsonb->>'standard_question' ILIKE :pattern
                       OR c.metadata::jsonb->>'similar_questions' ILIKE :pattern
                       OR c.content ILIKE :pattern)
            """),
            {"kb_id": kb_id, "pattern": f"%{query_lower}%"},
        )
        results: list[FAQRow] = []
        for r in rows:
            mapping = dict(r._mapping)
            similar_qs = _safe_json_list(mapping.get("similar_questions"))
            answers = _safe_json_list(mapping.get("answers"))
            negative_qs = _safe_json_list(mapping.get("negative_questions"))
            results.append(FAQRow(
                id=str(mapping["id"]),
                kb_id=str(mapping["kb_id"]),
                document_id=str(mapping["document_id"]),
                content=mapping["content"],
                standard_question=mapping.get("standard_question") or "",
                similar_questions=similar_qs,
                negative_questions=negative_qs,
                answers=answers,
                category=mapping.get("category") or "",
            ))
        return results

    # ------------------------------------------------------------------ #
    # 写入方法
    # ------------------------------------------------------------------ #

    async def update_keywords(self, chunk_id: str, keywords: list[str]) -> None:
        await self._session.execute(
            text("UPDATE chunk_text SET keywords = :kw WHERE id = :id"),
            {"kw": keywords, "id": chunk_id},
        )

    async def update_summary(self, chunk_id: str, summary_text: str) -> None:
        await self._session.execute(
            text("UPDATE chunk_text SET summary_text = :s WHERE id = :cid"),
            {"s": summary_text, "cid": chunk_id},
        )

    async def update_content_tokens(self, chunk_id: str) -> None:
        """更新 content_tokens（由 PG zhparser 从 content 计算）。"""
        await self._session.execute(
            text("""
                UPDATE chunk_text
                SET content_tokens = tsvector_to_array(
                        to_tsvector('zh_cfg', COALESCE(content, '')))
                WHERE id = :cid
            """),
            {"cid": chunk_id},
        )

    async def update_summary_tokens(self, chunk_id: str) -> None:
        """仅更新摘要分词结果（由 PG zhparser 从 summary_text 计算）。"""
        await self._session.execute(
            text("""
                UPDATE chunk_text
                SET summary_tokens = tsvector_to_array(
                        to_tsvector('zh_cfg', COALESCE(summary_text, '')))
                WHERE id = :cid
            """),
            {"cid": chunk_id},
        )

    # ------------------------------------------------------------------ #
    # 内部辅助
    # ------------------------------------------------------------------ #

    @staticmethod
    def _row_to_chunk_row(r: Row) -> ChunkRow:  # type: ignore[type-arg]
        mapping = dict(r._mapping)
        return ChunkRow(
            id=str(mapping["id"]),
            content=mapping["content"],
            document_id=str(mapping["document_id"]),
            kb_id=str(mapping["kb_id"]),
            seq_num=mapping["seq_num"],
            chunk_type=mapping.get("chunk_type", "text"),
            chunk_metadata=mapping.get("metadata") or {},
        )


def _safe_json_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    try:
        val = json.loads(raw) if isinstance(raw, str) else raw
        return list(val) if isinstance(val, list) else []
    except (json.JSONDecodeError, TypeError):
        return []
