"""删除功能端到端测试：插数 → 检索可见 → 逻辑删除 → 检索不可见 → 真实删除 → 数据清空。

真实完整链路（不跑 ingestion，SQL 直插控制耗时），连真实 dev 库：
- 逻辑删除（delete_document / delete_kb，deletion_logical 默认 true 只置标记）：
  检索被屏蔽但数据行保留
- 真实删除（_purge_document 直接调用）：该文档全部关联数据清空

运行: cd <worktree> && PYTHONPATH=<worktree>/src pytest tests/integration/test_deletion_e2e.py -v
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text as sql_text

from aion_knowledge.infrastructure.db import _engine, dispose_engine
from aion_knowledge.ingestion.deletion import _purge_document, delete_document, delete_kb
from aion_knowledge.retrieval.base import RetrieverContext
from aion_knowledge.retrieval.search.bm25_retriever import BM25Retriever

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def _cleanup_kb(kb_id: str) -> None:
    """清理测试 KB 的全部关联数据（测试开头/结尾各调一次，幂等）。"""
    await dispose_engine()
    async with _engine.begin() as conn:
        await conn.execute(
            sql_text("DELETE FROM chunk_text WHERE kb_id = :kb_id"),
            {"kb_id": kb_id},
        )
        await conn.execute(
            sql_text("DELETE FROM doc_knowledge_documents WHERE kb_id = :kb_id"),
            {"kb_id": kb_id},
        )
        await conn.execute(
            sql_text("DELETE FROM kb_knowledge_bases WHERE id = :kb_id"),
            {"kb_id": kb_id},
        )


async def _insert_kb(kb_id: str) -> None:
    async with _engine.begin() as conn:
        await conn.execute(
            sql_text(
                "INSERT INTO kb_knowledge_bases (id, name, tags, description, created_at, updated_at) "
                "VALUES (:id, :name, :tags, :desc, now(), now())"
            ),
            {"id": kb_id, "name": "Deletion-E2E-Test", "tags": [], "desc": "删除功能端到端测试"},
        )


async def _insert_doc(kb_id: str, doc_id: str) -> None:
    async with _engine.begin() as conn:
        await conn.execute(
            sql_text(
                "INSERT INTO doc_knowledge_documents "
                "(id, kb_id, doc_name, suffix, hash, size, status, tags, source_label, "
                " creator, file_path, chunk_strategy, created_at, updated_at, deleted) "
                "VALUES (:id, :kb_id, 'e2e.md', '', :hash, 1, 'completed', '{}', '', 'test', '', "
                " 'auto', now(), now(), false)"
            ),
            {"id": doc_id, "kb_id": kb_id, "hash": uuid.uuid4().hex},
        )


async def _insert_chunk(kb_id: str, doc_id: str, chunk_id: str, content: str) -> None:
    """插入 chunk_text 行。content_tokens 由 zhparser 从 content 计算（BM25 检索依赖）。"""
    async with _engine.begin() as conn:
        await conn.execute(
            sql_text(
                "INSERT INTO chunk_text "
                "(id, document_id, kb_id, content, context_header, keywords, seq_num, chunk_type, "
                " token_count, metadata, image_refs, summary_text, content_tokens, summary_tokens, created_at) "
                "VALUES (:id, :doc_id, :kb_id, :content, '', '{}', 0, 'text', 0, "
                " '{}'::json, '{}', '', "
                " tsvector_to_array(to_tsvector('zh_cfg', :content)), '{}', now())"
            ),
            {"id": chunk_id, "doc_id": doc_id, "kb_id": kb_id, "content": content},
        )


async def _mk_kb_doc_chunk() -> tuple[str, str, str]:
    """建 KB + 文档 + chunk（SQL 直插），返回 (kb_id, doc_id, chunk_id)。"""
    kb_id = str(uuid.uuid4())
    doc_id = str(uuid.uuid4())
    chunk_id = str(uuid.uuid4())
    await _cleanup_kb(kb_id)
    await _insert_kb(kb_id)
    await _insert_doc(kb_id, doc_id)
    await _insert_chunk(kb_id, doc_id, chunk_id, "e2e 检索内容")
    return kb_id, doc_id, chunk_id


async def _bm25_chunk_ids(kb_id: str) -> list[str]:
    """bm25 检索返回 chunk_id 列表。"""
    results = await BM25Retriever().retrieve(
        RetrieverContext(query="检索", kb_id=kb_id, top_k=10)
    )
    return [r.chunk_id for r in results]


async def _count_rows(table: str, **where) -> int:
    """查库行数，where 为 (列名, 值) 对。"""
    async with _engine.begin() as conn:
        cond = " AND ".join(f"{col} = :{col}" for col in where)
        result = await conn.execute(
            sql_text(f"SELECT COUNT(*) FROM {table} WHERE {cond}"), where
        )
        return int(result.scalar())


async def test_logical_delete_hides_doc_from_search() -> None:
    """逻辑删除文档：检索屏蔽、文档行 deleted=true、chunk 数据保留。"""
    kb_id, doc_id, chunk_id = await _mk_kb_doc_chunk()
    try:
        # 前置断言：删除前数据可见（空结果才有意义）
        before_ids = await _bm25_chunk_ids(kb_id)
        assert chunk_id in before_ids, f"逻辑删除前 bm25 应命中该 chunk，实际: {before_ids}"

        ok = await delete_document(kb_id, doc_id)
        assert ok is True, "delete_document 应返回 True"

        # 检索屏蔽
        assert await _bm25_chunk_ids(kb_id) == [], "逻辑删除后 bm25 应返回空"

        # 数据保留：文档行 deleted=true，chunk_text 未删
        async with _engine.begin() as conn:
            row = await conn.execute(
                sql_text("SELECT deleted FROM doc_knowledge_documents WHERE id = :doc"),
                {"doc": doc_id},
            )
            assert row.scalar() is True, "文档行 deleted 应为 true"
        assert await _count_rows("chunk_text", kb_id=kb_id) == 1, \
            "逻辑删除不应删除 chunk 数据"
    finally:
        await _cleanup_kb(kb_id)


async def test_kb_logical_delete_hides_everything() -> None:
    """逻辑删除 KB：KB 及其下文档全部屏蔽检索。"""
    kb_id, doc_id, chunk_id = await _mk_kb_doc_chunk()
    try:
        # 前置断言：删除前数据可见
        before_ids = await _bm25_chunk_ids(kb_id)
        assert chunk_id in before_ids, f"KB 删除前 bm25 应命中该 chunk，实际: {before_ids}"

        ok = await delete_kb(kb_id)
        assert ok is True, "delete_kb 应返回 True"

        # 检索屏蔽
        assert await _bm25_chunk_ids(kb_id) == [], "KB 逻辑删除后 bm25 应返回空"

        # KB 行与文档行均置 deleted=true
        async with _engine.begin() as conn:
            kb_row = await conn.execute(
                sql_text("SELECT deleted FROM kb_knowledge_bases WHERE id = :kb"),
                {"kb": kb_id},
            )
            doc_row = await conn.execute(
                sql_text("SELECT deleted FROM doc_knowledge_documents WHERE id = :doc"),
                {"doc": doc_id},
            )
        assert kb_row.scalar() is True, "KB 行 deleted 应为 true"
        assert doc_row.scalar() is True, "KB 下文档行 deleted 应为 true"
    finally:
        await _cleanup_kb(kb_id)


async def test_physical_delete_clears_data() -> None:
    """真实删除（_purge_document）：chunk 与文档行全部清空。"""
    kb_id, doc_id, chunk_id = await _mk_kb_doc_chunk()
    try:
        # 前置断言：purge 前数据可见
        before_ids = await _bm25_chunk_ids(kb_id)
        assert chunk_id in before_ids, f"purge 前 bm25 应命中该 chunk，实际: {before_ids}"

        await _purge_document(kb_id, doc_id)

        # 数据清空：chunk_text 该文档行数 = 0、文档行 = 0
        assert await _count_rows("chunk_text", kb_id=kb_id) == 0, \
            "purge 后 chunk_text 应清空"
        assert await _count_rows("doc_knowledge_documents", id=doc_id) == 0, \
            "purge 后文档行应删除"
    finally:
        await _cleanup_kb(kb_id)
