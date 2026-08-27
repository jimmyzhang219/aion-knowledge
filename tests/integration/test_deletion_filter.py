"""检索过滤集成测试：已删除文档的 chunk 不应出现在文档级检索结果中（任务 5）。

直接插行（不跑 ingestion），连真实 dev 库：
- KB 1 个（uuid4），文档 2 个（doc_ok 正常 / doc_del deleted=true）
- 每个文档各插 1 条 chunk_text（+ 对应 chunk_vector 行）
- 断言 bm25 / keyword / summary / faq / 向量层均不含 doc_del 的 chunk

运行: cd <worktree> && PYTHONPATH=<worktree>/src pytest tests/integration/test_deletion_filter.py -v
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text as sql_text

from aion_knowledge.common.uuid7 import uuid7
from aion_knowledge.infrastructure.db import _engine, dispose_engine, get_session
from aion_knowledge.retrieval.base import RetrieverContext
from aion_knowledge.retrieval.search.bm25_retriever import BM25Retriever
from aion_knowledge.retrieval.search.community_retriever import CommunityRetriever
from aion_knowledge.retrieval.search.keyword_retriever import KeywordRetriever
from aion_knowledge.retrieval.search.raptor_retriever import RAPTORRetriever
from aion_knowledge.retrieval.search.summary_retriever import SummaryRetriever
from aion_knowledge.retrieval.search.wiki_retriever import WikiRetriever
from aion_knowledge.storage.relational.chunk_repo import ChunkRepository
from aion_knowledge.storage.relational.vector_repo import VectorRepository

pytestmark = pytest.mark.asyncio(loop_scope="session")

# chunk_vector.embedding 为 vector(1024)，固定维度
_DIM = 1024


async def _cleanup_kb(kb_id: str) -> None:
    """清理测试 KB 的全部关联数据（测试开头/结尾各调一次，幂等）。"""
    await dispose_engine()
    async with _engine.begin() as conn:
        await conn.execute(
            sql_text("DELETE FROM chunk_community WHERE kb_id = :kb_id"),
            {"kb_id": kb_id},
        )
        await conn.execute(
            sql_text("DELETE FROM chunk_wiki WHERE kb_id = :kb_id"),
            {"kb_id": kb_id},
        )
        await conn.execute(
            sql_text("DELETE FROM chunk_raptor WHERE kb_id = :kb_id"),
            {"kb_id": kb_id},
        )
        await conn.execute(
            sql_text("DELETE FROM chunk_vector WHERE kb_id = :kb_id"),
            {"kb_id": kb_id},
        )
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
            {"id": kb_id, "name": "Deletion-Filter-Test", "tags": [], "desc": "检索过滤测试"},
        )


async def _insert_doc(kb_id: str, doc_id: str, doc_name: str, deleted: bool) -> None:
    async with _engine.begin() as conn:
        await conn.execute(
            sql_text(
                "INSERT INTO doc_knowledge_documents "
                "(id, kb_id, doc_name, suffix, hash, size, status, tags, source_label, "
                " creator, file_path, chunk_strategy, created_at, updated_at, deleted) "
                "VALUES (:id, :kb_id, :doc_name, '', :hash, 0, 'pending', '{}', '', 'test', '', "
                " 'auto', now(), now(), :deleted)"
            ),
            {
                "id": doc_id,
                "kb_id": kb_id,
                "doc_name": doc_name,
                "hash": uuid.uuid4().hex,
                "deleted": deleted,
            },
        )


async def _insert_chunk(
    kb_id: str,
    doc_id: str,
    chunk_id: str,
    *,
    content: str,
    keywords: list[str] | None = None,
    summary_text: str = "",
    chunk_type: str = "text",
    metadata_json: str = "{}",
) -> None:
    """插入 chunk_text 行。content_tokens 由 zhparser 从 content 计算（BM25 检索依赖）。"""
    async with _engine.begin() as conn:
        await conn.execute(
            sql_text(
                "INSERT INTO chunk_text "
                "(id, document_id, kb_id, content, context_header, keywords, seq_num, chunk_type, "
                " token_count, metadata, image_refs, summary_text, content_tokens, summary_tokens, created_at) "
                "VALUES (:id, :doc_id, :kb_id, :content, '', :keywords, 0, :chunk_type, 0, "
                " CAST(:metadata AS json), '{}', :summary_text, "
                " tsvector_to_array(to_tsvector('zh_cfg', :content)), '{}', now())"
            ),
            {
                "id": chunk_id,
                "doc_id": doc_id,
                "kb_id": kb_id,
                "content": content,
                "keywords": keywords or [],
                "chunk_type": chunk_type,
                "metadata": metadata_json,
                "summary_text": summary_text,
            },
        )


async def _insert_vector(kb_id: str, chunk_id: str, embedding: list[float]) -> None:
    async with _engine.begin() as conn:
        await conn.execute(
            sql_text(
                "INSERT INTO chunk_vector (id, chunk_id, kb_id, embedding, embedding_summary, payload) "
                "VALUES (:id, :chunk_id, :kb_id, CAST(:emb AS vector), "
                " CAST(:emb AS vector), '{}'::json)"
            ),
            {"id": uuid7(), "chunk_id": chunk_id, "kb_id": kb_id, "emb": str(embedding)},
        )


async def _insert_community(
    kb_id: str,
    chunk_id: str,
    embedding: list[float],
    *,
    summary: str = "hello community summary",
) -> None:
    """插入 1 条 chunk_community 行（KB 级，无文档归属）。"""
    async with _engine.begin() as conn:
        await conn.execute(
            sql_text(
                "INSERT INTO chunk_community "
                "(id, chunk_id, kb_id, community_id, community_level, summary, findings, "
                " embedding, payload) "
                "VALUES (:id, :chunk_id, :kb_id, 'c0', 0, :summary, '{}'::json, "
                " CAST(:emb AS vector), '{}'::json)"
            ),
            {
                "id": uuid7(),
                "chunk_id": chunk_id,
                "kb_id": kb_id,
                "summary": summary,
                "emb": str(embedding),
            },
        )


async def _insert_wiki(
    kb_id: str,
    chunk_id: str,
    *,
    page_slug: str = "deletion-filter-test",
    page_title: str = "hello wiki title",
    content: str = "hello wiki content",
) -> None:
    """插入 1 条 chunk_wiki 页面行（KB 级页面池，引用指定 chunk）。"""
    async with _engine.begin() as conn:
        await conn.execute(
            sql_text(
                "INSERT INTO chunk_wiki "
                "(id, kb_id, page_slug, page_title, content, taxonomy_path, status, payload, "
                " chunk_refs, source_refs, out_links, in_links) "
                "VALUES (:id, :kb_id, :slug, :title, :content, '', 'published', '{}'::json, "
                " CAST(:chunk_refs AS text[]), '{}', '{}', '{}')"
            ),
            {
                "id": uuid7(),
                "kb_id": kb_id,
                "slug": page_slug,
                "title": page_title,
                "content": content,
                "chunk_refs": [chunk_id],
            },
        )


async def _insert_raptor_root(
    kb_id: str,
    embedding: list[float],
    *,
    doc_id: str | None = None,
) -> str:
    """插入 1 条 raptor 树根（parent_id NULL，退化树）。

    doc_id 为空 = KB 级树，非空 = 文档级树。返回节点 id。
    children_ids 留默认 '{}'（退化树 → Stage 1 直接发射路径）。
    """
    node_id = str(uuid7())
    async with _engine.begin() as conn:
        await conn.execute(
            sql_text(
                "INSERT INTO chunk_raptor "
                "(id, kb_id, doc_id, parent_id, title, summary, layer, cluster_id, "
                " source_chunk_ids, tree_builder, clustering_method, output_mode, payload, embedding) "
                "VALUES (:id, :kb_id, :doc_id, NULL, '', :summary, 1, '', '{}', "
                " 'raptor', 'gmm', 'flat', '{}'::json, CAST(:emb AS vector))"
            ),
            {
                "id": node_id,
                "kb_id": kb_id,
                "doc_id": doc_id,
                "summary": "raptor tree summary",
                "emb": str(embedding),
            },
        )
    return node_id


def _assert_contains_ok_not_del(results, ok_id: str, del_id: str) -> None:
    ids = [r.chunk_id for r in results]
    assert ok_id in ids, f"doc_ok 的 chunk {ok_id} 应出现在结果中，实际: {ids}"
    assert del_id not in ids, f"已删文档的 chunk {del_id} 不应出现在结果中，实际: {ids}"


async def _setup_pair(kb_id: str) -> tuple[str, str, str, str]:
    """插入 1 个 KB + 2 个文档（doc_ok / doc_del）+ 各 1 条 text chunk。

    返回 (doc_ok, doc_del, chunk_ok, chunk_del)。
    """
    doc_ok = str(uuid.uuid4())
    doc_del = str(uuid.uuid4())
    chunk_ok = str(uuid7())
    chunk_del = str(uuid7())
    await _insert_kb(kb_id)
    await _insert_doc(kb_id, doc_ok, "ok.md", deleted=False)
    await _insert_doc(kb_id, doc_del, "del.md", deleted=True)
    return doc_ok, doc_del, chunk_ok, chunk_del


async def test_bm25_excludes_deleted_document_chunks() -> None:
    """bm25 检索：已删文档的 chunk 不应出现。"""
    kb_id = str(uuid.uuid4())
    await _cleanup_kb(kb_id)
    try:
        _doc_ok, _doc_del, chunk_ok, chunk_del = await _setup_pair(kb_id)
        await _insert_chunk(kb_id, _doc_ok, chunk_ok, content="hello retrieval filter test")
        await _insert_chunk(kb_id, _doc_del, chunk_del, content="hello retrieval deleted doc")

        results = await BM25Retriever().retrieve(
            RetrieverContext(query="hello", kb_id=kb_id, top_k=10)
        )
        _assert_contains_ok_not_del(results, chunk_ok, chunk_del)
    finally:
        await _cleanup_kb(kb_id)


async def test_keyword_excludes_deleted_document_chunks() -> None:
    """keyword 检索：已删文档的 chunk 不应出现。"""
    kb_id = str(uuid.uuid4())
    await _cleanup_kb(kb_id)
    try:
        _doc_ok, _doc_del, chunk_ok, chunk_del = await _setup_pair(kb_id)
        await _insert_chunk(kb_id, _doc_ok, chunk_ok, content="人工智能 智能问答", keywords=["人工智能"])
        await _insert_chunk(kb_id, _doc_del, chunk_del, content="人工智能 智能问答", keywords=["人工智能"])

        results = await KeywordRetriever().retrieve(
            RetrieverContext(query="人工智能", kb_id=kb_id, top_k=10)
        )
        _assert_contains_ok_not_del(results, chunk_ok, chunk_del)
    finally:
        await _cleanup_kb(kb_id)


async def test_summary_excludes_deleted_document_chunks() -> None:
    """summary 检索（BM25 + 向量 + 回填）：已删文档的 chunk 不应出现。"""
    kb_id = str(uuid.uuid4())
    await _cleanup_kb(kb_id)
    try:
        _doc_ok, _doc_del, chunk_ok, chunk_del = await _setup_pair(kb_id)
        await _insert_chunk(
            kb_id, _doc_ok, chunk_ok, content="ok content",
            summary_text="hello summary text",
        )
        await _insert_chunk(
            kb_id, _doc_del, chunk_del, content="del content",
            summary_text="hello summary text",
        )
        # 两个 chunk 的 embedding_summary 都写入（1024 维），query_embedding 同时命中两路
        await _insert_vector(kb_id, chunk_ok, [0.0] * (_DIM - 1) + [1.0])
        await _insert_vector(kb_id, chunk_del, [1.0] + [0.0] * (_DIM - 1))

        results = await SummaryRetriever().retrieve(
            RetrieverContext(query="summary", kb_id=kb_id, top_k=10,
                             query_embedding=[0.5] * _DIM)
        )
        _assert_contains_ok_not_del(results, chunk_ok, chunk_del)
    finally:
        await _cleanup_kb(kb_id)


async def test_vector_search_excludes_deleted_document_chunks() -> None:
    """向量层（vector_repo.similarity_search）：已删文档的 chunk 不应出现。"""
    kb_id = str(uuid.uuid4())
    await _cleanup_kb(kb_id)
    try:
        _doc_ok, _doc_del, chunk_ok, chunk_del = await _setup_pair(kb_id)
        await _insert_chunk(kb_id, _doc_ok, chunk_ok, content="ok content")
        await _insert_chunk(kb_id, _doc_del, chunk_del, content="del content")
        await _insert_vector(kb_id, chunk_ok, [0.0] * (_DIM - 1) + [1.0])
        await _insert_vector(kb_id, chunk_del, [1.0] + [0.0] * (_DIM - 1))

        async with get_session() as session:
            results = await VectorRepository(session).similarity_search(
                kb_id=kb_id, embedding=[0.5] * _DIM, top_k=10
            )
        ids = [r.chunk_id for r in results]
        assert chunk_ok in ids, f"doc_ok 的 chunk 应出现在向量结果中，实际: {ids}"
        assert chunk_del not in ids, f"已删文档的 chunk 不应出现在向量结果中，实际: {ids}"
    finally:
        await _cleanup_kb(kb_id)


async def test_faq_search_excludes_deleted_document() -> None:
    """faq 检索（search_faq）：已删文档的 faq chunk 不应出现。"""
    kb_id = str(uuid.uuid4())
    await _cleanup_kb(kb_id)
    try:
        _doc_ok, _doc_del, chunk_ok, chunk_del = await _setup_pair(kb_id)
        await _insert_chunk(
            kb_id, _doc_ok, chunk_ok, content="faq content", chunk_type="faq",
            metadata_json=('{"standard_question": "hello faq question", '
                           '"similar_questions": [], "negative_questions": [], '
                           '"answers": [], "category": ""}'),
        )
        await _insert_chunk(
            kb_id, _doc_del, chunk_del, content="faq content", chunk_type="faq",
            metadata_json=('{"standard_question": "hello faq question", '
                           '"similar_questions": [], "negative_questions": [], '
                           '"answers": [], "category": ""}'),
        )

        async with get_session() as session:
            rows = await ChunkRepository(session).search_faq(kb_id, "hello faq")
        ids = [str(r.id) for r in rows]
        assert chunk_ok in ids, f"doc_ok 的 faq 应出现在结果中，实际: {ids}"
        assert chunk_del not in ids, f"已删文档的 faq 不应出现在结果中，实际: {ids}"
    finally:
        await _cleanup_kb(kb_id)


async def test_kb_deleted_blocks_all_paths() -> None:
    """KB deleted=true：bm25 / community / wiki 均返回空（KB 级 join 生效）。"""
    kb_id = str(uuid.uuid4())
    await _cleanup_kb(kb_id)
    try:
        doc_ok, _doc_del, chunk_ok, _chunk_del = await _setup_pair(kb_id)
        await _insert_chunk(kb_id, doc_ok, chunk_ok, content="hello retrieval filter test")
        await _insert_community(kb_id, chunk_ok, [0.0] * (_DIM - 1) + [1.0])
        await _insert_wiki(kb_id, chunk_ok)

        # 前置断言：KB 未删时三路均应命中（确保数据可见，空结果才有意义）
        bm25_before = await BM25Retriever().retrieve(
            RetrieverContext(query="hello", kb_id=kb_id, top_k=10)
        )
        assert chunk_ok in [r.chunk_id for r in bm25_before], \
            f"KB 未删时 bm25 应命中，实际: {[r.chunk_id for r in bm25_before]}"
        comm_before = await CommunityRetriever().retrieve(
            RetrieverContext(query="hello", kb_id=kb_id, top_k=10,
                             query_embedding=[0.5] * _DIM)
        )
        assert comm_before, "KB 未删时 community 应命中"
        wiki_before = await WikiRetriever().retrieve(
            RetrieverContext(query="hello", kb_id=kb_id, top_k=10)
        )
        assert chunk_ok in [r.chunk_id for r in wiki_before], \
            f"KB 未删时 wiki 应命中，实际: {[r.chunk_id for r in wiki_before]}"

        # 标记 KB 逻辑删除
        async with _engine.begin() as conn:
            await conn.execute(
                sql_text("UPDATE kb_knowledge_bases SET deleted = true WHERE id = :kb_id"),
                {"kb_id": kb_id},
            )

        bm25_after = await BM25Retriever().retrieve(
            RetrieverContext(query="hello", kb_id=kb_id, top_k=10)
        )
        assert bm25_after == [], f"KB 已删时 bm25 应返回空，实际: {bm25_after}"
        comm_after = await CommunityRetriever().retrieve(
            RetrieverContext(query="hello", kb_id=kb_id, top_k=10,
                             query_embedding=[0.5] * _DIM)
        )
        assert comm_after == [], f"KB 已删时 community 应返回空，实际: {comm_after}"
        wiki_after = await WikiRetriever().retrieve(
            RetrieverContext(query="hello", kb_id=kb_id, top_k=10)
        )
        assert wiki_after == [], f"KB 已删时 wiki 应返回空，实际: {wiki_after}"
    finally:
        await _cleanup_kb(kb_id)


async def test_kb_deleted_blocks_raptor_kb_level_tree() -> None:
    """KB deleted=true：KB 级 raptor 树（doc_id NULL）不应再被选中（Stage 1/2 KB 过滤）。"""
    kb_id = str(uuid.uuid4())
    await _cleanup_kb(kb_id)
    try:
        await _insert_kb(kb_id)
        await _insert_raptor_root(kb_id, [0.0] * (_DIM - 1) + [1.0])

        # 前置断言：KB 未删时 KB 级树根应被选中（退化树 → Stage 1 直接发射）
        raptor_before = await RAPTORRetriever().retrieve(
            RetrieverContext(query="raptor", kb_id=kb_id, top_k=10,
                             query_embedding=[0.5] * _DIM)
        )
        assert len(raptor_before) == 1, f"KB 未删时 raptor 应命中 KB 级树，实际: {raptor_before}"

        # 标记 KB 逻辑删除
        async with _engine.begin() as conn:
            await conn.execute(
                sql_text("UPDATE kb_knowledge_bases SET deleted = true WHERE id = :kb_id"),
                {"kb_id": kb_id},
            )

        raptor_after = await RAPTORRetriever().retrieve(
            RetrieverContext(query="raptor", kb_id=kb_id, top_k=10,
                             query_embedding=[0.5] * _DIM)
        )
        assert raptor_after == [], f"KB 已删时 raptor 应返回空，实际: {raptor_after}"
    finally:
        await _cleanup_kb(kb_id)


async def test_raptor_excludes_deleted_document_tree() -> None:
    """文档 deleted=true：该文档的 raptor 树不参与检索（Stage 1/2 doc 过滤）。

    对照组：正常文档的树仍命中；已删文档的树从 Stage 1 即被过滤。
    """
    kb_id = str(uuid.uuid4())
    await _cleanup_kb(kb_id)
    try:
        doc_ok, doc_del, _chunk_ok, _chunk_del = await _setup_pair(kb_id)
        root_ok = await _insert_raptor_root(kb_id, [0.0] * (_DIM - 1) + [1.0], doc_id=doc_ok)
        await _insert_raptor_root(kb_id, [0.0] * (_DIM - 1) + [1.0], doc_id=doc_del)

        # 前置断言：doc_ok 的树命中；doc_del（deleted=true）的树不出现
        before = await RAPTORRetriever().retrieve(
            RetrieverContext(query="raptor", kb_id=kb_id, top_k=10,
                             query_embedding=[0.5] * _DIM)
        )
        assert [r.chunk_id for r in before] == [root_ok], \
            f"doc_ok 的树应命中、doc_del 的树不应出现，实际: {[r.chunk_id for r in before]}"

        # doc_ok 也标记删除 → 两棵文档级树均被过滤
        async with _engine.begin() as conn:
            await conn.execute(
                sql_text("UPDATE doc_knowledge_documents SET deleted = true WHERE id = :doc_id"),
                {"doc_id": doc_ok},
            )

        after = await RAPTORRetriever().retrieve(
            RetrieverContext(query="raptor", kb_id=kb_id, top_k=10,
                             query_embedding=[0.5] * _DIM)
        )
        assert after == [], f"文档删除后 raptor 应返回空，实际: {after}"
    finally:
        await _cleanup_kb(kb_id)
