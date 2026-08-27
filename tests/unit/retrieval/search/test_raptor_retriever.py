"""RAPTORRetriever 测试 — 树遍历检索：选树 → 逐层剪枝 → 叶子 top-N → 路径拼装。"""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aion_knowledge.retrieval.base import RetrieverContext
from aion_knowledge.retrieval.search.raptor_retriever import RAPTORRetriever


def _mock_session(root_rows: list, level_rows: list, leaf_rows: list) -> MagicMock:
    session = MagicMock()
    session.execute = AsyncMock(side_effect=[root_rows, *level_rows, *leaf_rows])
    return session


@pytest.mark.asyncio
async def test_retrieve_traverses_tree_and_assembles_paths():
    """命中应返回路径拼装：摘要链 + 叶子原文（top-N）。"""
    retriever = RAPTORRetriever()
    ctx = RetrieverContext(query="查询", kb_id="kb-1", query_embedding=[0.1, 0.2])

    roots = [
        SimpleNamespace(id="r1", doc_id="d1", layer=2, summary="根摘要",
                        source_chunk_ids=None),
    ]
    # 下一层：root 的摘要子节点（layer 1），via_parent 为下钻到它的遍历父
    level1 = [
        SimpleNamespace(id="n1", layer=1, summary="分支摘要", dist=0.1,
                        via_parent="r1",
                        source_chunk_ids=["c1", "c2", "c3", "c4", "c5", "c6"]),
    ]
    leaves = [
        SimpleNamespace(id=f"c{i}", content=f"叶子{i}")
        for i in range(1, 6)
    ]
    session = _mock_session(roots, [level1], [leaves])

    @asynccontextmanager
    async def mock_get_session():
        yield session

    with patch("aion_knowledge.retrieval.search.raptor_retriever.get_session", mock_get_session):
        results = await retriever.retrieve(ctx)

    assert len(results) == 1
    first = results[0]
    assert first.chunk_id == "n1"          # 路径末端摘要节点 id（chunk_raptor 空间）
    assert first.document_id == "d1"
    assert first.content.startswith("根摘要")
    assert "分支摘要" in first.content
    assert "叶子1" in first.content        # top-N 叶子已拼入
    assert first.source_paths == ["raptor"]
    assert first.metadata["source_chunk_ids"] == ["c1", "c2", "c3", "c4", "c5"]


@pytest.mark.asyncio
async def test_retrieve_chain_propagates_three_levels():
    """3 层树：content 应按 [root摘要, mid摘要, leaf摘要, 叶子原文] 顺序拼装。"""
    retriever = RAPTORRetriever()
    ctx = RetrieverContext(query="查询", kb_id="kb-1", query_embedding=[0.1, 0.2])

    roots = [
        SimpleNamespace(id="r1", doc_id="d1", layer=3, summary="根摘要",
                        source_chunk_ids=None, children_ids=["m1"]),
    ]
    mid = [
        SimpleNamespace(id="m1", layer=2, summary="中间摘要", dist=0.1,
                        via_parent="r1", source_chunk_ids=["c1", "c2"]),
    ]
    leaf = [
        SimpleNamespace(id="n1", layer=1, summary="叶子摘要", dist=0.2,
                        via_parent="m1", source_chunk_ids=["c1"]),
    ]
    leaves = [SimpleNamespace(id="c1", content="叶子原文")]
    session = _mock_session(roots, [mid, leaf], [leaves])

    @asynccontextmanager
    async def mock_get_session():
        yield session

    with patch("aion_knowledge.retrieval.search.raptor_retriever.get_session", mock_get_session):
        results = await retriever.retrieve(ctx)

    assert len(results) == 1
    first = results[0]
    assert first.chunk_id == "n1"
    assert first.document_id == "d1"       # 叶子行无 doc_id，沿遍历父链取 root 的
    assert first.content == "根摘要\n\n中间摘要\n\n叶子摘要\n\n叶子原文"
    assert first.metadata["source_chunk_ids"] == ["c1"]


@pytest.mark.asyncio
async def test_retrieve_sql_filters_and_params():
    """SQL 行为断言：选树过滤、unnest 下钻、叶子 JOIN、kb_id 参数。"""
    retriever = RAPTORRetriever()
    ctx = RetrieverContext(query="查询", kb_id="kb-1", query_embedding=[0.1, 0.2])

    roots = [
        SimpleNamespace(id="r1", doc_id="d1", layer=2, summary="根摘要",
                        source_chunk_ids=None),
    ]
    level1 = [
        SimpleNamespace(id="n1", layer=1, summary="分支摘要", dist=0.1,
                        via_parent="r1", source_chunk_ids=["c1"]),
    ]
    leaves = [SimpleNamespace(id="c1", content="叶子1")]
    session = _mock_session(roots, [level1], [leaves])

    @asynccontextmanager
    async def mock_get_session():
        yield session

    with patch("aion_knowledge.retrieval.search.raptor_retriever.get_session", mock_get_session):
        await retriever.retrieve(ctx)

    calls = session.execute.await_args_list
    assert len(calls) == 3

    root_sql, root_params = calls[0].args[0].text, calls[0].args[1]
    assert "parent_id IS NULL" in root_sql      # 只选 root
    assert "r.kb_id = :kb_id" in root_sql
    assert root_params["kb_id"] == "kb-1"

    level_sql, level_params = calls[1].args[0].text, calls[1].args[1]
    assert "unnest(p.children_ids)" in level_sql  # children_ids 下钻
    assert "DISTINCT ON (r.id)" in level_sql      # 按节点去重后再取 beam
    assert ":parent_ids" in level_sql
    assert "r.kb_id = :kb_id" in level_sql      # 防御性 kb 过滤
    assert "r.embedding IS NOT NULL" in level_sql
    assert level_params["parent_ids"] == ["r1"]
    assert level_params["kb_id"] == "kb-1"

    leaf_sql, leaf_params = calls[2].args[0].text, calls[2].args[1]
    assert "JOIN chunk_vector" in leaf_sql      # 经 chunk_vector 排序取 top-N
    assert "ct.kb_id = :kb_id" in leaf_sql
    assert leaf_params["kb_id"] == "kb-1"


@pytest.mark.asyncio
async def test_retrieve_shared_child_keeps_first_via_parent():
    """共享子节点多父下钻：同 id 行去重（SQL 层 DISTINCT ON，实现层先到先得），
    链保留最先到达的遍历父，不产生第二条路径。"""
    retriever = RAPTORRetriever()
    ctx = RetrieverContext(query="查询", kb_id="kb-1", query_embedding=[0.1, 0.2])

    roots = [
        SimpleNamespace(id="r1", doc_id="d1", layer=2, summary="根1摘要",
                        source_chunk_ids=None),
        SimpleNamespace(id="r2", doc_id="d2", layer=2, summary="根2摘要",
                        source_chunk_ids=None),
    ]
    # n1 同时是 r1、r2 的子节点（软聚类共享）：两行同 id、via_parent 不同
    level1 = [
        SimpleNamespace(id="n1", layer=1, summary="共享摘要", dist=0.1,
                        via_parent="r1", source_chunk_ids=["c1"]),
        SimpleNamespace(id="n1", layer=1, summary="共享摘要", dist=0.2,
                        via_parent="r2", source_chunk_ids=["c1"]),
    ]
    leaves = [SimpleNamespace(id="c1", content="叶子原文")]
    session = _mock_session(roots, [level1], [leaves])

    @asynccontextmanager
    async def mock_get_session():
        yield session

    with patch("aion_knowledge.retrieval.search.raptor_retriever.get_session", mock_get_session):
        results = await retriever.retrieve(ctx)

    assert len(results) == 1                    # 重复行不产生第二条路径
    first = results[0]
    assert first.chunk_id == "n1"
    assert first.content == "根1摘要\n\n共享摘要\n\n叶子原文"  # 链取最先到达的 r1


@pytest.mark.asyncio
async def test_retrieve_emits_single_node_tree_path():
    """退化树（root 即叶子，children_ids 为空）应直接发射 [root] 路径。"""
    retriever = RAPTORRetriever()
    ctx = RetrieverContext(query="查询", kb_id="kb-1", query_embedding=[0.1, 0.2])

    # 无 source_chunk_ids 属性：发射路径时缺省透传空列表，不执行叶子查询；
    # 下钻 level 查询仍会执行一次（frontier 非空），返回空即断链
    roots = [
        SimpleNamespace(id="r1", doc_id="d1", layer=1, summary="根摘要",
                        children_ids=[]),
    ]
    session = _mock_session(roots, [[]], [])

    @asynccontextmanager
    async def mock_get_session():
        yield session

    with patch("aion_knowledge.retrieval.search.raptor_retriever.get_session", mock_get_session):
        results = await retriever.retrieve(ctx)

    assert len(results) == 1
    first = results[0]
    assert first.chunk_id == "r1"
    assert first.document_id == "d1"
    assert first.content == "根摘要"
    assert first.source_paths == ["raptor"]
    assert first.metadata["source_chunk_ids"] == []


@pytest.mark.asyncio
async def test_retrieve_without_query_embedding_returns_empty():
    """无 query_embedding 返回 []（保持现状）。"""
    retriever = RAPTORRetriever()
    ctx = RetrieverContext(query="查询", kb_id="kb-1", query_embedding=None)
    assert await retriever.retrieve(ctx) == []
