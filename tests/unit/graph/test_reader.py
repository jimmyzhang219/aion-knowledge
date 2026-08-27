"""Neo4j 读取操作单元测试。"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from aion_knowledge.infrastructure.graph.reader import (
    expand_neighbors,
    load_kb_graph,
    search_entities,
)


@pytest.mark.asyncio
async def test_search_entities_empty():
    """空名称列表返回空列表。"""
    with patch(
        "aion_knowledge.infrastructure.graph.client.Neo4jConnection.execute_read",
        AsyncMock(return_value=[]),
    ):
        result = await search_entities("kb_test", [])
    assert result == []


@pytest.mark.asyncio
async def test_search_entities_with_names():
    """正常实体搜索。"""
    mock_results = [
        {"entity_name": "马云", "entity_type": "PERSON",
         "description": "创始人", "weight": 1.0, "similarity": 1.0},
    ]
    with patch(
        "aion_knowledge.infrastructure.graph.client.Neo4jConnection.execute_read",
        AsyncMock(return_value=mock_results),
    ):
        result = await search_entities("kb_test", ["马云"])
        assert result == mock_results


@pytest.mark.asyncio
async def test_expand_neighbors_empty():
    """空名称列表返回空列表。"""
    with patch(
        "aion_knowledge.infrastructure.graph.client.Neo4jConnection.execute_read",
        AsyncMock(return_value=[]),
    ):
        result = await expand_neighbors("kb_test", [])
    assert result == []


@pytest.mark.asyncio
async def test_expand_neighbors_with_names():
    """正常邻居展开。"""
    mock_results = [
        {"source": "马云", "target": "阿里巴巴",
         "type": "founder", "description": "创立了", "weight": 1.0},
    ]
    with patch(
        "aion_knowledge.infrastructure.graph.client.Neo4jConnection.execute_read",
        AsyncMock(return_value=mock_results),
    ):
        result = await expand_neighbors("kb_test", ["马云"])
        assert result == mock_results


@pytest.mark.asyncio
async def test_load_kb_graph_uses_relation_instance():
    """图谱读取使用两阶段模型（RELATION_INSTANCE），不再查询旧 RELATION 边。"""
    submitted: list[str] = []

    async def fake_execute_read(func, *args, **kwargs):
        class FakeTx:
            async def run(self, query, **params):
                submitted.append(query)
                result = AsyncMock()
                if "RELATION_INSTANCE" in query:
                    result.fetch.return_value = [
                        {"source": "A", "target": "B", "type": "link", "weight": 3.0},
                    ]
                else:
                    result.fetch.return_value = [
                        {"name": "A", "type": "ORG"},
                        {"name": "B", "type": "ORG"},
                    ]
                return result

        return await func(FakeTx(), *args, **kwargs)

    with patch("aion_knowledge.infrastructure.graph.reader.Neo4jConnection") as mock_cls:
        mock_cls.return_value.execute_read = AsyncMock(side_effect=fake_execute_read)
        entities, relations = await load_kb_graph("kb1")

    assert any("RELATION_INSTANCE" in q for q in submitted), "应查询 RELATION_INSTANCE 边"
    assert all("-[r:RELATION]->" not in q for q in submitted), "不应再查询旧 RELATION 边"
    rel_query = next(q for q in submitted if "RELATION_INSTANCE" in q)
    assert "SUM(r.weight) AS weight" in rel_query, "跨文档同实体对关系应按权重聚合"
    assert entities == [
        {"entity_name": "A", "entity_type": "ORG"},
        {"entity_name": "B", "entity_type": "ORG"},
    ]
    assert relations == [
        {"source_entity": "A", "target_entity": "B", "relation_type": "link", "weight": 3.0},
    ]


@pytest.mark.asyncio
async def test_search_entities_matches_aliases():
    """精确匹配应反查 Entity.aliases（别名也能命中 canonical 实体）。"""
    submitted: list[str] = []

    async def fake_execute_read(func, *args, **kwargs):
        class FakeResult:
            async def fetch(self, limit):
                return []

        class FakeTx:
            async def run(self, query, **params):
                submitted.append(query)
                return FakeResult()

        return await func(FakeTx(), *args, **kwargs)

    with patch("aion_knowledge.infrastructure.graph.reader.Neo4jConnection") as mock_cls:
        mock_cls.return_value.execute_read = AsyncMock(side_effect=fake_execute_read)
        await search_entities("kb1", ["iPhone"], top_k=10)

    exact = next(q for q in submitted if "CONTAINS" not in q)
    assert "any(a IN e.aliases WHERE a IN $names)" in exact, \
        "精确匹配应含别名反查条件"
    assert "e.name IN $names" in exact, "原名精确匹配应保留"


@pytest.mark.asyncio
async def test_expand_neighbors_matches_aliases():
    """邻居展开同样支持别名反查。"""
    submitted: list[str] = []

    async def fake_execute_read(func, *args, **kwargs):
        class FakeResult:
            async def fetch(self, limit):
                return []

        class FakeTx:
            async def run(self, query, **params):
                submitted.append(query)
                return FakeResult()

        return await func(FakeTx(), *args, **kwargs)

    with patch("aion_knowledge.infrastructure.graph.reader.Neo4jConnection") as mock_cls:
        mock_cls.return_value.execute_read = AsyncMock(side_effect=fake_execute_read)
        await expand_neighbors("kb1", ["iPhone"])

    assert any("e.aliases" in q for q in submitted), "展开查询应含别名反查条件"


@pytest.mark.asyncio
async def test_search_entities_returns_chunk_ids():
    """search_entities 应返回聚合后的 chunk_ids。"""
    async def fake_execute_read(func, *args, **kwargs):
        class FakeResult:
            async def fetch(self, limit):
                return [{"name": "马云", "type": "PERSON", "description": "创始人",
                         "weight": 1.0, "chunk_ids": ["c1", "c2"]}]
        class FakeTx:
            async def run(self, query, **params):
                return FakeResult()
        return await func(FakeTx(), *args, **kwargs)

    with patch("aion_knowledge.infrastructure.graph.reader.Neo4jConnection") as mock_cls:
        mock_cls.return_value.execute_read = AsyncMock(side_effect=fake_execute_read)
        result = await search_entities("kb1", ["马云"], top_k=1)

    assert result and result[0]["chunk_ids"] == ["c1", "c2"]


@pytest.mark.asyncio
async def test_search_entities_aggregates_chunk_ids_across_instances():
    """同一实体多个 instance 的 chunk_ids 应跨 instance 聚合去重。"""
    async def fake_execute_read(func, *args, **kwargs):
        class FakeResult:
            async def fetch(self, limit):
                return [
                    {"name": "马云", "type": "PERSON", "description": "创始人",
                     "weight": 1.0, "chunk_ids": ["c1", "c2"]},
                    {"name": "马云", "type": "PERSON", "description": "CEO",
                     "weight": 1.0, "chunk_ids": ["c2", "c3"]},
                ]
        class FakeTx:
            async def run(self, query, **params):
                return FakeResult()
        return await func(FakeTx(), *args, **kwargs)

    with patch("aion_knowledge.infrastructure.graph.reader.Neo4jConnection") as mock_cls:
        mock_cls.return_value.execute_read = AsyncMock(side_effect=fake_execute_read)
        result = await search_entities("kb1", ["马云"], top_k=2)

    assert len(result) == 1, "两条同名 record 应聚合为一个实体"
    assert result[0]["chunk_ids"] == ["c1", "c2", "c3"], \
        "跨 instance 的 chunk_ids 应去重合并（c2 不重复）"
