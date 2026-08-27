"""Neo4j 写入操作单元测试。"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from aion_knowledge.infrastructure.graph.writer import (
    add_graph,
    delete_document_graph,
    delete_graph,
    get_stats,
    merge_aliases,
    merge_entities,
)


@pytest.fixture
def captured_execute_write():
    """捕获提交 CQL 与参数的 execute_write 假实现。"""
    submitted: list[tuple[str, dict[str, str]]] = []

    async def fake_execute_write(func, *args, **kwargs):
        class FakeResult:
            """假查询结果：single 返回 None（无实例/无冲突），fetch 返回空。"""

            async def single(self):
                return None

            async def fetch(self, limit=None):
                return []

        class FakeTx:
            async def run(self, query, **params):
                submitted.append((query, params))
                return FakeResult()

        await func(FakeTx(), *args, **kwargs)

    with patch(
        "aion_knowledge.infrastructure.graph.client.Neo4jConnection.execute_write",
        side_effect=fake_execute_write,
    ):
        yield submitted


@pytest.mark.asyncio
async def test_add_graph_empty():
    """空实体/关系不应报错（含 doc_id 参数）。"""
    result = await add_graph("kb_test", "doc_test", {}, {})
    assert result is None


@pytest.mark.asyncio
async def test_add_graph_with_doc_id():
    """正常实体/关系写入（带 doc_id）。"""
    entities = {
        "马云": {"type": "PERSON", "descriptions": ["创始人"]},
        "阿里巴巴": {"type": "ORGANIZATION", "descriptions": ["公司"]},
    }
    relations = {
        ("马云", "阿里巴巴", "founder"): {"descriptions": ["创立了"], "weight": 1.0},
    }

    with patch(
        "aion_knowledge.infrastructure.graph.client.Neo4jConnection.execute_write",
        AsyncMock(return_value=None),
    ):
        result = await add_graph("kb_test", "doc_1", entities, relations)
        assert result is None


@pytest.mark.asyncio
async def test_add_graph_only_entities():
    """只有实体没有关系不应报错。"""
    entities = {"马云": {"type": "PERSON", "descriptions": ["创始人"]}}
    with patch(
        "aion_knowledge.infrastructure.graph.client.Neo4jConnection.execute_write",
        AsyncMock(return_value=None),
    ):
        result = await add_graph("kb_test", "doc_1", entities, {})
        assert result is None


@pytest.mark.asyncio
async def test_merge_entities_reattaches_instances(captured_execute_write):
    """实体消歧合并：别名入库 + DETACH DELETE remove + 实例归一化重挂。"""
    await merge_entities("kb_test", "keep", "remove")

    assert captured_execute_write, "应当提交合并查询"
    first_query, first_params = captured_execute_write[0]
    # run 1：别名注册 + 收集实例 + 删除别名节点
    assert "keep.aliases" in first_query, "应把 remove 名称写入 keep.aliases"
    assert "DETACH DELETE d" in first_query, (
        "应 DETACH DELETE 删除 remove 节点（多实例下 DELETE of, d 会抛错）"
    )
    assert "collect(inst)" in first_query, "应先 collect 压行再删除（防多行重复删除节点）"
    assert "-[r:RELATION]->" not in first_query, "不应再迁移旧 RELATION 边"
    expected = {"keep": "keep", "remove": "remove", "kb": "kb_test"}
    assert first_params == expected, "keep/remove/kb 参数应透传"
    # 无实例时不应继续改名/重挂（single 返回 None 即短路）
    assert len(captured_execute_write) == 1, "无实例时只应提交一次合并查询"


@pytest.mark.asyncio
async def test_merge_aliases_reattaches_instances(captured_execute_write):
    """同义实体合并：每个别名一次合并事务。"""
    await merge_aliases("kb_test", "canonical", ["alias1", "alias2"])

    submitted = captured_execute_write
    assert len(submitted) == 2, "每个别名一次合并事务"
    for (q, params), alias in zip(submitted, ["alias1", "alias2"]):
        assert "keep.aliases" in q, "应把别名写入 canonical 的 aliases"
        assert "DETACH DELETE d" in q
        assert "collect(inst)" in q
        assert "-[r:RELATION]->" not in q
        expected = {"keep": "canonical", "remove": alias, "kb": "kb_test"}
        assert params == expected, "canonical/alias/kb 参数应透传"


@pytest.mark.asyncio
async def test_merge_entities_normalizes_conflicting_instances():
    """同文档已有 keep 实例（冲突）时：边与属性并入 dup，删除原实例。"""
    submitted: list[tuple[str, dict[str, str]]] = []

    async def fake_execute_write(func, *args, **kwargs):
        class FakeResult:
            async def single(self):
                return None

            async def fetch(self, limit=None):
                return []

        class FakeTx:
            async def run(self, query, **params):
                submitted.append((query, params))
                if "RETURN [i IN insts" in query:
                    # run 1 返回一个别名实例（doc_id=doc_x）
                    result = AsyncMock()
                    result.single.return_value = {
                        "insts": [{"id": "inst-1", "doc_id": "doc_x"}],
                    }
                    return result
                if "MERGE (dup)-[r2:RELATION_INSTANCE" in query:
                    # 3b 冲突归并查询（先于 dup 检测判断，二者都含 "MATCH (dup"）
                    return AsyncMock()
                if "MATCH (dup:EntityInstance" in query:
                    # run 2 冲突检测：返回 dup 存在
                    result = AsyncMock()
                    result.single.return_value = {"dup": object()}
                    return result
                return FakeResult()

        await func(FakeTx(), *args, **kwargs)

    with patch(
        "aion_knowledge.infrastructure.graph.client.Neo4jConnection.execute_write",
        side_effect=fake_execute_write,
    ):
        await merge_entities("kb_test", "keep", "remove")

    queries = [q for q, _ in submitted]
    assert any("MATCH (dup:EntityInstance" in q for q in queries), "应检测同文档冲突实例"
    merge_q = next(q for q in queries if "MERGE (dup)" in q)
    assert "DELETE r" in merge_q, "重挂后应删除原边"
    assert "DETACH DELETE inst" in merge_q, "并入后应删除原实例"
    assert "SET dup.entity_name" not in merge_q, "dup 已是 keep 实例，不应改名"


@pytest.mark.asyncio
async def test_add_graph_resolves_aliases(captured_execute_write):
    """写入时应做别名解析：名称命中 canonical.aliases 则归一化（Cypher 内解析）。"""
    entities = {
        "iPhone": {"type": "PRODUCT", "descriptions": ["手机"]},
    }
    relations = {
        ("iPhone", "苹果", "brand_of"): {"descriptions": [], "weight": 1.0},
    }
    await add_graph("kb_test", "doc_1", entities, relations)

    for q, _ in captured_execute_write:
        if "MERGE (e:Entity" in q:
            assert "canon.aliases" in q, "Stage 1 应通过 Entity.aliases 解析别名"
            assert "CASE WHEN size(canons) > 0" in q, "命中别名时取 canonical"
        if "MERGE (s_inst)-[r:RELATION_INSTANCE" in q:
            assert "cs.aliases" in q, "Stage 2 source 应解析别名"
            assert "ct.aliases" in q, "Stage 2 target 应解析别名"


@pytest.mark.asyncio
async def test_delete_graph():
    """删除 KB 图谱。"""
    with patch(
        "aion_knowledge.infrastructure.graph.client.Neo4jConnection.execute_write",
        AsyncMock(return_value=None),
    ):
        result = await delete_graph("kb_test")
        assert result is None


@pytest.mark.asyncio
async def test_get_stats():
    """获取图谱统计。"""
    mock_stats = {"entity_count": 10, "relation_count": 20}
    with patch(
        "aion_knowledge.infrastructure.graph.client.Neo4jConnection.execute_read",
        AsyncMock(return_value=mock_stats),
    ):
        result = await get_stats("kb_test")
        assert result == mock_stats


@pytest.mark.asyncio
async def test_delete_document_graph():
    """删除文档的图谱贡献。"""
    with patch(
        "aion_knowledge.infrastructure.graph.client.Neo4jConnection.execute_write",
        AsyncMock(return_value=None),
    ):
        result = await delete_document_graph("kb_test", "doc_1")
        assert result is None


@pytest.mark.asyncio
async def test_delete_document_graph_noop():
    """删除不存在的文档不应报错。"""
    with patch(
        "aion_knowledge.infrastructure.graph.client.Neo4jConnection.execute_write",
        AsyncMock(return_value=None),
    ):
        result = await delete_document_graph("kb_test", "nonexistent_doc")
        assert result is None


@pytest.mark.asyncio
async def test_add_graph_passes_chunk_ids(captured_execute_write):
    """EntityInstance 写入应带 chunk_ids（取自 source_chunks），并去重累加。"""
    entities = {
        "马云": {"type": "PERSON", "descriptions": ["创始人"],
                "source_chunks": ["c1", "c2"]},
    }
    await add_graph("kb_test", "doc_1", entities, {})

    inst = [(q, p) for q, p in captured_execute_write if "EntityInstance" in q]
    assert inst, "应提交 EntityInstance 写入"
    q, params = inst[0]
    assert "chunk_ids" in q, "Cypher 应 SET inst.chunk_ids"
    # 此断言仅校验 Cypher 形态（跨调用去重表达式），真实去重行为由集成测试验证
    assert "[x IN inst.chunk_ids WHERE NOT x IN row.chunk_ids] + row.chunk_ids" in q, \
        "已存在时应去重累加，而非覆盖"
    assert params["data"][0]["chunk_ids"] == ["c1", "c2"], \
        "data 应透传 source_chunks"


@pytest.mark.asyncio
async def test_add_graph_chunk_ids_default_empty(captured_execute_write):
    """entity 缺 source_chunks 键时，chunk_ids 默认空列表（验证 ent.get(..., []) 兜底）。"""
    entities = {
        "马云": {"type": "PERSON", "descriptions": ["创始人"]},
    }
    await add_graph("kb_test", "doc_1", entities, {})

    inst = [(q, p) for q, p in captured_execute_write if "EntityInstance" in q]
    assert inst, "应提交 EntityInstance 写入"
    _, params = inst[0]
    assert params["data"][0]["chunk_ids"] == [], \
        "缺 source_chunks 时 chunk_ids 应为空列表，而非 None / 缺键"
