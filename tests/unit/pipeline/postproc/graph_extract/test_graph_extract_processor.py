"""GraphExtractModule 测试。"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from aion_knowledge.pipeline.postproc.base import PostProcContext
from aion_knowledge.pipeline.postproc.graph_extract.processor import GraphExtractModule


class TestGraphExtractModule:
    @pytest.mark.asyncio
    async def test_process(self):
        """基础提取：2 chunks 各产生不同实体，验证并发+去重走通后返回 2。"""
        module = GraphExtractModule()
        ctx = PostProcContext(
            document_id="00000000-0000-0000-0000-0000000000d1",
            kb_id="00000000-0000-0000-0000-000000000001",
            doc_name="t.md",
        )
        chunks = [
            {"chunk_uuid": "00000000-0000-0000-0000-000000000001", "content": "马云创建了阿里巴巴。"},
            {"chunk_uuid": "00000000-0000-0000-0000-000000000002", "content": "刘强东创建了京东集团公司。"},
        ]

        with patch.object(module, "_extract_with_gleaning", AsyncMock(side_effect=[
            ([{"name": "马云", "type": "PERSON", "description": "阿里巴巴创始人"}],
             [{"source": "马云", "target": "阿里巴巴", "type": "founder", "weight": 8}]),
            ([{"name": "刘强东", "type": "PERSON", "description": "京东创始人"}],
             [{"source": "刘强东", "target": "京东", "type": "founder", "weight": 8}]),
        ])):
            with patch.object(module, "_trigger_merge", AsyncMock(return_value=None)):
                with patch("aion_knowledge.pipeline.postproc.graph_extract.processor.get_llm_client_for_module", AsyncMock()):
                    count = await module.process(ctx, chunks)
                    assert count == 2  # 2 entities: 马云, 刘强东

    @pytest.mark.asyncio
    async def test_process_with_gleaning(self):
        """验证 gleaning 模式：第一轮提取后，gleaning 补充实体。"""
        module = GraphExtractModule()
        ctx = PostProcContext(
            document_id="00000000-0000-0000-0000-0000000000d1",
            kb_id="00000000-0000-0000-0000-000000000001",
            doc_name="t.md",
        )
        chunks = [
            {"chunk_uuid": "00000000-0000-0000-0000-000000000001", "content": "马云创建了阿里巴巴。张勇是CEO。"},
            {"chunk_uuid": "00000000-0000-0000-0000-000000000002", "content": "马云还创建了蚂蚁集团公司。"},
        ]

        with patch.object(module, "_extract_with_gleaning", AsyncMock(return_value=(
            [{"name": "马云", "type": "PERSON", "description": "阿里巴巴创始人"},
             {"name": "张勇", "type": "PERSON", "description": "阿里巴巴CEO"}],
            [{"source": "马云", "target": "阿里巴巴", "type": "founder", "weight": 8},
             {"source": "张勇", "target": "阿里巴巴", "type": "CEO", "weight": 7}],
        ))):
            with patch.object(module, "_trigger_merge", AsyncMock(return_value=None)):
                with patch("aion_knowledge.pipeline.postproc.graph_extract.processor.get_llm_client_for_module", AsyncMock()):
                    count = await module.process(ctx, chunks)
                    assert count == 2  # 2 entities: 马云, 张勇

    @pytest.mark.asyncio
    async def test_entity_dedup_within_doc(self):
        """同一文档内同名实体去重：两个 chunk 都提取"马云"，最终只算 1 个。"""
        module = GraphExtractModule()
        ctx = PostProcContext(
            document_id="00000000-0000-0000-0000-0000000000d1",
            kb_id="00000000-0000-0000-0000-000000000001",
            doc_name="t.md",
        )
        chunks = [
            {"chunk_uuid": "00000000-0000-0000-0000-000000000001", "content": "马云创建了阿里巴巴。"},
            {"chunk_uuid": "00000000-0000-0000-0000-000000000002", "content": "马云还创建了蚂蚁集团公司。"},
        ]

        with patch.object(module, "_extract_with_gleaning", AsyncMock(side_effect=[
            ([{"name": "马云", "type": "PERSON", "description": "阿里巴巴创始人"}],
             [{"source": "马云", "target": "阿里巴巴", "type": "founder", "weight": 8}]),
            ([{"name": "马云", "type": "PERSON", "description": "蚂蚁集团创始人"}],
             [{"source": "马云", "target": "蚂蚁集团", "type": "founder", "weight": 8}]),
        ])):
            with patch.object(module, "_trigger_merge", AsyncMock(return_value=None)):
                with patch("aion_knowledge.pipeline.postproc.graph_extract.processor.get_llm_client_for_module", AsyncMock()):
                    count = await module.process(ctx, chunks)
                    assert count == 1  # 只有 1 个去重后的实体

    @pytest.mark.asyncio
    async def test_source_chunks_dedup_within_chunk(self):
        """单个 chunk 内 LLM 重复输出同名实体时，source_chunks 不应对同一 chunk_uuid 重复累加。

        回归守卫：processor.py 的 source_chunks 累加必须有 chunk_uuid 去重，
        与 relations 的守卫对齐，否则 writer 写入 Neo4j 的 chunk_ids
        会带列表内重复（Cypher 表达式只做跨列表去重，不处理列表内重复）。
        """
        module = GraphExtractModule()
        ctx = PostProcContext(
            document_id="00000000-0000-0000-0000-0000000000d1",
            kb_id="00000000-0000-0000-0000-000000000001",
            doc_name="t.md",
        )
        chunks = [
            {"chunk_uuid": "00000000-0000-0000-0000-000000000001", "content": "马云创建了阿里巴巴。"},
        ]

        merge_mock = AsyncMock(return_value=None)
        # 同一 chunk 的 LLM 输出里"马云"出现 2 次（gleaning 可能导致）
        with patch.object(module, "_extract_with_gleaning", AsyncMock(return_value=(
            [{"name": "马云", "type": "PERSON", "description": "阿里巴巴创始人"},
             {"name": "马云", "type": "PERSON", "description": "阿里巴巴创始人"}],
            [],
        ))):
            with patch.object(module, "_trigger_merge", merge_mock):
                with patch("aion_knowledge.pipeline.postproc.graph_extract.processor.get_llm_client_for_module", AsyncMock()):
                    count = await module.process(ctx, chunks)
                    assert count == 1  # 去重后 1 个实体

        # _trigger_merge(kb_id, doc_id, all_entities, all_relations)
        all_entities = merge_mock.call_args.args[2]
        assert all_entities["马云"]["source_chunks"] == ["00000000-0000-0000-0000-000000000001"], \
            "同一 chunk_uuid 不应在 source_chunks 内重复"

    @pytest.mark.asyncio
    async def test_empty_chunks(self):
        """空 chunks 直接返回 0。"""
        module = GraphExtractModule()
        ctx = PostProcContext(document_id="00000000-0000-0000-0000-0000000000d1", kb_id="00000000-0000-0000-0000-000000000001", doc_name="t.md")
        assert await module.process(ctx, []) == 0

    def test_depends_on(self):
        """检查模块依赖声明。"""
        assert GraphExtractModule.depends_on == ["text"]
        assert GraphExtractModule.always_on is False

    def test_module_factory(self):
        """module() 工厂函数返回正确类型。"""
        from aion_knowledge.pipeline.postproc.graph_extract.processor import module
        assert isinstance(module(), GraphExtractModule)
