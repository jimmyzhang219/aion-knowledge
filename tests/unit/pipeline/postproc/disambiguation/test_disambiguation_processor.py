"""DisambiguationModule 测试（改造后版本）。"""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import pytest

from aion_knowledge.pipeline.postproc.base import PostProcContext
from aion_knowledge.pipeline.postproc.disambiguation.processor import DisambiguationModule

KB_UUID = uuid.uuid4()


class TestDisambiguationModule:
    @pytest.mark.asyncio
    async def test_process_with_graph_extract(self):
        """主路径：读取 KB 图谱实体 → 预过滤 → LLM 裁决 → 合并。"""
        module = DisambiguationModule()
        ctx = PostProcContext(document_id="d1", kb_id=str(KB_UUID), doc_name="t.md")
        chunks = [{"chunk_uuid": "00000000-0000-0000-0000-000000000001", "content": "test"}]

        with patch.object(module, "_load_kb_entities", AsyncMock(return_value=([
            {"entity_name": "Apple", "entity_type": "ORG"},
            {"entity_name": "App1e", "entity_type": "ORG"},
        ], []))):
            with patch("aion_knowledge.pipeline.postproc.disambiguation.processor.get_llm_client_for_module"):
                with patch.object(module, "_resolve_batches", AsyncMock(return_value=[("Apple", ["App1e"])])):
                    with patch("aion_knowledge.pipeline.postproc.disambiguation.processor.DisambiguationMerger") as mock_merger:
                        with patch("aion_knowledge.storage.relational.graph_repo.GraphMetadataRepository") as mock_repo_cls:
                            repo = mock_repo_cls.return_value
                            repo.load_checkpoint = AsyncMock(return_value=None)
                            merger = mock_merger.return_value
                            merger.batch_merge = AsyncMock()
                            count = await module.process(ctx, chunks)
                            assert count == 1
                            merger.batch_merge.assert_awaited_once_with([("Apple", ["App1e"])], str(KB_UUID))

    @pytest.mark.asyncio
    async def test_process_no_kb_entities_skips(self):
        """KB 图谱无数据时跳过消歧（fallback 已删除，不触发重复 LLM 提取）。"""
        module = DisambiguationModule()
        ctx = PostProcContext(document_id="d1", kb_id=str(KB_UUID), doc_name="t.md")
        chunks = [{"chunk_uuid": "00000000-0000-0000-0000-000000000001", "content": "苹果公司发布新款 iPhone 手机设备。"}]

        with patch.object(module, "_load_kb_entities", AsyncMock(return_value=([], []))):
            with patch("aion_knowledge.pipeline.postproc.disambiguation.processor.get_llm_client_for_module"):
                with patch("aion_knowledge.pipeline.postproc.graph_extract.extractor.extract_entities_with_gleaning", AsyncMock()) as mock_extract:
                    count = await module.process(ctx, chunks)
                    assert count == 0
                    mock_extract.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_chunks(self):
        module = DisambiguationModule()
        ctx = PostProcContext(document_id="d1", kb_id="kb1", doc_name="t.md")
        assert await module.process(ctx, []) == 0

    @pytest.mark.asyncio
    async def test_load_kb_entities_uses_neo4j(self):
        """_load_kb_entities 应调用 load_kb_graph，返回 (entities, relations)。"""
        module = DisambiguationModule()
        fake_entities = [{"entity_name": "苹果", "entity_type": "ORG"}]
        fake_relations = [{"source_entity": "苹果", "target_entity": "iPhone", "relation_type": "发布", "weight": 1.0}]
        with patch("aion_knowledge.infrastructure.graph.load_kb_graph",
                   AsyncMock(return_value=(fake_entities, fake_relations))) as mock_load:
            entities, relations = await module._load_kb_entities(str(KB_UUID))
        mock_load.assert_awaited_once_with(str(KB_UUID))
        assert entities == fake_entities
        assert relations == fake_relations

    def test_depends_on_updated(self):
        assert "graph_extract" in DisambiguationModule.depends_on
        assert "text" in DisambiguationModule.depends_on
        assert DisambiguationModule.always_on is False

    def test_module_factory(self):
        from aion_knowledge.pipeline.postproc.disambiguation.processor import module
        assert isinstance(module(), DisambiguationModule)

    @pytest.mark.asyncio
    async def test_process_skips_when_hash_unchanged(self):
        """KB 图 hash 未变时跳过消歧（不调 LLM 裁决、不合并）。"""
        module = DisambiguationModule()
        ctx = PostProcContext(document_id="d1", kb_id=str(KB_UUID), doc_name="t.md")
        chunks = [{"chunk_uuid": "00000000-0000-0000-0000-000000000001", "content": "test"}]
        entities = [{"entity_name": "Apple", "entity_type": "ORG"}]
        relations = []

        with patch.object(module, "_load_kb_entities", AsyncMock(return_value=(entities, relations))):
            with patch("aion_knowledge.pipeline.postproc.community.checkpoint.compute_graph_hash",
                       return_value="hash-123"):
                with patch("aion_knowledge.storage.relational.graph_repo.GraphMetadataRepository") as mock_repo:
                    repo = mock_repo.return_value
                    repo.load_checkpoint = AsyncMock(return_value="hash-123")
                    repo.save_checkpoint = AsyncMock()
                    with patch("aion_knowledge.pipeline.postproc.disambiguation.processor.get_llm_client_for_module"):
                        with patch.object(module, "_resolve_batches", AsyncMock()) as mock_resolve:
                            count = await module.process(ctx, chunks)
                            assert count == 0
                            mock_resolve.assert_not_awaited()
                            repo.save_checkpoint.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_process_runs_when_hash_changed(self):
        """KB 图 hash 变化时执行消歧并保存检查点。"""
        module = DisambiguationModule()
        ctx = PostProcContext(document_id="d1", kb_id=str(KB_UUID), doc_name="t.md")
        chunks = [{"chunk_uuid": "00000000-0000-0000-0000-000000000001", "content": "test"}]
        entities = [
            {"entity_name": "Apple", "entity_type": "ORG"},
            {"entity_name": "App1e", "entity_type": "ORG"},
        ]
        relations = []

        with patch.object(module, "_load_kb_entities", AsyncMock(return_value=(entities, relations))):
            with patch("aion_knowledge.pipeline.postproc.community.checkpoint.compute_graph_hash",
                       return_value="hash-new"):
                with patch("aion_knowledge.storage.relational.graph_repo.GraphMetadataRepository") as mock_repo:
                    repo = mock_repo.return_value
                    repo.load_checkpoint = AsyncMock(return_value="hash-old")
                    repo.save_checkpoint = AsyncMock()
                    with patch("aion_knowledge.pipeline.postproc.disambiguation.processor.get_llm_client_for_module"):
                        with patch.object(module, "_resolve_batches", AsyncMock(return_value=[("Apple", ["App1e"])])):
                            with patch("aion_knowledge.pipeline.postproc.disambiguation.processor.DisambiguationMerger") as mock_merger:
                                merger = mock_merger.return_value
                                merger.batch_merge = AsyncMock()
                                count = await module.process(ctx, chunks)
                                assert count == 1
                                repo.save_checkpoint.assert_awaited_once_with(str(KB_UUID), "disambiguation", {"graph_hash": "hash-new"})

    @pytest.mark.asyncio
    async def test_process_saves_checkpoint_when_no_candidates(self):
        """hash 变化但预过滤无候选时，也保存检查点（下次命中跳过）。"""
        module = DisambiguationModule()
        ctx = PostProcContext(document_id="d1", kb_id=str(KB_UUID), doc_name="t.md")
        chunks = [{"chunk_uuid": "00000000-0000-0000-0000-000000000001", "content": "test"}]
        entities = [{"entity_name": "Apple", "entity_type": "ORG"}]
        relations = []

        with patch.object(module, "_load_kb_entities", AsyncMock(return_value=(entities, relations))):
            with patch("aion_knowledge.pipeline.postproc.community.checkpoint.compute_graph_hash",
                       return_value="hash-new"):
                with patch("aion_knowledge.storage.relational.graph_repo.GraphMetadataRepository") as mock_repo_cls:
                    repo = mock_repo_cls.return_value
                    repo.load_checkpoint = AsyncMock(return_value="hash-old")
                    repo.save_checkpoint = AsyncMock()
                    with patch("aion_knowledge.pipeline.postproc.disambiguation.processor.get_llm_client_for_module"):
                        count = await module.process(ctx, chunks)
                        assert count == 0
                        repo.save_checkpoint.assert_awaited_once_with(str(KB_UUID), "disambiguation", {"graph_hash": "hash-new"})
