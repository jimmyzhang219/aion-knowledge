"""RaptorModule 测试。"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aion_knowledge.pipeline.postproc.base import PostProcContext
from aion_knowledge.pipeline.postproc.raptor.processor import RaptorModule, module


class TestRaptorModule:
    @pytest.mark.asyncio
    async def test_skips_less_than_two_chunks(self):
        module = RaptorModule()
        ctx = PostProcContext(
            document_id="00000000-0000-0000-0000-000000000000",
            kb_id="00000000-0000-0000-0000-000000000000",
            doc_name="t.md",
        )
        assert await module.process(ctx, [{"chunk_uuid": "c1", "content": "单独"}]) == 0

    @pytest.mark.asyncio
    async def test_empty_chunks(self):
        module = RaptorModule()
        ctx = PostProcContext(
            document_id="00000000-0000-0000-0000-000000000000",
            kb_id="00000000-0000-0000-0000-000000000000",
            doc_name="t.md",
        )
        assert await module.process(ctx, []) == 0

    @pytest.mark.asyncio
    async def test_skips_structured_data(self):
        module = RaptorModule()
        ctx = PostProcContext(
            document_id="00000000-0000-0000-0000-000000000000",
            kb_id="00000000-0000-0000-0000-000000000000", doc_name="t.xlsx",
            file_type="xlsx", parser_id="naive",
        )
        assert await module.process(ctx, [
            {"chunk_uuid": "c1", "content": "a"},
            {"chunk_uuid": "c2", "content": "b"},
        ]) == 0

    @pytest.mark.asyncio
    async def test_module_factory(self):
        assert isinstance(module(), RaptorModule)

    @pytest.mark.asyncio
    async def test_module_always_on(self):
        m = RaptorModule()
        assert m.always_on is False
        assert "text" in m.depends_on

    @pytest.mark.asyncio
    async def test_process_with_enough_chunks(self):
        """集成风格测试：mock 所有外部依赖，验证 processor 流程走通。"""
        module = RaptorModule()

        ctx = PostProcContext(
            document_id="00000000-0000-0000-0000-000000000000",
            kb_id="00000000-0000-0000-0000-000000000000",
            doc_name="t.md",
            file_type="pdf", parser_id="naive",
        )
        uid1 = "00000000-0000-0000-0000-000000000001"
        uid2 = "00000000-0000-0000-0000-000000000002"
        chunks = [
            {"chunk_uuid": uid1, "content": "第一段内容。"},
            {"chunk_uuid": uid2, "content": "第二段内容。"},
        ]

        mock_session = MagicMock()
        mock_session.flush = AsyncMock()
        mock_session.execute = AsyncMock()

        with (
            patch("aion_knowledge.pipeline.postproc.raptor.processor.raptor_config") as mock_cfg,
            patch("aion_knowledge.pipeline.postproc.raptor.processor.get_llm_client_for_module") as mock_llm,
            patch("aion_knowledge.pipeline.postproc.raptor.processor._load_vectors", AsyncMock(return_value={uid1: [0.1]*128, uid2: [0.2]*128})),
            patch("aion_knowledge.pipeline.postproc.raptor.processor._EmbeddingAdapter") as mock_adapter,
            patch("aion_knowledge.infrastructure.db.get_session") as mock_gs,
        ):
            mock_cfg.clustering_method = "gmm"
            mock_cfg.output_mode = "flat"
            mock_cfg.max_cluster = 4
            mock_cfg.max_token = 256
            mock_cfg.threshold = 0.1
            mock_cfg.prompt = "{cluster_content}"
            mock_cfg.random_seed = 0
            mock_cfg.small_layer_collapse = 2
            mock_cfg.max_errors = 3

            mock_gs.return_value.__aenter__.return_value = mock_session
            mock_llm.return_value.generate = AsyncMock(return_value="标题行\n摘要正文内容")
            mock_adapter.return_value.encode = AsyncMock(return_value=[0.5] * 128)

            count = await module.process(ctx, chunks)

            assert count >= 1


@pytest.mark.asyncio
async def test_process_deletes_old_rows_before_insert():
    """重跑幂等：写入前先删除同 (kb_id, doc_id) 的旧行。"""
    module = RaptorModule()
    ctx = PostProcContext(
        document_id="00000000-0000-0000-0000-000000000000",
        kb_id="00000000-0000-0000-0000-000000000000",
        doc_name="t.md", file_type="pdf", parser_id="naive",
    )
    uid1 = "00000000-0000-0000-0000-000000000001"
    uid2 = "00000000-0000-0000-0000-000000000002"
    chunks = [
        {"chunk_uuid": uid1, "content": "第一段内容。"},
        {"chunk_uuid": uid2, "content": "第二段内容。"},
    ]

    mock_session = MagicMock()
    mock_session.flush = AsyncMock()
    mock_session.execute = AsyncMock()

    with (
        patch("aion_knowledge.pipeline.postproc.raptor.processor.raptor_config") as mock_cfg,
        patch("aion_knowledge.pipeline.postproc.raptor.processor.get_llm_client_for_module") as mock_llm,
        patch("aion_knowledge.pipeline.postproc.raptor.processor._load_vectors", AsyncMock(return_value={uid1: [0.1]*128, uid2: [0.2]*128})),
        patch("aion_knowledge.pipeline.postproc.raptor.processor._EmbeddingAdapter") as mock_adapter,
        patch("aion_knowledge.infrastructure.db.get_session") as mock_gs,
    ):
        mock_cfg.clustering_method = "gmm"
        mock_cfg.output_mode = "flat"
        mock_cfg.max_cluster = 4
        mock_cfg.max_token = 256
        mock_cfg.threshold = 0.1
        mock_cfg.prompt = "{cluster_content}"
        mock_cfg.random_seed = 0
        mock_cfg.small_layer_collapse = 2
        mock_cfg.max_errors = 3

        mock_gs.return_value.__aenter__.return_value = mock_session
        mock_llm.return_value.generate = AsyncMock(return_value="标题行\n摘要正文内容")
        mock_adapter.return_value.encode = AsyncMock(return_value=[0.5] * 128)

        await module.process(ctx, chunks)

    deletes = [
        c for c in mock_session.execute.await_args_list
        if "DELETE FROM chunk_raptor" in str(c.args[0])
    ]
    assert deletes, "写入前应执行 DELETE FROM chunk_raptor（重跑幂等）"

    # children_ids / parent_id 应为 processor 显式传递（不是列默认值兜底）
    added = [c.args[0] for c in mock_session.add.call_args_list]
    assert added, "应插入摘要行"
    for row in added:
        assert "children_ids" in row.__dict__, "processor 应显式传递 children_ids"
        assert "parent_id" in row.__dict__, "processor 应显式传递 parent_id"


@pytest.mark.asyncio
async def test_process_llm_all_failed_skips_delete():
    """LLM 全失败（吞错不抛异常）零摘要：不执行 DELETE，保留旧树。"""
    module = RaptorModule()
    ctx = PostProcContext(
        document_id="00000000-0000-0000-0000-000000000000",
        kb_id="00000000-0000-0000-0000-000000000000",
        doc_name="t.md", file_type="pdf", parser_id="naive",
    )
    uid1 = "00000000-0000-0000-0000-000000000001"
    uid2 = "00000000-0000-0000-0000-000000000002"
    chunks = [
        {"chunk_uuid": uid1, "content": "第一段内容。"},
        {"chunk_uuid": uid2, "content": "第二段内容。"},
    ]

    mock_session = MagicMock()
    mock_session.flush = AsyncMock()
    mock_session.execute = AsyncMock()

    with (
        patch("aion_knowledge.pipeline.postproc.raptor.processor.raptor_config") as mock_cfg,
        patch("aion_knowledge.pipeline.postproc.raptor.processor.get_llm_client_for_module") as mock_llm,
        patch("aion_knowledge.pipeline.postproc.raptor.processor._load_vectors", AsyncMock(return_value={uid1: [0.1]*128, uid2: [0.2]*128})),
        patch("aion_knowledge.pipeline.postproc.raptor.processor._EmbeddingAdapter") as mock_adapter,
        patch("aion_knowledge.infrastructure.db.get_session") as mock_gs,
    ):
        mock_cfg.clustering_method = "gmm"
        mock_cfg.output_mode = "flat"
        mock_cfg.max_cluster = 4
        mock_cfg.max_token = 256
        mock_cfg.threshold = 0.1
        mock_cfg.prompt = "{cluster_content}"
        mock_cfg.random_seed = 0
        mock_cfg.small_layer_collapse = 2
        mock_cfg.max_errors = 3

        mock_gs.return_value.__aenter__.return_value = mock_session
        # 全部簇的 LLM 摘要都失败（error_count=1 < max_errors=3，吞错返回 None → 零摘要）
        mock_llm.return_value.generate = AsyncMock(side_effect=Exception("LLM 挂了"))
        mock_adapter.return_value.encode = AsyncMock(return_value=[0.5] * 128)

        count = await module.process(ctx, chunks)

    assert count == 0, "LLM 全失败时应零插入"
    deletes = [
        c for c in mock_session.execute.await_args_list
        if "DELETE FROM chunk_raptor" in str(c.args[0])
    ]
    assert not deletes, "零摘要时不应执行 DELETE（旧树保留，避免检索窗口空洞）"
