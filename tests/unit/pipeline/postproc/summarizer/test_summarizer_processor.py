"""SummarizerModule 测试。"""
from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest

from aion_knowledge.pipeline.postproc.base import PostProcContext
from aion_knowledge.pipeline.postproc.summarizer.processor import SummarizerModule


class TestSummarizerModule:
    @pytest.mark.asyncio
    async def test_process_writes_summary_text_tokens_and_vector(self):
        module = SummarizerModule()
        ctx = PostProcContext(document_id="d1", kb_id="kb1", doc_name="t.md")
        chunks = [{"chunk_uuid": "c1", "content": "需要摘要的文本内容。"}]

        mock_session = Mock()
        mock_session.execute = AsyncMock()

        with (
            patch.object(module, "_summarize_chunk", AsyncMock(return_value="这是摘要。")),
            patch(
                "aion_knowledge.pipeline.postproc.summarizer.processor.create_embedder"
            ) as mock_create,
            patch("aion_knowledge.pipeline.postproc.summarizer.processor.get_llm_client_for_module"),
            patch("aion_knowledge.infrastructure.db.get_session") as mock_gs,
        ):
            mock_provider = AsyncMock()
            mock_provider.embed_documents.return_value = [[0.1, 0.2, 0.3]]
            mock_create.return_value = mock_provider
            mock_gs.return_value.__aenter__.return_value = mock_session
            count = await module.process(ctx, chunks)
            assert count == 1

            # 验证写入三个存储层：chunk_text.summary_text、chunk_text.summary_tokens、chunk_vector
            assert mock_session.execute.call_count == 3
            sql = mock_session.execute.call_args_list[0].args[0].text
            assert "UPDATE chunk_text SET summary_text" in sql

    @pytest.mark.asyncio
    async def test_empty_chunks(self):
        module = SummarizerModule()
        ctx = PostProcContext(document_id="d1", kb_id="kb1", doc_name="t.md")
        assert await module.process(ctx, []) == 0

    @pytest.mark.asyncio
    async def test_skips_empty_content(self):
        module = SummarizerModule()
        ctx = PostProcContext(document_id="d1", kb_id="kb1", doc_name="t.md")
        with patch(
            "aion_knowledge.pipeline.postproc.summarizer.processor.get_llm_client_for_module"
        ):
            assert await module.process(ctx, [{"chunk_uuid": "c1", "content": ""}]) == 0

    def test_depends_on(self):
        assert SummarizerModule.depends_on == ["text", "vector"]
        assert SummarizerModule.always_on is False

    def test_module_factory(self):
        from aion_knowledge.pipeline.postproc.summarizer.processor import module
        assert isinstance(module(), SummarizerModule)
