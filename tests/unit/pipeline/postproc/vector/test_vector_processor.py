"""VectorModule 测试。"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from aion_knowledge.pipeline.postproc.base import PostProcContext
from aion_knowledge.pipeline.postproc.vector.processor import VectorModule, module


class TestVectorModule:
    """验证 VectorModule 行为。"""

    def test_module_factory(self):
        """module() 工厂返回 VectorModule 实例。"""
        m = module()
        assert isinstance(m, VectorModule)
        assert m.always_on is True
        assert m.depends_on == ["text", "vlm_caption"]

    @pytest.mark.asyncio
    async def test_process_skip_empty_chunks_negative(self):
        """负向：校验空列表和空内容均返回 0。"""
        m = module()
        ctx = PostProcContext(document_id="doc-1", kb_id="kb-1", doc_name="t.md")

        # 空列表
        assert await m.process(ctx, []) == 0
        # 全部空内容
        chunks = [{"content": "", "chunk_uuid": "uuid-1"}]
        assert await m.process(ctx, chunks) == 0

    @pytest.mark.asyncio
    async def test_process_tokenizes_content_and_summary(self):
        """process() 应嵌入向量并写入 chunk_vector。"""
        ctx = PostProcContext(document_id="doc-1", kb_id="kb-1", doc_name="t.md")
        chunks = [{
            "content": "人工智能技术",
            "chunk_uuid": "uuid-1",
            "seq_num": 0,
            "chunk_type": "text",
        }]

        with (
            patch("aion_knowledge.infrastructure.db.get_session") as mock_gs,
            patch(
                "aion_knowledge.pipeline.postproc.vector.processor.create_embedder",
            ) as mock_create,
            patch(
                "aion_knowledge.storage.relational.vector_repo.VectorRepository",
            ) as mock_repo_class,
        ):
            mock_session = AsyncMock()
            mock_gs.return_value.__aenter__.return_value = mock_session

            mock_provider = AsyncMock()
            mock_provider.embed_documents = AsyncMock(return_value=[[0.1, 0.2]])
            mock_create.return_value = mock_provider

            mock_repo_instance = AsyncMock()
            mock_repo_class.return_value = mock_repo_instance

            m = module()
            count = await m.process(ctx, chunks)

        assert count == 1
        # 验证 insert 被调用（不再验证 update_tokens）
        mock_repo_instance.insert.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_process_tokenize_short_words_filtered(self):
        """process() 嵌入短词内容并写入 chunk_vector。"""
        ctx = PostProcContext(document_id="doc-1", kb_id="kb-1", doc_name="t.md")
        chunks = [{
            "content": "的 了 是 人工智能",
            "chunk_uuid": "uuid-1",
            "seq_num": 0,
            "chunk_type": "text",
        }]

        with (
            patch("aion_knowledge.infrastructure.db.get_session") as mock_gs,
            patch(
                "aion_knowledge.pipeline.postproc.vector.processor.create_embedder",
            ) as mock_create,
            patch(
                "aion_knowledge.storage.relational.vector_repo.VectorRepository",
            ) as mock_repo_class,
        ):
            mock_session = AsyncMock()
            mock_gs.return_value.__aenter__.return_value = mock_session

            mock_provider = AsyncMock()
            mock_provider.embed_documents = AsyncMock(return_value=[[0.1, 0.2]])
            mock_create.return_value = mock_provider

            mock_repo_instance = AsyncMock()
            mock_repo_class.return_value = mock_repo_instance

            m = module()
            count = await m.process(ctx, chunks)

        assert count == 1
        # 验证 insert 被调用（不再验证 update_tokens）
        mock_repo_instance.insert.assert_awaited_once()
