"""KeywordExtractModule 三层架构测试。"""
from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest

from aion_knowledge.pipeline.postproc.base import PostProcContext
from aion_knowledge.pipeline.postproc.keyword_extract.processor import (
    KeywordExtractModule,
)


class TestKeywordExtractModule:
    @pytest.mark.asyncio
    async def test_tier1_llm_extraction(self):
        """Tier 1: LLM 自由生成关键词，通过 Repo 写入。"""
        module = KeywordExtractModule()
        ctx = PostProcContext(document_id="d1", kb_id="kb1", doc_name="t.md")
        chunk_uuid = "00000000-0000-0000-0000-000000000001"
        chunks = [{"chunk_uuid": chunk_uuid, "content": "人工智能和机器学习在医疗领域的应用"}]

        mock_session = AsyncMock()

        @asynccontextmanager
        async def mock_get_session():
            yield mock_session

        with (
            patch.object(module, "_load_kb_tags", AsyncMock(return_value=[])),
            patch.object(module, "_extract_keywords_llm", AsyncMock(return_value=["人工智能", "机器学习", "医疗"])),
            patch("aion_knowledge.pipeline.postproc.keyword_extract.processor.get_llm_client_for_module"),
            patch("aion_knowledge.infrastructure.db.get_session", mock_get_session),
            patch("aion_knowledge.storage.relational.chunk_repo.ChunkRepository") as mock_repo_class,
        ):
            mock_repo_instance = AsyncMock()
            mock_repo_class.return_value = mock_repo_instance

            count = await module.process(ctx, chunks)
            assert count == 1
            mock_repo_instance.update_keywords.assert_awaited_once_with(
                chunk_uuid, ["人工智能", "机器学习", "医疗"],
            )

    @pytest.mark.asyncio
    async def test_tier2_exact_match(self):
        """Tier 2: 内容包含某 tag，精确子串匹配应返回该 tag。"""
        module = KeywordExtractModule()
        tags = ["人工智能", "医疗", "科技"]
        content = "人工智能在医疗领域的应用"
        matched = module._match_tags_exact(content, tags)
        assert "人工智能" in matched
        assert "医疗" in matched
        assert "科技" not in matched  # "科技" 不在 content 中

    @pytest.mark.asyncio
    async def test_tier2_no_match(self):
        """Tier 2: 内容不含任何 tag，返回空列表。"""
        module = KeywordExtractModule()
        tags = ["区块链", "人工智能"]
        content = "今天的天气很好"
        matched = module._match_tags_exact(content, tags)
        assert matched == []

    @pytest.mark.asyncio
    async def test_tier3_llm_selection(self):
        """Tier 3: Tier 2 匹配不足 3 个时，LLM 从剩余 tag 中选取。"""
        module = KeywordExtractModule()
        ctx = PostProcContext(document_id="d1", kb_id="kb1", doc_name="t.md")
        chunk_uuid = "00000000-0000-0000-0000-000000000001"
        chunks = [{"chunk_uuid": chunk_uuid, "content": "人工智能在医疗领域的应用"}]

        mock_session = AsyncMock()

        @asynccontextmanager
        async def mock_get_session():
            yield mock_session

        with (
            patch.object(module, "_load_kb_tags", AsyncMock(return_value=["人工智能", "医疗", "深度学习"])),
            patch.object(module, "_extract_keywords_llm", AsyncMock(return_value=["AI"])),
            patch.object(module, "_select_tags_llm", AsyncMock(return_value=["深度学习"])),
            patch("aion_knowledge.pipeline.postproc.keyword_extract.processor.get_llm_client_for_module"),
            patch("aion_knowledge.infrastructure.db.get_session", mock_get_session),
            patch("aion_knowledge.storage.relational.chunk_repo.ChunkRepository") as mock_repo_class,
        ):
            mock_repo_instance = AsyncMock()
            mock_repo_class.return_value = mock_repo_instance

            count = await module.process(ctx, chunks)
            assert count == 1
            # Tier 1: ["AI"], Tier 2: ["人工智能", "医疗"], Tier 3: ["深度学习"]
            mock_repo_instance.update_keywords.assert_awaited_once_with(
                chunk_uuid, ["AI", "人工智能", "医疗", "深度学习"],
            )

    @pytest.mark.asyncio
    async def test_tier3_select_tags_validates(self):
        """Tier 3: LLM 返回的内容被过滤，只保留在 tags 列表中的值。"""
        module = KeywordExtractModule()
        remaining_tags = ["深度学习", "机器学习"]

        llm_mock = AsyncMock()
        # LLM 返回一个有效 tag 和一个无效 tag
        llm_mock.generate = AsyncMock(return_value="深度学习, 无效标签")

        result = await module._select_tags_llm(llm_mock, "文本内容", remaining_tags, [])
        assert "深度学习" in result
        assert "无效标签" not in result

    @pytest.mark.asyncio
    async def test_combined_dedup(self):
        """三层结果有重复时，keywords 中应去重。"""
        module = KeywordExtractModule()
        ctx = PostProcContext(document_id="d1", kb_id="kb1", doc_name="t.md")
        chunk_uuid = "00000000-0000-0000-0000-000000000001"
        chunks = [{"chunk_uuid": chunk_uuid, "content": "人工智能和机器学习是当今科技领域最热门的话题之一。"}]

        mock_session = AsyncMock()

        @asynccontextmanager
        async def mock_get_session():
            yield mock_session

        with (
            patch.object(module, "_load_kb_tags", AsyncMock(return_value=["人工智能", "机器学习"])),
            # Tier 1 返回 "人工智能"，Tier 2 也会匹配 "人工智能"，应去重
            patch.object(module, "_extract_keywords_llm", AsyncMock(return_value=["人工智能", "机器学习"])),
            patch.object(module, "_select_tags_llm", AsyncMock(return_value=[])),
            patch("aion_knowledge.pipeline.postproc.keyword_extract.processor.get_llm_client_for_module"),
            patch("aion_knowledge.infrastructure.db.get_session", mock_get_session),
            patch("aion_knowledge.storage.relational.chunk_repo.ChunkRepository") as mock_repo_class,
        ):
            mock_repo_instance = AsyncMock()
            mock_repo_class.return_value = mock_repo_instance

            count = await module.process(ctx, chunks)
            assert count == 1
            # 去重后 "人工智能" 只出现一次
            mock_repo_instance.update_keywords.assert_awaited_once_with(
                chunk_uuid, ["人工智能", "机器学习"],
            )

    @pytest.mark.asyncio
    async def test_keywords_written_to_chunk_text(self):
        """验证 keywords 通过 Repo 写入。"""
        module = KeywordExtractModule()
        ctx = PostProcContext(document_id="d1", kb_id="kb1", doc_name="t.md")
        chunk_uuid = "00000000-0000-0000-0000-000000000001"
        chunks = [{"chunk_uuid": chunk_uuid, "content": "深度学习在图像识别中的应用"}]

        mock_session = AsyncMock()

        @asynccontextmanager
        async def mock_get_session():
            yield mock_session

        with (
            patch.object(module, "_load_kb_tags", AsyncMock(return_value=["深度学习", "图像识别"])),
            patch.object(module, "_extract_keywords_llm", AsyncMock(return_value=["深度学习", "计算机视觉"])),
            patch.object(module, "_select_tags_llm", AsyncMock(return_value=[])),
            patch("aion_knowledge.pipeline.postproc.keyword_extract.processor.get_llm_client_for_module"),
            patch("aion_knowledge.infrastructure.db.get_session", mock_get_session),
            patch("aion_knowledge.storage.relational.chunk_repo.ChunkRepository") as mock_repo_class,
        ):
            mock_repo_instance = AsyncMock()
            mock_repo_class.return_value = mock_repo_instance

            await module.process(ctx, chunks)
            mock_repo_instance.update_keywords.assert_awaited_once_with(
                chunk_uuid, ["深度学习", "计算机视觉", "图像识别"],
            )

    @pytest.mark.asyncio
    async def test_empty_chunks(self):
        """空 chunks 应返回 0。"""
        module = KeywordExtractModule()
        ctx = PostProcContext(document_id="d1", kb_id="kb1", doc_name="t.md")
        count = await module.process(ctx, [])
        assert count == 0

    @pytest.mark.asyncio
    async def test_skips_empty_content(self):
        """空内容的 chunk 应跳过。"""
        module = KeywordExtractModule()
        ctx = PostProcContext(document_id="d1", kb_id="kb1", doc_name="t.md")
        chunks = [{"chunk_uuid": "c1", "content": ""}]

        with (
            patch.object(module, "_load_kb_tags", AsyncMock(return_value=["tag"])),
            patch("aion_knowledge.pipeline.postproc.keyword_extract.processor.get_llm_client_for_module"),
        ):
            count = await module.process(ctx, chunks)
            assert count == 0

    @pytest.mark.asyncio
    async def test_skips_chunk_without_uuid(self):
        """无 chunk_uuid 的 chunk 应跳过。"""
        module = KeywordExtractModule()
        ctx = PostProcContext(document_id="d1", kb_id="kb1", doc_name="t.md")
        chunks = [{"content": "some content"}]

        with (
            patch.object(module, "_load_kb_tags", AsyncMock(return_value=["tag"])),
            patch("aion_knowledge.pipeline.postproc.keyword_extract.processor.get_llm_client_for_module"),
        ):
            count = await module.process(ctx, chunks)
            assert count == 0

    def test_module_factory(self):
        from aion_knowledge.pipeline.postproc.keyword_extract.processor import module
        assert isinstance(module(), KeywordExtractModule)
