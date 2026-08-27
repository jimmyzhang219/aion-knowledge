"""QuestionGenModule 测试 — 生成问题，嵌入向量，写入 chunk_vector。"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from aion_knowledge.pipeline.postproc.base import PostProcContext
from aion_knowledge.pipeline.postproc.question_gen.processor import QuestionGenModule


class TestQuestionGenModule:
    @pytest.mark.asyncio
    async def test_process_updates_chunk_vector(self):
        """生成问题 → 嵌入 → UPDATE chunk_vector 含 questions + embedding_questions。"""
        module = QuestionGenModule()
        ctx = PostProcContext(document_id="d1", kb_id="kb1", doc_name="t.md")
        chunks = [{"chunk_uuid": "c1", "content": "Python 是一种解释型高级编程语言。"}]

        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__ = AsyncMock()

        mock_embedder = AsyncMock()
        mock_embedder.embed_documents.return_value = [[0.1, 0.2, 0.3]]

        with (
            patch.object(module, "_generate_questions", AsyncMock(return_value=["什么是Python？", "Python有哪些特点？"])),
            patch("aion_knowledge.pipeline.postproc.question_gen.processor.create_embedder") as mock_create,
            patch("aion_knowledge.pipeline.postproc.question_gen.processor.get_llm_client_for_module"),
            patch("aion_knowledge.infrastructure.db.get_session", return_value=AsyncMock().__aenter__.return_value) as mock_gs,
        ):
            mock_create.return_value = mock_embedder
            mock_gs.return_value = mock_session
            count = await module.process(ctx, chunks)
            assert count == 1
            # 验证 UPDATE 包含 questions + embedding_questions
            call_sql = mock_session.execute.call_args[0][0].text
            assert "UPDATE chunk_vector" in call_sql
            assert "embedding_questions" in call_sql

    @pytest.mark.asyncio
    async def test_questions_joined_with_comma(self):
        """生成的问题列表被逗号拼接为单字符串。"""
        module = QuestionGenModule()
        ctx = PostProcContext(document_id="d1", kb_id="kb1", doc_name="t.md")
        chunks = [{"chunk_uuid": "c1", "content": "机器学习是人工智能的分支。"}]

        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__ = AsyncMock()

        mock_embedder = AsyncMock()
        mock_embedder.embed_documents.return_value = [[0.1, 0.2, 0.3]]

        with (
            patch.object(module, "_generate_questions", AsyncMock(return_value=["什么是机器学习？", "机器学习与AI的关系？"])),
            patch("aion_knowledge.pipeline.postproc.question_gen.processor.create_embedder") as mock_create,
            patch("aion_knowledge.pipeline.postproc.question_gen.processor.get_llm_client_for_module"),
            patch("aion_knowledge.infrastructure.db.get_session", return_value=AsyncMock().__aenter__.return_value) as mock_gs,
        ):
            mock_create.return_value = mock_embedder
            mock_gs.return_value = mock_session
            await module.process(ctx, chunks)
            params = mock_session.execute.call_args[0][1]
            assert params["q"] == "什么是机器学习？，机器学习与AI的关系？"

    @pytest.mark.asyncio
    async def test_fallback_when_embedding_fails(self):
        """嵌入失败时，只写入 questions 文本，不写向量。"""
        module = QuestionGenModule()
        ctx = PostProcContext(document_id="d1", kb_id="kb1", doc_name="t.md")
        chunks = [{"chunk_uuid": "c1", "content": "测试内容。这是用于测试嵌入失败情况的长文本内容。"}]

        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__ = AsyncMock()

        # embedding 失败
        mock_embedder = AsyncMock()
        mock_embedder.embed_documents.side_effect = Exception("embedding failed")

        with (
            patch.object(module, "_generate_questions", AsyncMock(return_value=["问题？"])),
            patch("aion_knowledge.pipeline.postproc.question_gen.processor.create_embedder") as mock_create,
            patch("aion_knowledge.pipeline.postproc.question_gen.processor.get_llm_client_for_module"),
            patch("aion_knowledge.infrastructure.db.get_session", return_value=AsyncMock().__aenter__.return_value) as mock_gs,
        ):
            mock_create.return_value = mock_embedder
            mock_gs.return_value = mock_session
            count = await module.process(ctx, chunks)
            assert count == 1
            # 写入但无 embedding_questions
            call_sql = mock_session.execute.call_args[0][0].text
            assert "embedding_questions" not in call_sql
            assert "questions" in call_sql

    @pytest.mark.asyncio
    async def test_empty_chunks(self):
        module = QuestionGenModule()
        ctx = PostProcContext(document_id="d1", kb_id="kb1", doc_name="t.md")
        assert await module.process(ctx, []) == 0

    @pytest.mark.asyncio
    async def test_skips_empty_content(self):
        module = QuestionGenModule()
        ctx = PostProcContext(document_id="d1", kb_id="kb1", doc_name="t.md")
        with patch("aion_knowledge.pipeline.postproc.question_gen.processor.get_llm_client_for_module"):
            assert await module.process(ctx, [{"chunk_uuid": "c1", "content": ""}]) == 0

    @pytest.mark.asyncio
    async def test_skips_chunk_without_uuid(self):
        module = QuestionGenModule()
        ctx = PostProcContext(document_id="d1", kb_id="kb1", doc_name="t.md")
        chunks = [{"content": "some content"}]
        with (
            patch.object(module, "_generate_questions", AsyncMock(return_value=["问题？"])),
            patch("aion_knowledge.pipeline.postproc.question_gen.processor.get_llm_client_for_module"),
        ):
            assert await module.process(ctx, chunks) == 0

    def test_module_factory(self):
        from aion_knowledge.pipeline.postproc.question_gen.processor import module
        assert isinstance(module(), QuestionGenModule)
