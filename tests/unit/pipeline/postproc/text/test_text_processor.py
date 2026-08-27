"""TextModule 测试。"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, Mock, patch

import pytest

from aion_knowledge.pipeline.postproc.base import PostProcContext
from aion_knowledge.pipeline.postproc.text.processor import TextModule, module

DOC_UUID = uuid.uuid4()
KB_UUID = uuid.uuid4()

class TestTextModule:
    """验证 TextModule 行为。"""

    def test_module_factory(self):
        """module() 工厂返回 TextModule 实例。"""
        m = module()
        assert isinstance(m, TextModule)
        assert m.always_on is True
        assert m.depends_on == []

    @pytest.mark.asyncio
    async def test_process_inserts_chunks_and_parents(self):
        """process() 应插入子块 + 父块，返回插入总数。"""
        ctx = PostProcContext(document_id=str(DOC_UUID), kb_id=str(KB_UUID), doc_name="t.md")
        chunks = [
            {"content": "chunk1", "seq_num": 0, "token_count": 10, "chunk_id": "c1", "heading_path": ""},
            {"content": "chunk2", "seq_num": 1, "token_count": 20, "chunk_id": "c2", "heading_path": ""},
        ]

        mock_session = Mock()
        mock_session.flush = AsyncMock()
        mock_session.execute = AsyncMock()  # 用于捕获 content_tokens UPDATE

        with patch("aion_knowledge.infrastructure.db.get_session") as mock_gs:
            mock_gs.return_value.__aenter__.return_value = mock_session
            m = module()
            count = await m.process(ctx, chunks)

        # 2 个子块 + 1 个父块 (每 2 个子块合并为 1 个父块)
        assert count == 3
        assert mock_session.add_all.call_count == 2
        # 验证 content_tokens 批量 UPDATE
        assert mock_session.execute.call_count >= 1
        content_tokens_calls = [
            call for call in mock_session.execute.call_args_list
            if hasattr(call[0][0], 'text') and 'content_tokens' in call[0][0].text
        ]
        assert len(content_tokens_calls) == 1
        sql: str = content_tokens_calls[0][0][0].text
        assert "UPDATE chunk_text" in sql and "content_tokens" in sql
        assert "tsvector_to_array" in sql
        assert "to_tsvector" in sql

    @pytest.mark.asyncio
    async def test_process_empty_chunks(self):
        """空 chunks 应返回 0。"""
        ctx = PostProcContext(document_id=str(DOC_UUID), kb_id=str(KB_UUID), doc_name="t.md")
        with patch("aion_knowledge.infrastructure.db.get_session") as mock_gs:
            mock_session = Mock()
            mock_gs.return_value.__aenter__.return_value = mock_session
            m = module()
            count = await m.process(ctx, [])
        assert count == 0
        mock_session.add_all.assert_not_called()
        mock_session.execute.assert_not_called()
