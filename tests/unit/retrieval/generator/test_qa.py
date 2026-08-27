"""Tests for qa — LLM answer generation."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aion_knowledge.retrieval.generator.qa import build_prompt, generate_answer


class TestBuildPrompt:
    """Prompt 组装测试。"""

    def test_build_prompt_with_context(self):
        """验证 prompt 包含上下文和问题。"""
        context = [{"content": "文档内容", "score": 0.9}]
        prompt = build_prompt("什么是 RRF", context)
        assert "什么是 RRF" in prompt
        assert "文档内容" in prompt
        assert "检索结果" in prompt

    def test_build_prompt_empty_context(self):
        """空上下文也能生成合理 prompt。"""
        prompt = build_prompt("测试", [])
        assert "测试" in prompt
        assert "无检索结果" in prompt


class TestGenerateAnswer:
    """LLM 调用测试。"""

    @pytest.mark.asyncio
    async def test_generate_answer_non_stream(self):
        """非流式模式返回完整 answer 字符串。"""
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(return_value="这是回答")

        with patch("aion_knowledge.retrieval.generator.qa.create_llm", return_value=mock_llm):
            result = await generate_answer("测试", [], stream=False)

        assert result == "这是回答"
        mock_llm.generate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_generate_answer_stream(self):
        """流式模式返回 AsyncIterator。"""
        mock_llm = MagicMock()
        mock_llm.stream = MagicMock()

        with patch("aion_knowledge.retrieval.generator.qa.create_llm", return_value=mock_llm):
            result = await generate_answer("测试", [], stream=True)

        assert result == mock_llm.stream.return_value
        mock_llm.stream.assert_called_once()
