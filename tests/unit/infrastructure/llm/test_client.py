"""LLMClient 单元测试。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aion_knowledge.infrastructure.llm.client import LLMClient


class TestLLMClient:
    """LLMClient 核心功能测试。"""

    @pytest.mark.asyncio
    async def test_generate_returns_content(self):
        """generate 返回 model.ainvoke 的 content。"""
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "回答内容"
        mock_model.ainvoke = AsyncMock(return_value=mock_response)

        client = LLMClient(mock_model)
        result = await client.generate("你好")

        assert result == "回答内容"
        mock_model.ainvoke.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_generate_with_system_prompt(self):
        """generate 包含 system_prompt。"""
        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(return_value=MagicMock(content="ok"))
        client = LLMClient(mock_model)

        await client.generate("prompt", system_prompt="你是助手")

        args, _ = mock_model.ainvoke.await_args
        messages = args[0]
        assert len(messages) == 2
        assert messages[0].type == "system"
        assert messages[0].content == "你是助手"

    @pytest.mark.asyncio
    async def test_generate_text_delegates_to_generate(self):
        """generate_text 委托给 generate。"""
        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(return_value=MagicMock(content="text"))
        client = LLMClient(mock_model)

        result = await client.generate_text("测试")
        assert result == "text"

    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self):
        """stream 逐块 yield content。"""
        mock_model = MagicMock()

        class FakeChunk:
            def __init__(self, content: str):
                self.content = content

        async def fake_astream(*args, **kwargs):
            for token in ["A", "B", "C"]:
                yield FakeChunk(token)

        mock_model.astream = fake_astream
        client = LLMClient(mock_model)

        tokens = [t async for t in client.stream("prompt")]
        assert tokens == ["A", "B", "C"]

    @pytest.mark.asyncio
    async def test_generate_with_images(self):
        """generate_with_images 包含图片数据。"""
        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(return_value=MagicMock(content="图片描述"))
        client = LLMClient(mock_model)

        img_data = [(b"\x89PNG\r\n", "image/png")]
        result = await client.generate_with_images("描述", img_data)

        assert result == "图片描述"
        args, _ = mock_model.ainvoke.await_args
        messages = args[0]
        assert len(messages) == 1
        content = messages[0].content
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"

    @pytest.mark.asyncio
    async def test_generate_structured_returns_dict(self):
        """generate_structured 返回 dict。"""
        mock_model = MagicMock()
        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(return_value={"name": "test", "value": 42})
        mock_model.with_structured_output = MagicMock(return_value=mock_structured)

        client = LLMClient(mock_model)
        result = await client.generate_structured(
            "提取数据",
            {"type": "object", "properties": {"name": {"type": "string"}}},
        )

        assert result == {"name": "test", "value": 42}
        mock_model.with_structured_output.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_default_response_format(self):
        """generate 默认传入 response_format={"type": "text"}。"""
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "回答"
        mock_model.ainvoke = AsyncMock(return_value=mock_response)
        client = LLMClient(mock_model)

        await client.generate("你好")

        _args, kwargs = mock_model.ainvoke.await_args
        assert "response_format" in kwargs
        assert kwargs["response_format"] == {"type": "text"}

    @pytest.mark.asyncio
    async def test_generate_with_json_response_format(self):
        """generate 传入 response_format={"type": "json_object"} 时透传。"""
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"key": "value"}'
        mock_model.ainvoke = AsyncMock(return_value=mock_response)
        client = LLMClient(mock_model)

        await client.generate("返回 JSON", response_format={"type": "json_object"})

        _args, kwargs = mock_model.ainvoke.await_args
        assert kwargs["response_format"] == {"type": "json_object"}
