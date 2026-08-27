"""测试 VLMOCREngine。"""

from unittest.mock import AsyncMock

import pytest

from aion_knowledge.infrastructure.ocr.vlm_ocr import VLMOCREngine


@pytest.fixture
def mock_vlm_client():
    client = AsyncMock()
    client.generate_with_images = AsyncMock(return_value="Extracted text content")
    return client


@pytest.mark.asyncio
async def test_vlm_ocr_returns_text(mock_vlm_client):
    engine = VLMOCREngine(llm=mock_vlm_client)
    result = await engine.ocr(b"fake_image_bytes")
    assert result == "Extracted text content"
    mock_vlm_client.generate_with_images.assert_awaited_once()


@pytest.mark.asyncio
async def test_vlm_ocr_empty_result(mock_vlm_client):
    mock_vlm_client.generate_with_images.return_value = ""
    engine = VLMOCREngine(llm=mock_vlm_client)
    result = await engine.ocr(b"fake_image_bytes")
    assert result == ""


@pytest.mark.asyncio
async def test_vlm_ocr_calls_sanitizer(mock_vlm_client):
    mock_vlm_client.generate_with_images.return_value = "```html\n<p>Test</p>\n```"
    engine = VLMOCREngine(llm=mock_vlm_client)
    result = await engine.ocr(b"fake_image_bytes")
    # sanitizer 会剥离代码围栏并转换 HTML
    assert "Test" in result
    assert "```" not in result
