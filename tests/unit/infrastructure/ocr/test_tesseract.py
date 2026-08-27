"""测试 TesseractOCREngine。"""

from unittest.mock import patch

import pytest

from aion_knowledge.infrastructure.ocr.engine import OCRNotAvailable
from aion_knowledge.infrastructure.ocr.tesseract import TesseractOCREngine


@pytest.mark.asyncio
async def test_tesseract_available_and_returns_text():
    engine = TesseractOCREngine()
    with patch("pytesseract.image_to_string", return_value="Extracted text") as mock:
        with patch("pytesseract.get_tesseract_version", return_value="5.0"):
            with patch("PIL.Image.open"):
                result = await engine.ocr(b"fake_image_bytes")
    assert result == "Extracted text"
    mock.assert_called_once()


@pytest.mark.asyncio
async def test_tesseract_raises_ocronotavailable_when_missing():
    engine = TesseractOCREngine()
    with patch("pytesseract.get_tesseract_version", side_effect=Exception("not found")):
        with pytest.raises(OCRNotAvailable):
            await engine.ocr(b"fake_image_bytes")


@pytest.mark.asyncio
async def test_tesseract_uses_cache():
    """可用性检测是延迟初始化且缓存的。"""
    engine = TesseractOCREngine()
    # get_tesseract_version 应只调用一次
    with patch("pytesseract.image_to_string", return_value="text") as img_mock:
        with patch("pytesseract.get_tesseract_version", return_value="5.0") as ver_mock:
            with patch("PIL.Image.open"):
                await engine.ocr(b"img1")
                await engine.ocr(b"img2")
    assert ver_mock.call_count == 1  # 只检测一次
    assert img_mock.call_count == 2
