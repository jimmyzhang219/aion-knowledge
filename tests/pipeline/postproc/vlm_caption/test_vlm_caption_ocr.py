"""测试 VLMCaptionModule OCR 集成。"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aion_knowledge.infrastructure.ocr.engine import OCRNotAvailable
from aion_knowledge.pipeline.postproc.base import PostProcContext
from aion_knowledge.pipeline.postproc.vlm_caption.processor import VLMCaptionModule


@pytest.fixture
def ctx():
    return PostProcContext(
        document_id="doc-uuid",
        kb_id="kb-uuid",
        doc_name="test.pdf",
    )


def make_chunk(chunk_uuid: str, image_source_type: str | None = None) -> dict:
    chunk = {
        "chunk_uuid": chunk_uuid,
        "image_refs": ["images/test_page_1.jpg"],
        "content": "![test_page_1.jpg](images/test_page_1.jpg)",
    }
    if image_source_type:
        chunk["metadata"] = {"image_source_type": image_source_type}
    return chunk


@pytest.mark.asyncio
async def test_scanned_pdf_goes_through_ocr_path(ctx):
    """scanned_pdf 图片应触发 OCR + Caption。"""
    chunks = [make_chunk("chunk-1", image_source_type="scanned_pdf")]

    mod = VLMCaptionModule()
    with (
        patch.object(mod, "_download_images", return_value=[(b"fake_img", "image/jpeg")]),
        patch.object(mod, "_run_vlm_caption", AsyncMock(return_value="Caption text")),
        patch.object(mod, "_run_paddle_ocr", AsyncMock(return_value="")),
        patch.object(mod, "_run_tesseract_ocr", AsyncMock(return_value="OCR text")),
        patch.object(mod, "_update_chunk_content", AsyncMock()) as update_mock,
    ):
        count = await mod.process(ctx, chunks)

    assert count == 1
    update_mock.assert_awaited_once()
    # 验证 OCR + Caption 合并
    call_content = update_mock.call_args[0][1]
    assert "OCR text" in call_content
    assert "Caption text" in call_content
    assert "<!-- ocr-text -->" in call_content
    assert "<!-- vlm-caption -->" in call_content
    assert "<!-- /ocr-text -->" in call_content
    assert "<!-- /vlm-caption -->" in call_content


@pytest.mark.asyncio
async def test_normal_image_uses_caption_only(ctx):
    """非 scanned_pdf 图片只走 Caption 路径。"""
    chunks = [make_chunk("chunk-2")]  # 无 image_source_type

    mod = VLMCaptionModule()
    with (
        patch.object(mod, "_download_images", return_value=[(b"fake_img", "image/jpeg")]),
        patch.object(mod, "_run_vlm_caption", AsyncMock(return_value="Caption text")),
        patch.object(mod, "_run_paddle_ocr", AsyncMock()) as paddle_mock,
        patch.object(mod, "_run_tesseract_ocr", AsyncMock()) as ocr_mock,
        patch.object(mod, "_update_chunk_content", AsyncMock()) as update_mock,
    ):
        count = await mod.process(ctx, chunks)

    assert count == 1
    paddle_mock.assert_not_called()  # 非 scanned_pdf 不调用 OCR
    ocr_mock.assert_not_called()  # 不调用 OCR
    update_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_tesseract_fallback_to_vlm_ocr(ctx):
    """Tesseract 不可用/空结果 → VLM OCR 降级。"""
    chunks = [make_chunk("chunk-3", image_source_type="scanned_pdf")]

    mod = VLMCaptionModule()
    with (
        patch.object(mod, "_download_images", return_value=[(b"fake_img", "image/jpeg")]),
        patch.object(mod, "_run_vlm_caption", AsyncMock(return_value="Caption text")),
        patch.object(mod, "_run_paddle_ocr", AsyncMock(return_value="")),
        patch.object(mod, "_run_tesseract_ocr", AsyncMock(return_value="")),
        patch.object(mod, "_run_vlm_ocr", AsyncMock(return_value="VLM OCR text")),
        patch.object(mod, "_update_chunk_content", AsyncMock()) as update_mock,
    ):
        count = await mod.process(ctx, chunks)

    assert count == 1
    # 合并内容中应包含 VLM OCR 结果
    call_content = update_mock.call_args[0][1]
    assert "VLM OCR text" in call_content


@pytest.mark.asyncio
async def test_ocr_and_caption_both_fail_skips_update(ctx):
    """OCR 和 Caption 都失败 → 不 UPDATE。"""
    chunks = [make_chunk("chunk-4", image_source_type="scanned_pdf")]

    mod = VLMCaptionModule()
    with (
        patch.object(mod, "_download_images", return_value=[(b"fake_img", "image/jpeg")]),
        patch.object(mod, "_run_vlm_caption", AsyncMock(return_value="")),
        patch.object(mod, "_run_paddle_ocr", AsyncMock(return_value="")),
        patch.object(mod, "_run_tesseract_ocr", AsyncMock(return_value="")),
        patch.object(mod, "_run_vlm_ocr", AsyncMock(return_value="")),
        patch.object(mod, "_update_chunk_content", AsyncMock()) as update_mock,
    ):
        count = await mod.process(ctx, chunks)

    assert count == 0
    update_mock.assert_not_called()


@pytest.mark.asyncio
async def test_paddle_primary_engine_used(ctx):
    """PaddleOCR 有结果 → Tesseract / VLM 均不被调用。"""
    chunks = [make_chunk("chunk-5", image_source_type="scanned_pdf")]

    mod = VLMCaptionModule()
    with (
        patch.object(mod, "_download_images", return_value=[(b"fake_img", "image/jpeg")]),
        patch.object(mod, "_run_vlm_caption", AsyncMock(return_value="Caption text")),
        patch.object(mod, "_run_paddle_ocr", AsyncMock(return_value="Paddle OCR text")),
        patch.object(mod, "_run_tesseract_ocr", AsyncMock()) as tesseract_mock,
        patch.object(mod, "_run_vlm_ocr", AsyncMock()) as vlm_mock,
        patch.object(mod, "_update_chunk_content", AsyncMock()) as update_mock,
    ):
        count = await mod.process(ctx, chunks)

    assert count == 1
    tesseract_mock.assert_not_called()  # Paddle 有结果，不再降级
    vlm_mock.assert_not_called()
    call_content = update_mock.call_args[0][1]
    assert "Paddle OCR text" in call_content
    assert "<!-- ocr-text -->" in call_content


@pytest.mark.asyncio
async def test_paddle_fallback_to_tesseract(ctx):
    """PaddleOCR 空结果 → Tesseract 有结果 → VLM 不被调用。"""
    chunks = [make_chunk("chunk-6", image_source_type="scanned_pdf")]

    mod = VLMCaptionModule()
    with (
        patch.object(mod, "_download_images", return_value=[(b"fake_img", "image/jpeg")]),
        patch.object(mod, "_run_vlm_caption", AsyncMock(return_value="Caption text")),
        patch.object(mod, "_run_paddle_ocr", AsyncMock(return_value="")),
        patch.object(mod, "_run_tesseract_ocr", AsyncMock(return_value="Tesseract text")),
        patch.object(mod, "_run_vlm_ocr", AsyncMock()) as vlm_mock,
        patch.object(mod, "_update_chunk_content", AsyncMock()) as update_mock,
    ):
        count = await mod.process(ctx, chunks)

    assert count == 1
    vlm_mock.assert_not_called()
    call_content = update_mock.call_args[0][1]
    assert "Tesseract text" in call_content


@pytest.mark.asyncio
async def test_run_paddle_ocr_returns_empty_on_ocronotavailable():
    """引擎不可用（OCRNotAvailable）→ 空串降级信号。"""
    mod = VLMCaptionModule()
    engine = MagicMock()
    engine.ocr = AsyncMock(side_effect=OCRNotAvailable("not available"))
    with patch(
        "aion_knowledge.pipeline.postproc.vlm_caption.processor.get_engine",
        return_value=engine,
    ):
        result = await mod._run_paddle_ocr(b"fake_img")
    assert result == ""
