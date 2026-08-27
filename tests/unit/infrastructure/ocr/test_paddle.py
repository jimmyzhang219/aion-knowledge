"""测试 PaddleOCREngine（mock paddleocr/cv2，不要求真实安装）。"""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from aion_knowledge.infrastructure.ocr.engine import OCRNotAvailable
from aion_knowledge.infrastructure.ocr.paddle import PaddleOCREngine


def _install_fake_modules() -> types.ModuleType:
    """向 sys.modules 注入假 paddleocr / cv2 模块（测试环境不装真依赖）。"""
    fake_ocr = types.ModuleType("paddleocr")
    fake_ocr.PaddleOCR = MagicMock()
    fake_cv2 = types.ModuleType("cv2")
    fake_cv2.imdecode = MagicMock(return_value="fake_bgr_img")
    fake_cv2.IMREAD_COLOR = 1  # 真实 cv2 必有此常量，实现中 imdecode 使用
    sys.modules["paddleocr"] = fake_ocr
    sys.modules["cv2"] = fake_cv2
    return fake_ocr


@pytest.fixture(autouse=True)
def _restore_fake_modules():
    """每个用例后复原 sys.modules，避免假 paddleocr/cv2 污染会话。"""
    saved = {name: sys.modules.get(name) for name in ("paddleocr", "cv2")}
    yield
    for name, prev in saved.items():
        if prev is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = prev


@pytest.mark.asyncio
async def test_paddle_available_and_returns_joined_text():
    fake_ocr = _install_fake_modules()
    engine = PaddleOCREngine()
    # predict 返回两个结果对象，各含若干 rec_texts
    fake_ocr.PaddleOCR.return_value.predict.return_value = [
        {"rec_texts": ["第一行", "第二行"]},
        {"rec_texts": ["第三行"]},
    ]
    result = await engine.ocr(b"fake_image_bytes")
    assert result == "第一行\n第二行\n第三行"
    fake_ocr.PaddleOCR.assert_called_once_with(
        lang="ch",
        engine="onnxruntime",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


@pytest.mark.asyncio
async def test_paddle_empty_result_returns_empty_string():
    fake_ocr = _install_fake_modules()
    fake_ocr.PaddleOCR.return_value.predict.return_value = [{"rec_texts": []}]
    engine = PaddleOCREngine()
    assert await engine.ocr(b"fake_image_bytes") == ""


@pytest.mark.asyncio
async def test_paddle_raises_ocronotavailable_when_missing():
    """paddleocr 未安装（sys.modules 置 None → ImportError）→ OCRNotAvailable。"""
    engine = PaddleOCREngine()
    with patch.dict(sys.modules, {"paddleocr": None}):
        with pytest.raises(OCRNotAvailable):
            await engine.ocr(b"fake_image_bytes")


@pytest.mark.asyncio
async def test_paddle_construction_failure_is_cached():
    """构造失败缓存：首次抛 OCRNotAvailable 后，后续调用不再重复构造。"""
    fake_ocr = _install_fake_modules()
    fake_ocr.PaddleOCR.side_effect = Exception("model download failed")
    engine = PaddleOCREngine()
    with pytest.raises(OCRNotAvailable):
        await engine.ocr(b"img1")
    with pytest.raises(OCRNotAvailable):
        await engine.ocr(b"img2")
    assert fake_ocr.PaddleOCR.call_count == 1  # 失败缓存，不重试构造


@pytest.mark.asyncio
async def test_paddle_engine_instance_is_cached():
    """可用性与实例构造只发生一次，predict 每次调用。"""
    fake_ocr = _install_fake_modules()
    engine = PaddleOCREngine()
    fake_ocr.PaddleOCR.return_value.predict.return_value = [{"rec_texts": ["x"]}]
    await engine.ocr(b"img1")
    await engine.ocr(b"img2")
    assert fake_ocr.PaddleOCR.call_count == 1  # 只构造一次
    assert fake_ocr.PaddleOCR.return_value.predict.call_count == 2


@pytest.mark.asyncio
async def test_paddle_raises_ocronotavailable_when_cv2_missing():
    """cv2 缺失（sys.modules 置 None）→ OCRNotAvailable，且不执行构造。"""
    fake_ocr = _install_fake_modules()
    engine = PaddleOCREngine()
    with patch.dict(sys.modules, {"cv2": None}):
        with pytest.raises(OCRNotAvailable):
            await engine.ocr(b"fake_image_bytes")
    assert fake_ocr.PaddleOCR.call_count == 0  # 可用性检查先失败，不构造


@pytest.mark.asyncio
async def test_get_engine_returns_process_singleton():
    """get_engine 返回进程级单例（模型/失败缓存跨文档复用）。"""
    from aion_knowledge.infrastructure.ocr.paddle import get_engine

    assert get_engine() is get_engine()
