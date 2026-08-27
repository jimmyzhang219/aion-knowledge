"""PaddleOCR 本地 OCR 引擎（onnxruntime 后端，无需安装 paddlepaddle）。

依赖（可选组 paddle-ocr）：paddleocr>=3.7.0 + onnxruntime + opencv-contrib-python。
未安装时抛 OCRNotAvailable，由调用方降级到 Tesseract。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from aion_knowledge.infrastructure.ocr.engine import OCREngine, OCRNotAvailable

logger = logging.getLogger(__name__)


class PaddleOCREngine(OCREngine):
    """PaddleOCR 3.x 引擎 — onnxruntime 推理后端。

    构造与推理均为同步重操作（首次构造会下载模型），放入线程池；
    paddleocr 懒加载，未安装时抛 OCRNotAvailable。
    """

    def __init__(self) -> None:
        self._ocr: Any | None = None  # 懒加载的 PaddleOCR 实例
        self._available: bool | None = None

    async def _ensure_available(self) -> None:
        if self._available is not None:
            if not self._available:
                raise OCRNotAvailable("PaddleOCR not available")
            return
        try:
            import cv2  # noqa: F401  # paddle-ocr 组提供，仅作可用性探测
            from paddleocr import (  # 未安装时抛 ImportError
                PaddleOCR,
            )

            # 构造会首次下载模型（~/.paddlex/official_models/）并加载，为同步重操作
            self._ocr = await asyncio.to_thread(
                PaddleOCR,
                lang="ch",
                engine="onnxruntime",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
            self._available = True
        except Exception as e:
            logger.warning("PaddleOCR not available: %s", e)
            self._available = False  # 失败缓存，进程内不再重试
            raise OCRNotAvailable("PaddleOCR unavailable") from e

    async def ocr(self, image_bytes: bytes) -> str:
        await self._ensure_available()
        ocr = self._ocr
        assert ocr is not None  # _ensure_available 成功即已构造

        def _run() -> str:
            import cv2  # 可选依赖（paddle-ocr 组），未安装时走降级
            import numpy as np

            img = cv2.imdecode(
                np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            results = ocr.predict(input=img)
            texts = [t for res in results for t in res["rec_texts"]]
            return "\n".join(texts).strip()

        return await asyncio.to_thread(_run)


_engine: PaddleOCREngine | None = None


def get_engine() -> PaddleOCREngine:
    """返回进程级 PaddleOCR 引擎单例。

    模块实例按文档创建，若引擎实例只挂在模块上，失败缓存与模型加载
    会随文档生命周期重置（每文档重载模型/重复下载失败）。进程级单例
    保证「构造失败后不再重试」语义在整个进程生效。
    """
    global _engine
    if _engine is None:
        _engine = PaddleOCREngine()
    return _engine
