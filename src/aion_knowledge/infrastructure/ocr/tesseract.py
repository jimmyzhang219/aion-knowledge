"""Tesseract 本地 OCR 引擎。"""

from __future__ import annotations

import asyncio
import logging
from io import BytesIO
from typing import cast

from aion_knowledge.infrastructure.ocr.engine import OCREngine, OCRNotAvailable

logger = logging.getLogger(__name__)


class TesseractOCREngine(OCREngine):
    """本地 Tesseract OCR 引擎。"""

    def __init__(self) -> None:
        self._available: bool | None = None

    async def _check_available(self) -> None:
        if self._available is not None:
            return
        try:
            import pytesseract  # type: ignore[import-untyped]  # pytesseract 无 stubs

            pytesseract.get_tesseract_version()
            self._available = True
        except Exception as e:
            logger.warning("Tesseract not available: %s", e)
            self._available = False
            raise OCRNotAvailable("Tesseract binary not found") from e

    async def ocr(self, image_bytes: bytes) -> str:
        await self._check_available()
        import pytesseract
        from PIL import Image

        # Tesseract + PIL 为同步阻塞调用，放入线程池避免阻塞 event loop
        def _run() -> str:
            img = Image.open(BytesIO(image_bytes))
            # pytesseract 无 stubs，返回类型为 Any，收敛为 str
            return cast(str, pytesseract.image_to_string(img)).strip()

        return await asyncio.to_thread(_run)
