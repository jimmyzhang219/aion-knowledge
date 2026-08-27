"""OCR 引擎抽象基类与异常。"""

from __future__ import annotations

from abc import ABC, abstractmethod

from aion_knowledge.common.errors import AionError


class OCRNotAvailable(AionError):  # noqa: N818  # 名称沿用既有引用面（29 处），重命名超出本次范围
    """OCR 引擎不可用时抛出（如 PaddleOCR / Tesseract 未安装）。"""


class OCREngine(ABC):
    """OCR 引擎抽象接口。"""

    @abstractmethod
    async def ocr(self, image_bytes: bytes) -> str:
        """提取图片文字，返回 Markdown 文本。

        返回空字符串表示图片中无文字内容。
        引擎不可用时抛出 OCRNotAvailable。
        """
        ...
