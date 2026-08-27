"""VLM 降级 OCR 引擎。"""

from __future__ import annotations

import logging

from aion_knowledge.infrastructure.llm import LLMClient
from aion_knowledge.infrastructure.ocr.engine import OCREngine
from aion_knowledge.infrastructure.ocr.sanitizer import sanitize_ocr

logger = logging.getLogger(__name__)

# 扫描 PDF OCR prompt
_OCR_PROMPT = (
    "<system_prompt>\n"
    "你是一个 OCR 与文档版面分析助手。"
    "输入图片来自一份扫描版 PDF 文档的页面。\n"
    "你的任务是从图片中仔细提取所有文字内容和版面结构，"
    "并以纯 Markdown 格式输出结果。\n"
    "</system_prompt>\n\n"
    "<instructions>\n"
    "1. 忽略页眉、页脚和页码。\n"
    "2. 尽量保留原文档的段落和层级结构。\n"
    "3. 如果有表格，使用 Markdown 表格语法表示。\n"
    "4. 如果有数学公式，使用 $ 或 $$ 包裹的 LaTeX 格式。\n"
    "5. 仅输出提取的文字内容，不要包含任何 HTML 标签、推理过程或不相关的注释。\n"
    "6. 如果图片中完全没有可识别的文字内容，仅回复：无文字内容。\n"
    "</instructions>"
)


class VLMOCREngine(OCREngine):
    """VLM 降级 OCR 引擎 — 复用现有 VLM 客户端。"""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def ocr(self, image_bytes: bytes) -> str:
        logger.debug("VLMOCR: starting OCR via VLM client")
        mime = "image/jpeg"
        result = await self._llm.generate_with_images(
            prompt=_OCR_PROMPT,
            images=[(image_bytes, mime)],
            max_tokens=4096,
        )
        cleaned = sanitize_ocr(result)
        if cleaned:
            logger.debug("VLMOCR: extracted %d chars", len(cleaned))
        else:
            logger.info("VLMOCR: no text content extracted")
        return cleaned
