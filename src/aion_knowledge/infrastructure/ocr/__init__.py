"""OCR 基础设施 — PaddleOCR + Tesseract + VLM 降级 OCR 引擎。"""

from aion_knowledge.infrastructure.ocr.engine import OCREngine, OCRNotAvailable
from aion_knowledge.infrastructure.ocr.paddle import PaddleOCREngine
from aion_knowledge.infrastructure.ocr.sanitizer import sanitize_ocr
from aion_knowledge.infrastructure.ocr.tesseract import TesseractOCREngine
from aion_knowledge.infrastructure.ocr.vlm_ocr import VLMOCREngine

__all__ = [
    "OCREngine",
    "OCRNotAvailable",
    "sanitize_ocr",
    "PaddleOCREngine",
    "TesseractOCREngine",
    "VLMOCREngine",
]
