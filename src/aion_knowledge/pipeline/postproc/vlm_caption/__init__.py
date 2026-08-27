"""VLM 图片描述 + OCR 模块。

对包含图片引用的 chunk 执行 VLM Caption 和 OCR（PaddleOCR → Tesseract → VLM 三级降级），
将描述文字和 OCR 结果合并回 chunk_text.content。
"""
