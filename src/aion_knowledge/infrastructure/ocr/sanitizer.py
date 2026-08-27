"""OCR 后处理清洗"""

from __future__ import annotations

import logging
import re

from markdownify import markdownify as md

logger = logging.getLogger(__name__)

_KNOWN_EMPTY_REPLIES = [
    "无文字内容", "无法识别", "no text", "no text content",
    "no content", "empty", "图片中没有文字", "图片中没有可识别的文字",
]

# 英文短语使用词边界匹配避免误杀合法文字
_EMPTY_PATTERNS: list[re.Pattern[str]] = []
for _p in _KNOWN_EMPTY_REPLIES:
    if _p.isascii():
        _EMPTY_PATTERNS.append(re.compile(rf"\b{re.escape(_p)}\b", re.I))
    else:
        _EMPTY_PATTERNS.append(re.compile(re.escape(_p), re.I))

_CODE_BLOCK_PATTERN = re.compile(r"^\s*```[a-zA-Z]*\s*\n(.*?)\n\s*```\s*$", re.S)
_HTML_DOC_PATTERN = re.compile(
    r"(?i)^\s*(<!DOCTYPE|<html|<body|<div|<p[\s>]|<table|<h[1-6][\s>])"
)
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
_MULTIPLE_NEWLINES = re.compile(r"\n{3,}")


def sanitize_ocr(raw: str | None) -> str:
    """清洗 VLM/Tesseract OCR 输出，返回纯净 Markdown 文本。"""
    if not raw:
        return ""
    text = raw.strip()
    if not text:
        return ""

    # 1. 剥离 Markdown 代码围栏
    text = _strip_code_block(text)

    # 2. 检测空 HTML 骨架（仅标签、无实质文本）
    plain = _HTML_TAG_PATTERN.sub("", text).strip()
    if len(plain) < 2 and _HTML_TAG_PATTERN.search(text):
        return ""

    # 3. HTML → Markdown 转换
    if _looks_like_html(text):
        text = _convert_html_to_md(text).strip()
        if not text:
            return ""

    # 4. 空回复检测
    if _is_empty_reply(text):
        return ""

    # 5. 连续换行压缩
    text = _MULTIPLE_NEWLINES.sub("\n\n", text)
    return text.strip()


def _strip_code_block(text: str) -> str:
    m = _CODE_BLOCK_PATTERN.match(text)
    return m.group(1).strip() if m else text


def _looks_like_html(text: str) -> bool:
    if _HTML_DOC_PATTERN.match(text):
        return True
    tags = _HTML_TAG_PATTERN.findall(text)
    if not tags:
        return False
    tag_chars = sum(len(t) for t in tags)
    return tag_chars / len(text) > 0.3


def _convert_html_to_md(content: str) -> str:
    try:
        return md(content)
    except Exception as e:
        logger.warning("markdownify conversion failed: %s", e)
        return content


def _is_empty_reply(text: str) -> bool:
    check = text.rstrip(".!?。！？")
    for pat in _EMPTY_PATTERNS:
        if pat.search(check):
            return True
    return False
