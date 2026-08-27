"""纯函数清洗步骤 — 每个函数接收文本并返回清洗后的文本。"""

import logging
import re
import unicodedata

from aion_knowledge.pipeline.cleaner.rules import (
    COLLAPSE_BLANKS_RE,
    CONTROL_CHAR_RE,
    FENCED_CODE_RE,
    FULLWIDTH_END,
    FULLWIDTH_OFFSET,
    FULLWIDTH_START,
    HEADING_RE,
    LIST_ITEM_RE,
    PAGE_NUMBER_RE,
    SEPARATOR_RE,
    TRAILING_SPACE_RE,
    ZERO_WIDTH_CHARS,
)

logger = logging.getLogger(__name__)


# ── Step 1：字符级净化 ──────────────────────────────────────────────


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def remove_zero_width(text: str) -> str:
    return text.translate(str.maketrans("", "", ZERO_WIDTH_CHARS))


def remove_control_chars(text: str) -> str:
    return CONTROL_CHAR_RE.sub("", text)


def fullwidth_to_halfwidth(text: str) -> str:
    result = []
    for ch in text:
        code = ord(ch)
        if FULLWIDTH_START <= code <= FULLWIDTH_END:
            result.append(chr(code - FULLWIDTH_OFFSET))
        else:
            result.append(ch)
    return "".join(result)


# ── Step 2：空白字符规整 ──────────────────────────────────────────


def normalize_line_endings(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def strip_trailing_whitespace(text: str) -> str:
    return TRAILING_SPACE_RE.sub("", text)


def collapse_blank_lines(text: str) -> str:
    return COLLAPSE_BLANKS_RE.sub("\n\n\n", text)


def trim_file_ends(text: str) -> str:
    return text.strip("\n")


# ── Step 3：Markdown 格式归一 ─────────────────────────────────────


def ensure_heading_spacing(text: str) -> str:
    lines = text.split("\n")
    result: list[str] = []
    for i, line in enumerate(lines):
        if i > 0 and HEADING_RE.match(line) and result[-1] != "":
            result.append("")
        result.append(line)
    return "\n".join(result)


def ensure_list_spacing(text: str) -> str:
    lines = text.split("\n")
    result: list[str] = []
    in_list = False
    for line in lines:
        is_list_item = bool(LIST_ITEM_RE.match(line))
        is_fence = bool(FENCED_CODE_RE.match(line))
        if not in_list and is_list_item:
            if result and result[-1] != "":
                result.append("")
            in_list = True
        elif in_list and not is_list_item and not is_fence:
            if line != "":
                result.append("")
            in_list = False
        elif in_list and not is_list_item and is_fence:
            in_list = False
        result.append(line)
    return "\n".join(result)


def ensure_code_block_spacing(text: str) -> str:
    lines = text.split("\n")
    result: list[str] = []
    in_code_block = False
    for line in lines:
        if FENCED_CODE_RE.match(line):
            if not in_code_block:
                if result and result[-1] != "":
                    result.append("")
                in_code_block = True
            else:
                in_code_block = False
            result.append(line)
        else:
            result.append(line)
    return "\n".join(result)


# ── Step 4：文档噪声过滤 ─────────────────────────────────────────


def remove_page_numbers(text: str) -> str:
    return PAGE_NUMBER_RE.sub("", text)


def remove_separators(text: str) -> str:
    return SEPARATOR_RE.sub("", text)


# ── Step 5：Markdown 语法剥离（入库时清理，供 LLM 消费）───────────

_MARKDOWN_PATTERNS: list[tuple[str, str]] = [
    (r"```[\s\S]*?```", ""),          # 代码块
    (r"\|", ""),                       # 表格单元格标记 → 删除 |，保留内容
    (r"\[([^\]]*)\]\([^)]*\)", r"\1"),  # 链接 → 文本
    (r"!\[.*?\]\(.*?\)", ""),           # 图片
    (r"<[^>]+>", ""),                   # HTML 标签
    (r"`([^`]+)`", r"\1"),              # 行内代码 → 文本
    (r"#{1,6}\s*", ""),                 # 标题标记
    (r"^\s*[-*+]\s+", ""),              # 无序列表
    (r"^\s*\d+\.\s+", ""),              # 有序列表
    (r"\n{3,}", "\n\n"),                # 多余空行
]


def clean_passage(text: str) -> str:
    """清理 Markdown 语法，保留纯文本内容。"""
    for pattern, replacement in _MARKDOWN_PATTERNS:
        text = re.sub(pattern, replacement, text, flags=re.MULTILINE)
    return text.strip()
