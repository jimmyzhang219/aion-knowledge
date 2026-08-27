"""Chunker 工具函数：token 计数、语义对齐、正则常量。

=== 正则常量 ===

chunker 模块使用一组正则常量来检测文档中的结构元素。所有常量定义在此，
供 ``document_analysis.py``、``marker_splitter.py``、``heading_splitter.py``
等文件共享。

===== ============================ ==================================
常量名                           用途
===== ============================ ==================================
``MARKDOWN_HEADING_RE``          匹配 ``#`` ~ ``######`` 的 Markdown 标题
``NUMBERED_SECTION_RE``          匹配 "1.1 简介" / "IV. Results" 等编号标题
``ALL_CAPS_HEADING_RE``          匹配 "INTRODUCTION" 等全大写短标题（4-80 字符）
``VISUAL_SEPARATOR_RE``          匹配 ``---`` / ``===`` / ``***`` / ``___``
``EXCESSIVE_BLANKS_RE``          匹配 3 个或以上连续换行（空段落）
``PAGE_FOOTER_RE``               匹配 "Seite 1 von 10" 等页脚页码行
``CHAPTER_RE``                   多语言章节关键词（"Kapitel 1" / "Chapter 2" / "第3章"）
===== ============================ ==================================

=== Token 计数 ===

``count_tokens()`` 使用 tiktoken cl100k_base（与 GPT-4 / text-embedding-3 一致）。
当 tiktoken 不可用时（离线环境/受限依赖），回退到语言感知的字符比率估算：
- 中文：1 token ≈ 1.7 字符
- 英文：1 token ≈ 4.0 字符
- 德文：1 token ≈ 4.5 字符
- 混合（默认）：1 token ≈ 3.0 字符

=== 语义对齐 ===

``align_overlap()`` 将计算出的重叠起始位置对齐到最近的换行符，
防止从行中间截断内容（避免如 "前面是算法实" 这种语义截断）。
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


def count_tokens(text: str, language: str = "mixed") -> int:
    """计算文本的 token 数（cl100k_base 编码）。

    .. deprecated::
        请直接使用 ``aion_knowledge.common.model_registry.count_tokens``。
        此函数委托给 ``common.model_registry`` 以确保整个代码库使用
        同一份 token 计数实现。

    Args:
        text: 待计数的文本
        language: tiktoken 不可用时的回退语言模式（"zh" / "en" / "de" / "mixed"）

    Returns:
        token 数量（>= 0）
    """
    from aion_knowledge.common.model_registry import count_tokens as _count
    return _count(text, language)


def count_tokens_batch(texts: List[str], language: str = "mixed") -> List[int]:
    """批量计算 token 数。

    .. deprecated::
        请直接使用 ``aion_knowledge.common.model_registry.count_tokens_batch``。
    """
    from aion_knowledge.common.model_registry import count_tokens_batch as _count_batch
    return _count_batch(texts, language)


# ── 正则常量 ──

# Markdown 标题
MARKDOWN_HEADING_RE = r"^(#{1,6})\s+(.+?)\s*#*\s*$"

# 编号章节，如 "1.1 简介"、"2.3 Methods"、"IV. Results"
NUMBERED_SECTION_RE = r"^(?:\d+(?:\.\d+){1,3}\.{0,1}|(?:\d+|[IVX]{1,5})\.)\s+\S"

# 全大写短标题
ALL_CAPS_HEADING_RE = r"^[A-ZÄÖÜ][A-ZÄÖÜ \-]{3,80}:?\s*$"

# 视觉分隔线 ---/===/***/___
VISUAL_SEPARATOR_RE = r"^(?:-{3,}|={3,}|\*{3,}|_{3,})\s*$"

# 过多空行（3+）
EXCESSIVE_BLANKS_RE = r"\n{3,}"

# 页脚 "页码" 行
PAGE_FOOTER_RE = r"^(?:Seite|Page|页码?)\s+\d+(?:\s*(?:von|of|/)\s*\d+)?\s*$"

# 多语言章节关键词
CHAPTER_RE = {
    "de": r"^(?:Kapitel|Abschnitt|Teil)\s+(?:\d+|[IVX]{1,5})[\.:] ",
    "en": r"^(?:Chapter|Section|Part)\s+(?:\d+|[IVX]{1,5})[\.:] ",
    "zh": r"^第\s*[一二三四五六七八九十百千零〇0-9]+\s*(?:章|节|節|部分|篇)",
}


def align_overlap(text: str, target_start: int, min_start: int) -> int:
    """将重叠起始位置对齐到最近的换行符，避免截断行。

    Args:
        text: 完整文本
        target_start: 理想起始位置（字符偏移）
        min_start: 允许的最小起始位置

    Returns:
        对齐后的起始位置（字符偏移）
    """
    if target_start <= 0:
        return 0

    pos = target_start
    while pos > min_start and pos > 0:
        if text[pos - 1] == "\n":
            return pos
        pos -= 1
    return target_start
