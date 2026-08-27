"""Cleaner —— 文档清洗模块。

在 Parser（文档解析）和 Chunker（文档切分）之间对 Markdown 文本做统一清洗。

职责范围：
    - 字符级净化：Unicode 归一化、删除零宽字符/控制字符、全角→半角转换
    - 空白字符规整：统一换行符、删除行尾空格、合并空行、裁剪文件首尾空行
    - Markdown 格式归一： 确保标题/列表/代码块前后空行一致性
    - 文档噪声过滤：删除页码、分隔线等非语义元素

架构：
    Cleaner 是门面类，按 4 阶段流水线执行所有清洗步骤。每条规则是独立纯函数，
    无状态依赖，可单独通过 clean_passage() 调用。

位置：Parser → **Cleaner** → Chunker
"""

import logging

from aion_knowledge.pipeline.cleaner.cleaner import (
    clean_passage,
    collapse_blank_lines,
    ensure_code_block_spacing,
    ensure_heading_spacing,
    ensure_list_spacing,
    fullwidth_to_halfwidth,
    normalize_line_endings,
    normalize_unicode,
    remove_control_chars,
    remove_page_numbers,
    remove_separators,
    remove_zero_width,
    strip_trailing_whitespace,
    trim_file_ends,
)

logger = logging.getLogger(__name__)


class Cleaner:
    """文档清洗门面。

    按固定流水线顺序执行所有清洗步骤。
    每条规则是独立纯函数，无状态依赖。
    """

    def clean(self, text: str) -> str:
        """执行完整清洗流水线。"""
        if not text:
            return text

        # Step 1：字符级净化
        text = normalize_unicode(text)
        text = remove_zero_width(text)
        text = remove_control_chars(text)
        text = fullwidth_to_halfwidth(text)

        # Step 2：空白字符规整
        text = normalize_line_endings(text)
        text = strip_trailing_whitespace(text)
        text = collapse_blank_lines(text)
        text = trim_file_ends(text)

        # Step 3：Markdown 格式归一
        text = ensure_heading_spacing(text)
        text = ensure_list_spacing(text)
        text = ensure_code_block_spacing(text)

        # Step 4：文档噪声过滤
        text = remove_page_numbers(text)
        text = remove_separators(text)

        return text


__all__ = [
    "Cleaner",
    "clean_passage",
]
