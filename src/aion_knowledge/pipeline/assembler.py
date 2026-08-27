"""Assembler — 按 seq_num 排序合并所有 chunk 类型。"""

from __future__ import annotations

from typing import Any


def assemble(
    text_chunks: list[dict[str, Any]],
    tables: list[dict[str, Any]],
    images: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """按 seq_num 排序合并所有 chunk 类型。

    Args:
        text_chunks: 文本块列表。
        tables: 表格块列表。
        images: 图片块列表。

    Returns:
        按 seq_num 升序排列的合并列表。
    """
    all_chunks = text_chunks + tables + images
    all_chunks.sort(key=lambda c: c["seq_num"])
    return all_chunks
