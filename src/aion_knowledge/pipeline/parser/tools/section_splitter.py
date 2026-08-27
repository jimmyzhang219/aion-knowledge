"""按图片标记切分文档 sections，附着 context_above/below。"""
from __future__ import annotations

import re
from typing import Any

IMG_PATTERN = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')


def split_sections(content: str) -> list[dict[str, Any]]:
    """将 Markdown 按图片标记切分为 sections。

    返回: [{"type": "text", "content": "..."}, {"type": "image", "image_url": "...", "content": ""}, ...]
    """
    sections: list[dict[str, Any]] = []
    pos = 0
    for match in IMG_PATTERN.finditer(content):
        start, end = match.start(), match.end()
        if pos < start:
            text = content[pos:start].strip()
            if text:
                sections.append({"type": "text", "content": text, "image_url": None})
        sections.append({
            "type": "image",
            "content": "",
            "image_url": match.group(2),
        })
        pos = end
    remaining = content[pos:].strip()
    if remaining:
        sections.append({"type": "text", "content": remaining, "image_url": None})

    for i, sec in enumerate(sections):
        if sec["type"] == "image":
            for j in range(i - 1, -1, -1):
                if sections[j]["type"] == "text" and sections[j]["content"]:
                    sec["context_above"] = _align_start_to_boundary(
                        sections[j]["content"][-500:]
                    )
                    break
            for j in range(i + 1, len(sections)):
                if sections[j]["type"] == "text" and sections[j]["content"]:
                    sec["context_below"] = _align_end_to_boundary(
                        sections[j]["content"][:500]
                    )
                    break

    return sections


def _align_start_to_boundary(text: str) -> str:
    """将截取片段向前对齐到最近的段落/句子边界，避免断句开头。"""
    # 优先段落边界
    for sep in ("\n\n", "\n", "。", "！", "？", "；"):
        idx = text.find(sep)
        if idx != -1:
            return text[idx + len(sep):]
    return text


def _align_end_to_boundary(text: str) -> str:
    """将截取片段向后对齐到最近的段落/句子边界，避免断句结尾。"""
    # 优先段落边界，从后往前找
    for sep in ("\n\n", "\n", "。", "！", "？", "；"):
        idx = text.rfind(sep)
        if idx != -1:
            return text[:idx + len(sep)]
    return text
