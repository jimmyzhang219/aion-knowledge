"""社区报告文本拼接 —— processor 嵌入端与 retriever 展示端共用，保证同源。"""
from __future__ import annotations

from typing import Any


def build_community_text(title: str, summary: str, findings: list[dict[str, Any]] | None) -> str:
    """社区报告 → 文本（title + summary + findings，空片段跳过）。

    Args:
        title: 社区标题（payload.title）。
        summary: 社区摘要（summary 列）。
        findings: 发现列表（findings JSONB，元素含 summary/explanation）。

    Returns:
        拼接文本；全部为空时返回空串。
    """
    parts = [title, summary]
    parts.extend(
        f"{f.get('summary', '')}: {f.get('explanation', '')}"
        for f in (findings or [])
    )
    return "\n".join(p for p in parts if p)
