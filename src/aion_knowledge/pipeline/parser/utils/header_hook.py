"""Markdown 标题追踪器，用于拆分时上下文保持。"""
import re
from typing import List


def header_column_mismatch(header: str, content: str) -> bool:
    """检查 header 的列数与 content 表列数是否匹配。"""
    h_cols = header.count("|") - 1
    if h_cols <= 0:
        return False
    c_lines = [ln for ln in content.split("\n") if "|" in ln]
    if not c_lines:
        return False
    c_cols = c_lines[0].count("|") - 1
    return h_cols != c_cols


class HeaderTracker:
    """追踪 markdown 标题层级。"""

    HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    def __init__(self) -> None:
        self.headers: List[str] = []
        self.header_ended_this_unit: bool = False

    def update(self, text: str) -> None:
        self.header_ended_this_unit = False
        for match in self.HEADER_RE.finditer(text):
            self.headers.append(match.group(0))
            self.header_ended_this_unit = True

    def get_headers(self) -> str:
        return " ".join(self.headers[-3:]) if self.headers else ""
