"""Pipeline 层测试辅助夹具。"""

from __future__ import annotations

from pathlib import Path

import pytest_asyncio


@pytest_asyncio.fixture
async def temp_markdown_file(tmp_path: Path) -> str:
    """创建一个临时的 Markdown 文件并返回路径。"""
    content = "# Title 1\n\nHello world.\n\n## Section 1\n\nSome text here.\n\n### Subsection\n\nMore details.\n\n## Section 2\n\nFinal paragraph."
    fpath = tmp_path / "test.md"
    fpath.write_text(content, encoding="utf-8")
    return str(fpath)


@pytest_asyncio.fixture
async def temp_raw_file(tmp_path: Path) -> str:
    """创建一个临时的原始文件（模拟下载的 PDF 等）。"""
    content = b"dummy raw file content for testing"
    fpath = tmp_path / "raw.pdf"
    fpath.write_bytes(content)
    return str(fpath)
