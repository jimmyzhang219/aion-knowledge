"""文件名后缀推断工具。"""

from __future__ import annotations

import mimetypes


def infer_suffix(
    filename: str | None = None,
    content_type: str | None = None,
    default: str = "txt",
) -> str:
    """从文件名或 MIME 类型推断文件后缀。

    优先级：filename 扩展名 → MIME → default

    用法：
        suffix = infer_suffix(filename="test.pdf")           # "pdf"
        suffix = infer_suffix(content_type="text/csv")       # "csv"
        suffix = infer_suffix(filename="unknown")             # "txt"
    """
    if filename and "." in filename:
        ext = filename.rsplit(".", 1)[-1].lower()
        if ext:
            return ext
    if content_type:
        guessed = mimetypes.guess_extension(content_type)
        if guessed:
            return guessed.lstrip(".")
    return default
