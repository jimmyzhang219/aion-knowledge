import io
import logging
import re
from typing import Any, cast

from markitdown import DocumentConverterResult, MarkItDown

from aion_knowledge.common.config import settings
from aion_knowledge.common.uuid7 import uuid7
from aion_knowledge.pipeline.parser.base import BaseParser, ParsedDocument
from aion_knowledge.pipeline.parser.chain import PipelineParser
from aion_knowledge.pipeline.parser.concurrency import parser_worker_limit
from aion_knowledge.pipeline.parser.image_extractor import (
    attach_pptx_media_to_markdown,
    markdown_needs_pptx_media_attach,
)
from aion_knowledge.pipeline.parser.markdown import MarkdownParser

logger = logging.getLogger(__name__)

# 匹配 Markdown 中的 data: URI 图片引用
_DATA_URI_PATTERN = re.compile(r'!\[([^\]]*)\]\((data:image/[^;]+;base64,([^)]+))\)')

# MIME → 扩展名映射
_MIME_EXT = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/svg+xml": ".svg",
}


def _extract_data_uris(text: str) -> tuple[str, dict[str, str]]:
    """扫描 text 中的 data:image/*;base64,... URI，提取为 images 字典并替换为 ref key。

    Returns:
        (清理后的 text, {ref: base64_data})
    """
    images: dict[str, str] = {}
    resolved: dict[str, str] = {}  # 原始 URI → ref_key（去重）

    def repl(match: re.Match[str]) -> str:
        alt = match.group(1)
        uri = match.group(2)
        b64_data = match.group(3)

        # 跳过空数据或已处理的重复 URI
        if not b64_data or uri in resolved:
            if uri in resolved:
                return f"![{alt}]({resolved[uri]})"
            return match.group(0)

        # 从 MIME 推断扩展名
        mime = uri[5:].split(";")[0] if ";" in uri else uri[5:]
        ext = _MIME_EXT.get(mime, ".png")
        ref_key = f"images/{uuid7().hex}{ext}"

        images[ref_key] = b64_data
        resolved[uri] = ref_key

        logger.info(
            "图片 data URI 提取: %s → %s (%s, %d bytes)",
            uri[:60], ref_key, mime, len(b64_data),
        )
        return f"![{alt}]({ref_key})"

    text = _DATA_URI_PATTERN.sub(repl, text)
    return text, images


class StdMarkitdownParser(BaseParser):
    """标准 MarkItDown 解析器包装器。

    该解析器使用 markitdown 库将各种文档格式
    （docx、pptx、pdf 等）转换为文本/Markdown。
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # 这里的 super() 会调用 BaseParser 的初始化，确保 self.file_type 被正确赋值
        super().__init__(*args, **kwargs)
        self.markitdown = MarkItDown()

    def parse_into_text(self, content: bytes) -> ParsedDocument:
        """使用 MarkItDown 解析内容。

        使用 self.file_type（继承自 BaseParser）来提示流格式。
        """
        # 延迟导入：normalize_ppt_bytes 由 Office 解析器模块提供
        from aion_knowledge.pipeline.parser.office import normalize_ppt_bytes

        ext = self.file_type
        ft = (ext or "").lstrip(".").lower()
        pptx_bytes: bytes | None = None
        if ft in ("ppt", "pptx"):
            content, ext = normalize_ppt_bytes(content, ft)
            pptx_bytes = content
            ft = "pptx"
        elif ext and not ext.startswith("."):
            ext = "." + ext

        with parser_worker_limit("markitdown", settings.markitdown_max_workers):
            # 用 keep_data_uris=True 保留 base64 图片，后续统一提取到 images 字典
            result = self._convert_markitdown(content, ext, keep_data_uris=True)
            if result is None:
                logger.warning(
                    "MarkItDown failed with keep_data_uris=True for %s; "
                    "retrying without data URIs (image data will be lost)",
                    ft or ext,
                )
                result = self._convert_markitdown(content, ext, keep_data_uris=False)

        # 重试后仍失败时 result 为 None，原实现同样在访问 .text_content 时抛 AttributeError，
        # 此处 cast 仅消除类型窄化，保持原崩溃行为不变
        text = cast(DocumentConverterResult, result).text_content
        images: dict[str, str] = {}

        # 从 markitdown 输出中提取 data: URI 图片 → images 字典 → 替换为 ref key
        try:
            text, images = _extract_data_uris(text)
        except Exception as exc:
            logger.warning("提取 data URI 图片失败: %s", exc)

        if pptx_bytes is not None and markdown_needs_pptx_media_attach(text):
            text, images = attach_pptx_media_to_markdown(text, pptx_bytes)
        return ParsedDocument(content=text, images=images)

    def _convert_markitdown(
        self, content: bytes, ext: str | None, *, keep_data_uris: bool
    ) -> DocumentConverterResult | None:
        """调用 markitdown 转换；失败返回 None（调用方据此回退重试）。"""
        try:
            return self.markitdown.convert(
                io.BytesIO(content),
                file_extension=ext,
                keep_data_uris=keep_data_uris,
            )
        except Exception:
            if keep_data_uris:
                return None
            raise


class MarkitdownParser(PipelineParser):
    _parser_cls = (StdMarkitdownParser, MarkdownParser)
