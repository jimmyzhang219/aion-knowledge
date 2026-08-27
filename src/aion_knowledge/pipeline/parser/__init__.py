"""Parser —— 文档解析器模块。

将多种格式的原始文档解析为 Markdown 文本 + 内嵌图片的 ParsedDocument。

支持的文件格式：
    - 文本类：Markdown (.md)
    - 办公文档：Word (.doc/.docx)、Excel (.xls/.xlsx)、PowerPoint (.ppt/.pptx)
    - PDF：内置解析、MarkItDown、OpenDataLoader（版面分析）、外部服务（MinerU / PaddleOCR-VL）
    - 电子书：EPUB
    - 网页归档：MHTML
    - 图片：jpg/png/gif/bmp/tiff/webp
    - 网页：按 URL 实时抓取

引擎架构：
    通过 ParserEngineRegistry 按文件类型路由到具体解析器。支持多引擎共存，
    可通过 parser_engine 参数切换引擎。

位置：**Parser** → Cleaner → Chunker

用法：
    from aion_knowledge.pipeline.parser import Parser

    parser = Parser()
    doc = parser.parse_file("report.pdf", "pdf", content)
    # doc.content: markdown text
    # doc.images: {ref_path: base64_data}

    doc = parser.parse_url("https://example.com", title="Example")

子模块：
    - base.py:      基础解析器接口、ParsedDocument 数据类
    - chain.py:     解析链（FirstParser / PipelineParser 组合模式）
    - concurrency:  并发控制
    - pdf.py:       PDF 解析（PyMuPDF + 扫描件 OCR）
    - office.py:    Office 文档解析（python-docx / openpyxl / python-pptx / antiword）
    - markdown.py:  Markdown 直接读取
    - epub.py:      EPUB 电子书解析
    - mhtml.py:     MHTML 网页归档解析
    - web.py:       网页实时抓取与清洗
    - external_pdf.py: 外部 PDF 服务调用
    - markitdown.py:   MarkItDown 引擎封装
    - opendataloader.py: OpenDataLoader 版面分析引擎
    - registry.py:  引擎注册表
    - utils/:       图片下载、URL 解析等工具函数
"""

import logging
from typing import Any

from aion_knowledge.pipeline.parser.base import BaseParser, ParsedDocument
from aion_knowledge.pipeline.parser.chain import FirstParser, PipelineParser
from aion_knowledge.pipeline.parser.concurrency import parser_worker_limit
from aion_knowledge.pipeline.parser.epub import EPUBParser
from aion_knowledge.pipeline.parser.external_pdf import ExternalPdfParser
from aion_knowledge.pipeline.parser.image import ImageParser
from aion_knowledge.pipeline.parser.image_extractor import (
    attach_pptx_media_to_markdown,
    extract_pptx_media_rasterized,
    list_pptx_media,
    rasterize_media_bytes,
)
from aion_knowledge.pipeline.parser.markdown import MarkdownParser
from aion_knowledge.pipeline.parser.markitdown import MarkitdownParser
from aion_knowledge.pipeline.parser.mhtml import MHTMLParser
from aion_knowledge.pipeline.parser.office import (
    DocParser,
    Docx2Parser,
    DocxParser,
    ExcelParser,
    convert_ppt_to_pptx_bytes,
    fill_merged_cells_xlsx,
    normalize_ppt_bytes,
    repair_xlsx_bytes,
)
from aion_knowledge.pipeline.parser.opendataloader import (
    OpenDataLoaderParser,
    opendataloader_available,
)
from aion_knowledge.pipeline.parser.pdf import PDFParser, PDFScannedParser
from aion_knowledge.pipeline.parser.registry import (
    BUILTIN_ENGINE,
    ParserEngineRegistry,
)
from aion_knowledge.pipeline.parser.web import WebParser

logger = logging.getLogger(__name__)


# ── registry 实例 ──────────────────────────────────────────────────────

registry = ParserEngineRegistry()


# ── 注册默认引擎 ──────────────────────────────────────────────────────

image_types = {ext: ImageParser for ext in ("jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp")}

registry.register(BUILTIN_ENGINE, {
    "docx": Docx2Parser,
    "doc": DocParser,
    "pdf": PDFParser,
    "md": MarkdownParser,
    "markdown": MarkdownParser,
    "xlsx": ExcelParser,
    "xls": ExcelParser,
    "epub": EPUBParser,
    "mhtml": MHTMLParser,
    **image_types,
}, description="内置解析引擎")

registry.register("markitdown", {
    "md": MarkitdownParser,
    "markdown": MarkitdownParser,
    "pdf": MarkitdownParser,
    "docx": MarkitdownParser,
    "doc": MarkitdownParser,
    "pptx": MarkitdownParser,
    "ppt": MarkitdownParser,
    "xlsx": MarkitdownParser,
    "xls": MarkitdownParser,
    "csv": MarkitdownParser,
}, description="MarkItDown 解析引擎（微软 MarkItDown 库）")

registry.register("opendataloader", {"pdf": OpenDataLoaderParser},
                 description="OpenDataLoader PDF（版面分析，需 Java 11+）",
                 check_available=lambda overrides: opendataloader_available(overrides, quick=True),
                 unavailable_hint="请安装 opendataloader-pdf 与 Java 11+")

registry.register("external_pdf", {"pdf": ExternalPdfParser},
                 description="外部 PDF 解析服务（MinerU / PaddleOCR-VL）",
                 check_available=lambda overrides: (
                     (True, "") if bool(ExternalPdfParser.resolve_url(overrides))
                     else (False, "请配置 AION_PDF_EXTERNAL_URL")
                 ),
                 unavailable_hint="请配置 pdf_external_url")


# ── 公共导出 ───────────────────────────────────────────────────────────

__all__ = [
    "BaseParser", "ParsedDocument",
    "ParserEngineRegistry", "registry",
    "FirstParser", "PipelineParser",
    "parser_worker_limit",
    "PDFParser", "PDFScannedParser",
    "Docx2Parser", "DocParser", "ExcelParser", "DocxParser",
    "MarkdownParser",
    "EPUBParser",
    "MHTMLParser",
    "ImageParser",
    "WebParser",
    "MarkitdownParser",
    "OpenDataLoaderParser",
    "ExternalPdfParser",
    "Parser",
    "extract_pptx_media_rasterized", "attach_pptx_media_to_markdown",
    "list_pptx_media", "rasterize_media_bytes",
    "repair_xlsx_bytes", "fill_merged_cells_xlsx",
    "normalize_ppt_bytes", "convert_ppt_to_pptx_bytes",
]


class Parser:
    """文档解析门面类。

    统一入口，支持文件解析和 URL 解析。
    内部通过 registry 路由到对应格式的解析器。
    """

    def __init__(self) -> None:
        self.registry = registry

    def parse_file(
        self,
        file_name: str,
        file_type: str,
        content: bytes,
        parser_engine: str | None = None,
        engine_overrides: dict[str, Any] | None = None,
    ) -> ParsedDocument:
        engine = parser_engine or ""
        overrides = engine_overrides or {}
        cls = self.registry.get_parser_class(engine, file_type)
        parser = cls(file_name=file_name, file_type=file_type, **overrides)
        return parser.parse(content)

    def parse_url(
        self,
        url: str,
        title: str = "",
        parser_engine: str | None = None,
        engine_overrides: dict[str, Any] | None = None,
    ) -> ParsedDocument:
        parser = WebParser(title=title)
        return parser.parse(url.encode())
