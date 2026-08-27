"""外部 PDF 解析服务 —— 委托给 MinerU / PaddleOCR-VL 等 HTTP API。

外部服务将 PDF 转为 Markdown（含图片），通过 ``merge_tables`` 等参数
启用跨页表格合并等高级功能。当服务不可用或返回空结果时回退到内置扫描渲染。
"""

import logging
from typing import Any, Dict, Mapping, Optional

from aion_knowledge.common.config import settings
from aion_knowledge.pipeline.parser.base import BaseParser, ParsedDocument

logger = logging.getLogger(__name__)

_MIN_CHARS_PER_PAGE = 10


class ExternalPdfParser(BaseParser):
    """委托给外部 HTTP 服务进行 PDF → Markdown 转换。

    服务契约（POST multipart/form-data）:
        - 请求: ``file`` 字段携带 PDF 字节流
        - 请求: ``merge_tables=true`` 等查询参数
        - 响应: ``{"content": "markdown 文本", "images": {"ref": "base64", ...}}``

    支持 MinerU 和 PaddleOCR-VL 等兼容此接口的服务。
    当 ``pdf_external_url`` 未配置时引发 RuntimeError。
    """

    @staticmethod
    def resolve_url(overrides: Optional[Mapping[str, Any]] = None) -> str:
        """解析外部 PDF 解析器的 URL，优先使用每次上传的覆盖配置。"""
        if overrides:
            v = overrides.get("pdf_external_url")
            if v and str(v).strip():
                return str(v).strip()
        return settings.pdf_external_url

    def __init__(self, *args: Any, **kwargs: Any):
        self._engine_overrides: Dict[str, Any] = {
            k: v for k, v in kwargs.items() if k.startswith("pdf_external_")
        }
        super().__init__(*args, **kwargs)

    def parse_into_text(self, content: bytes) -> ParsedDocument:
        import httpx

        url = self.resolve_url(self._engine_overrides)
        if not url:
            raise RuntimeError(
                "外部 PDF 解析器未配置 URL。"
                "请设置 AION_PDF_EXTERNAL_URL 环境变量或配置 pdf_external_url。"
            )

        api_key = self._get_override("pdf_external_api_key", settings.pdf_external_api_key)
        merge_tables = self._get_override(
            "pdf_external_merge_tables", settings.pdf_external_merge_tables
        )
        timeout = self._get_override("pdf_external_timeout", settings.pdf_external_timeout)

        files = {
            "file": (self.file_name or "document.pdf", content, "application/pdf"),
        }
        params: Dict[str, str] = {}
        if merge_tables:
            params["merge_tables"] = "true"

        headers: Dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        logger.info(
            "ExternalPdfParser: POST %s (merge_tables=%s, file=%s)",
            url, merge_tables, self.file_name,
        )

        try:
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                resp = client.post(url, files=files, params=params, headers=headers)
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            logger.warning("ExternalPdfParser: 请求失败，回退到内置扫描: %s", e)
            return self._fallback_to_scanned(content)

        text = (data.get("content") or "").strip()
        images: Dict[str, str] = data.get("images") or {}

        if len(text) < _MIN_CHARS_PER_PAGE and not images:
            logger.warning(
                "ExternalPdfParser: %s 返回内容过少，回退到内置扫描",
                self.file_name,
            )
            return self._fallback_to_scanned(content)

        logger.info(
            "ExternalPdfParser: %s -> content_len=%d images=%d",
            self.file_name, len(text), len(images),
        )
        return ParsedDocument(
            content=text,
            images=images,
            metadata={
                "parser_engine": "external_pdf",
                "pdf_external_url": url,
                "merge_tables": merge_tables,
            },
        )

    def _get_override(self, key: str, default: Any) -> Any:
        """从每次上传的覆盖配置或全局设置中取值。"""
        v = self._engine_overrides.get(key)
        return v if v is not None else default

    def _fallback_to_scanned(self, content: bytes) -> ParsedDocument:
        from aion_knowledge.pipeline.parser.pdf import PDFScannedParser

        logger.info("ExternalPdfParser: 回退到 PDFScannedParser for %s", self.file_name)
        return PDFScannedParser(
            file_name=self.file_name, file_type=self.file_type
        ).parse_into_text(content)
