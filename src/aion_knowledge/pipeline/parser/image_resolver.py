"""通用 Markdown 图片解析器：支持 data URI、远程 HTTP(S)、本地路径、HTML <img> 标签。

适用于任何文档类型解析后产生的 Markdown 内容，扫描其中所有
![]() 引用和 <img> 标签，尝试获取图片数据，替换为 images/uuid.png 引用。

容错：获取失败时记录警告，保留原始引用，不阻断管线。
"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from PIL import Image

from aion_knowledge.common.uuid7 import uuid7
from aion_knowledge.infrastructure.security import (
    SSRFSafeHTTPClient,
    validate_url_for_ssrf,
)

logger = logging.getLogger(__name__)

_IMAGE_PATTERN = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg", ".tiff", ".ico"})
_REQUEST_TIMEOUT = 30.0

_MIN_IMAGE_DIMENSION = 64
_MIN_IMAGE_BYTES = 512

# HTML img patterns
_IMG_HTML_DATA_URI = re.compile(
    r'(?i)<img\s[^>]*?src\s*=\s*["\'](data:image/[^;]+;base64,[^"\']+)["\'][^>]*?/?\s*>'
)
_IMG_HTML_RELATIVE_SRC = re.compile(
    r'(?i)<img\b([^>]*?)\bsrc\s*=\s*[\'"]([^\'"]+)[\'"]([^>]*)>'
)

# Linked image unwrap: [![alt](img_url)](link_url)
_RE_LINKED_IMAGE = re.compile(
    r'\[!\[([^\]]*)\]\(([^()\s]*(?:\([^)]*\)[^()\s]*)*)\)\]'
    r'\([^()\s]*(?:\([^)]*\)[^()\s]*)*\)'
)


def is_icon_image(data: bytes) -> bool:
    """Return True if the image data looks like a small icon or decorative element."""
    try:
        img = Image.open(io.BytesIO(data))
        w, h = img.size
        return w < _MIN_IMAGE_DIMENSION and h < _MIN_IMAGE_DIMENSION
    except Exception:
        return len(data) < _MIN_IMAGE_BYTES


@dataclass
class StoredImage:
    original_ref: str
    serving_url: str
    mime_type: str


class ImageResolver:
    """Markdown 图片解析器，支持 data URI、远程、本地、HTML <img> 图片的解析与替换。

    使用例：
        resolver = ImageResolver()
        updated, images = resolver.resolve_images(content, existing_images)
    """

    def __init__(
        self,
        max_remote: int = 30,
        max_data_uri: int = 30,
        max_single_size: int = 10 * 1024 * 1024,
    ):
        self._max_remote = max_remote
        self._max_data_uri = max_data_uri
        self._max_single_size = max_single_size

    def resolve_images(
        self,
        content: str,
        existing_images: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, Dict[str, str]]:
        """解析 Markdown 内容中的图片引用，返回 (更新后内容, 图片映射)。"""
        images: Dict[str, str] = dict(existing_images) if existing_images else {}
        self._saved_refs: Dict[str, str] = {}
        self._remote_count = 0
        self._data_uri_count = 0

        content = self._unwrap_linked_images(content)
        content = self._resolve_data_uri_images(content, images)
        content = self._resolve_remote_images(content, images)
        content = self._resolve_html_images(content, images)
        content = self._resolve_local_images(content, images)

        return content, images

    # ── helpers ────────────────────────────────────────────────

    @staticmethod
    def _ext_from_mime(mime_type: str) -> Optional[str]:
        mapping = {
            "image/png": ".png", "image/jpeg": ".jpg", "image/gif": ".gif",
            "image/webp": ".webp", "image/bmp": ".bmp", "image/svg+xml": ".svg",
        }
        return mapping.get(mime_type)

    @staticmethod
    def _sniff_mime(data: bytes) -> Optional[str]:
        if len(data) < 4:
            return None
        if data[:4] == b'\x89PNG':
            return "image/png"
        if data[:2] in (b'\xFF\xD8',):
            return "image/jpeg"
        if data[:3] == b'GIF':
            return "image/gif"
        if data[:4] == b'RIFF' and len(data) >= 12 and data[8:12] == b'WEBP':
            return "image/webp"
        if data[:2] == b'BM':
            return "image/bmp"
        return None

    def _save_image(
        self,
        data: bytes,
        original_ref: str,
        mime_type: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> Optional[str]:
        """保存图片到 images 字典，返回 ref_key。图标或失败时返回 None。"""
        if is_icon_image(data):
            return None

        # 文件名去重：相同 filename 复用已有 ref_key
        if filename:
            existing = self._saved_refs.get(f"__filename__:{filename}")
            if existing:
                self._saved_refs[original_ref] = existing
                return existing

        if not mime_type:
            mime_type = self._sniff_mime(data) or "image/png"
        ext = self._ext_from_mime(mime_type) or ".png"
        ref_key = f"images/{uuid7().hex}{ext}"

        self._saved_refs[original_ref] = ref_key
        if filename:
            self._saved_refs[f"__filename__:{filename}"] = ref_key

        return ref_key

    def _is_already_resolved(self, ref: str) -> Optional[str]:
        """Return existing ref_key if this ref was already processed, else None."""
        return self._saved_refs.get(ref)

    # ── unwrap linked images ───────────────────────────────────

    @staticmethod
    def _unwrap_linked_images(content: str) -> str:
        return _RE_LINKED_IMAGE.sub(r'![(\1)](\2)', content)

    # ── data URI images ────────────────────────────────────────

    def _resolve_data_uri_images(self, content: str, images: Dict[str, str]) -> str:
        def repl(match: re.Match[str]) -> str:
            if self._data_uri_count >= self._max_data_uri:
                return match.group(0)

            alt = match.group(1)
            url = match.group(2).strip()

            if not url.startswith("data:"):
                return match.group(0)

            existing = self._is_already_resolved(url)
            if existing:
                return f"![{alt}]({existing})"

            try:
                header, _, b64_data = url.partition(",")
                if not b64_data:
                    return match.group(0)
                is_base64 = ";base64" in header
                mediatype = header[5:header.find(";")] if ";" in header else header[5:]
                img_bytes = base64.b64decode(b64_data) if is_base64 else b64_data.encode("latin-1")

                if len(img_bytes) > self._max_single_size:
                    logger.warning("data URI 图片超限: %d bytes", len(img_bytes))
                    return match.group(0)

                mime_map = {"image/jpeg": ".jpg", "image/png": ".png", "image/gif": ".gif",
                            "image/webp": ".webp", "image/bmp": ".bmp", "image/svg+xml": ".svg"}
                ref_key = f"images/{uuid7().hex}{mime_map.get(mediatype, '.png')}"

                if is_icon_image(img_bytes):
                    logger.debug("过滤图标 data URI 图片: %s", url[:50])
                    return ""

                images[ref_key] = b64_data if is_base64 else base64.b64encode(img_bytes).decode()
                self._saved_refs[url] = ref_key
                self._data_uri_count += 1
                logger.info("data URI 图片解析成功: %s → %s", url[:50], ref_key)
                return f"![{alt}]({ref_key})"
            except Exception as exc:
                logger.warning("data URI 解析失败: %s", exc)
                return match.group(0)

        content = _IMAGE_PATTERN.sub(repl, content)
        return content

    # ── remote images ──────────────────────────────────────────

    def _resolve_remote_images(self, content: str, images: Dict[str, str]) -> str:
        """解析远程 HTTP(S) 图片，含 SSRF 校验、图标过滤、数量限制。"""
        content = self._unwrap_linked_images(content)

        def repl(match: re.Match[str]) -> str:
            if self._remote_count >= self._max_remote:
                return match.group(0)

            alt = match.group(1)
            url = match.group(2).strip()

            if not url.startswith(("http://", "https://")):
                return match.group(0)

            existing = self._is_already_resolved(url)
            if existing:
                return f"![{alt}]({existing})"

            # SSRF 校验
            try:
                validate_url_for_ssrf(url)
            except Exception as e:
                logger.warning("远程图片被 SSRF 拦截: %s (%s)", url, e)
                return match.group(0)

            # 下载
            try:
                client = SSRFSafeHTTPClient(timeout=_REQUEST_TIMEOUT)
                resp = client.get(url)
                if resp.status_code != 200:
                    logger.warning("远程图片下载失败, HTTP %d: %s", resp.status_code, url)
                    return match.group(0)

                data = resp.content
                if len(data) > self._max_single_size:
                    logger.warning("远程图片超限: %s (%d bytes)", url, len(data))
                    return match.group(0)

                ct = resp.headers.get("content-type", "")
                mime_type = ct.split(";")[0].strip() if ct else None
                if mime_type and not mime_type.startswith(("image/", "application/octet-stream")):
                    ext = os.path.splitext(url.split("?")[0])[1].lower()
                    if ext not in _IMAGE_EXTENSIONS:
                        logger.warning("非图片资源: %s (Content-Type: %s)", url, mime_type)
                        return match.group(0)

                ref_key = self._save_image(data, url, mime_type=mime_type)
                if ref_key is None:
                    logger.debug("远程图片被过滤（图标）: %s", url)
                    return ""
                images[ref_key] = base64.b64encode(data).decode()
                self._remote_count += 1
                logger.info("远程图片解析成功: %s → %s", url, ref_key)
                return f"![{alt}]({ref_key})"
            except Exception as exc:
                logger.warning("远程图片处理失败: %s — %s", url, exc)
                return match.group(0)

        content = _IMAGE_PATTERN.sub(repl, content)
        return content

    # ── HTML images ────────────────────────────────────────────

    def _resolve_html_images(self, content: str, images: Dict[str, str]) -> str:
        """解析 HTML <img> 标签中的图片引用。

        当前处理 data:URI 和本地路径引用，实际 SSRF 集成在后续任务中实现。
        """

        def replace_data_uri(match: re.Match[str]) -> str:
            data_uri = match.group(1)
            alt = "image"
            try:
                header, _, b64_data = data_uri.partition(",")
                if not b64_data:
                    return match.group(0)
                is_base64 = ";base64" in header
                mediatype = header[5:header.find(";")] if ";" in header else header[5:]
                img_bytes = base64.b64decode(b64_data) if is_base64 else b64_data.encode("latin-1")

                if len(img_bytes) > self._max_single_size:
                    return match.group(0)

                if is_icon_image(img_bytes):
                    return ""

                mime_map = {"image/jpeg": ".jpg", "image/png": ".png", "image/gif": ".gif",
                            "image/webp": ".webp", "image/bmp": ".bmp", "image/svg+xml": ".svg"}
                ref_key = f"images/{uuid7().hex}{mime_map.get(mediatype, '.png')}"
                images[ref_key] = b64_data if is_base64 else base64.b64encode(img_bytes).decode()
                self._saved_refs[data_uri] = ref_key
                self._data_uri_count += 1
                return f"![{alt}]({ref_key})"
            except Exception as exc:
                logger.warning("HTML data:URI 解析失败: %s", exc)
                return match.group(0)

        content = _IMG_HTML_DATA_URI.sub(replace_data_uri, content)

        def replace_relative(match: re.Match[str]) -> str:
            src = match.group(2).strip()

            if not src or src.startswith(("http://", "https://", "data:")):
                return match.group(0)

            existing = self._is_already_resolved(src)
            if existing:
                return f"![image]({existing})"

            try:
                if not os.path.isfile(src):
                    return match.group(0)
                with open(src, "rb") as f:
                    data = f.read()
            except Exception:
                return match.group(0)

            ref_key = self._save_image(data, src, filename=os.path.basename(src))
            if ref_key is None:
                return ""
            images[ref_key] = base64.b64encode(data).decode()
            return f"![image]({ref_key})"

        content = _IMG_HTML_RELATIVE_SRC.sub(replace_relative, content)
        return content

    # ── local images ───────────────────────────────────────────

    def _resolve_local_images(self, content: str, images: Dict[str, str]) -> str:
        def repl(match: re.Match[str]) -> str:
            alt = match.group(1)
            url = match.group(2).strip()

            if not url or url.startswith(("http://", "https://", "data:", "images/")):
                return match.group(0)

            existing = self._is_already_resolved(url)
            if existing:
                return f"![{alt}]({existing})"

            try:
                if not os.path.isfile(url):
                    logger.debug("本地图片不存在: %s", url)
                    return match.group(0)
                with open(url, "rb") as f:
                    data = f.read()
            except Exception as exc:
                logger.debug("读取本地图片失败: %s — %s", url, exc)
                return match.group(0)

            ref_key = self._save_image(data, url, filename=os.path.basename(url))
            if ref_key is None:
                return ""
            images[ref_key] = base64.b64encode(data).decode()
            logger.info("本地图片解析成功: %s → %s", url, ref_key)
            return f"![{alt}]({ref_key})"

        content = _IMAGE_PATTERN.sub(repl, content)
        return content


# ── 向后兼容入口 ───────────────────────────────────────────
def resolve_images(
    content: str,
    existing_images: Optional[Dict[str, str]] = None,
) -> Tuple[str, Dict[str, str]]:
    """解析 Markdown 图片引用（向后兼容函数）。"""
    return ImageResolver().resolve_images(content, existing_images)
