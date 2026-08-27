"""当 MarkItDown 无法内联时，提取并栅格化 PPTX 中嵌入的图片（如 WMF）。"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
import subprocess
import tempfile
import zipfile
from typing import Dict, List, Tuple

from aion_knowledge.common.uuid7 import uuid7

logger = logging.getLogger(__name__)

_MARKDOWN_IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_RASTER_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
_VECTOR_EXT = {".wmf", ".emf", ".svg"}


def _find_convert() -> str | None:
    for path in ("/usr/bin/convert", "/usr/local/bin/convert"):
        if os.path.isfile(path):
            return path
    try:
        result = subprocess.run(
            ["which", "convert"], capture_output=True, text=True, check=False
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except OSError:
        pass
    return None


def _rasterize_with_imagemagick(data: bytes, suffix: str) -> bytes | None:
    convert = _find_convert()
    if not convert:
        return None
    with tempfile.TemporaryDirectory() as temp_dir:
        src = os.path.join(temp_dir, f"input{suffix}")
        dst = os.path.join(temp_dir, "output.png")
        with open(src, "wb") as handle:
            handle.write(data)
        try:
            result = subprocess.run(
                [convert, src, dst],
                capture_output=True,
                timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning("ImageMagick convert failed: %s", exc)
            return None
        if result.returncode != 0 or not os.path.isfile(dst):
            stderr = (result.stderr or b"").decode("utf-8", errors="ignore")
            logger.warning("ImageMagick convert exit %s: %s", result.returncode, stderr)
            return None
        with open(dst, "rb") as handle:
            return handle.read()


def _rasterize_with_pillow(data: bytes) -> bytes | None:
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        # open() 返回 ImageFile（Image 子类），convert() 返回 Image，统一注解为基类类型
        img: Image.Image = Image.open(io.BytesIO(data))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        out = io.BytesIO()
        img.save(out, format="PNG")
        return out.getvalue()
    except Exception as exc:
        logger.debug("Pillow could not open media bytes: %s", exc)
        return None


def rasterize_media_bytes(name: str, data: bytes) -> bytes | None:
    ext = os.path.splitext(name)[1].lower()
    if ext in _RASTER_EXT:
        png = _rasterize_with_pillow(data)
        if png:
            return png
    if ext in _VECTOR_EXT or ext in _RASTER_EXT:
        return _rasterize_with_imagemagick(data, ext or ".bin")
    return _rasterize_with_imagemagick(data, ext or ".bin")


def list_pptx_media(pptx_bytes: bytes) -> List[Tuple[str, bytes]]:
    """返回 ppt/media/ 下每个文件的（ZIP 路径，原始字节），按归档顺序排列。"""
    items: List[Tuple[str, bytes]] = []
    with zipfile.ZipFile(io.BytesIO(pptx_bytes)) as archive:
        for name in archive.namelist():
            if not name.startswith("ppt/media/"):
                continue
            base = os.path.basename(name)
            if not base or base.startswith("."):
                continue
            items.append((name, archive.read(name)))
    return items


def extract_pptx_media_rasterized(pptx_bytes: bytes) -> List[bytes]:
    """将 ppt/media 下的所有资源栅格化为 PNG 字节，跳过失败项。"""
    rasterized: List[bytes] = []
    for path, raw in list_pptx_media(pptx_bytes):
        png = rasterize_media_bytes(os.path.basename(path), raw)
        if png:
            rasterized.append(png)
            logger.info("Rasterized pptx media %s (%d -> %d bytes)", path, len(raw), len(png))
        else:
            logger.warning("Failed to rasterize pptx media %s", path)
    return rasterized


def _is_unresolved_image_ref(url: str) -> bool:
    if not url or url.startswith("data:") or url.startswith("images/"):
        return False
    if url.startswith(("http://", "https://")):
        return False
    return True


def attach_pptx_media_to_markdown(
    markdown: str, pptx_bytes: bytes
) -> Tuple[str, Dict[str, str]]:
    """将未解析的 ![](...) 引用替换为 images/ 路径和内联图片数据。"""
    media = extract_pptx_media_rasterized(pptx_bytes)
    if not media:
        return markdown, {}

    images: Dict[str, str] = {}
    media_iter = iter(media)

    def repl(match: re.Match[str]) -> str:
        alt, url = match.group(1), match.group(2)
        if not _is_unresolved_image_ref(url):
            return match.group(0)
        try:
            png = next(media_iter)
        except StopIteration:
            return match.group(0)
        ref = f"images/{uuid7()}.png"
        images[ref] = base64.b64encode(png).decode()
        return f"![{alt}]({ref})"

    return _MARKDOWN_IMAGE.sub(repl, markdown), images


def markdown_needs_pptx_media_attach(markdown: str) -> bool:
    return any(_is_unresolved_image_ref(m.group(2)) for m in _MARKDOWN_IMAGE.finditer(markdown))
