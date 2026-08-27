"""编解码工具函数。"""
import base64
import binascii
import io
import logging
from typing import List

from PIL import Image

logger = logging.getLogger(__name__)


def decode_image(image: Image.Image, quality: int = 85) -> str:
    """将 PIL Image 转换为 base64 编码的 JPEG 字符串。"""
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def encode_bytes(text: str) -> bytes:
    """将字符串编码为字节。"""
    return text.encode("utf-8") if isinstance(text, str) else text


def encode_image(image: str, errors: str = "strict") -> bytes:
    """解码 base64 编码的图片字符串为字节。"""
    try:
        return base64.b64decode(image)
    except binascii.Error as e:
        if errors == "ignore":
            return b""
        raise e


def decode_bytes(
    content: bytes,
    encodings: List[str] | None = None,
) -> str:
    """自动检测编码并将字节解码为字符串。"""
    if encodings is None:
        encodings = [
            "utf-8", "gb18030", "gb2312", "gbk",
            "big5", "ascii", "latin-1",
        ]
    for encoding in encodings:
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    text = content.decode(encoding="latin-1", errors="replace")
    logger.warning(
        "Unable to determine correct encoding, using latin-1 as fallback. "
        "This may cause character issues."
    )
    return text
