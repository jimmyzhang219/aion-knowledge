import base64
import logging

from aion_knowledge.pipeline.parser.base import BaseParser, ParsedDocument

logger = logging.getLogger(__name__)


class ImageParser(BaseParser):
    """独立图片文件解析器。

    将图片以 Markdown 引用形式返回，原始图片数据存放在 ParsedDocument.images 中，
    原始图片数据由 ``executor.run_parser`` 中的 ``_upload_images`` 上传存储。
    """

    def parse_into_text(self, content: bytes) -> ParsedDocument:
        logger.info("Parsing image file=%s, size=%d bytes", self.file_name, len(content))

        ref_path = f"images/{self.file_name}"

        text = f"![{self.file_name}]({ref_path})"
        images = {ref_path: base64.b64encode(content).decode()}

        return ParsedDocument(content=text, images=images)
