"""文档解析基础：BaseParser ABC + ParsedDocument 输出模型。"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ParsedDocument(BaseModel):
    """解析器输出模型：markdown 文本 + base64 图片映射 + 元数据。"""
    content: str = Field(default="", description="markdown 格式的文档文本")
    images: dict[str, str] = Field(
        default_factory=dict,
        description="图片引用路径 → base64 编码数据",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="解析元数据（页数、引擎等）",
    )

    def is_valid(self) -> bool:
        return bool(self.content)


class BaseParser(ABC):
    """解析器基类。子类只需实现 parse_into_text()。

    约定：
    - parse_into_text(content: bytes) -> ParsedDocument
    - parse(content) 调用 parse_into_text 并加日志
    """

    def __init__(
        self,
        file_name: str = "",
        file_type: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.file_name = file_name
        self.file_type = file_type or os.path.splitext(file_name)[1].lstrip(".")
        logger.info("Initializing %s for file=%s, type=%s",
                     self.__class__.__name__, file_name, self.file_type)

    @abstractmethod
    def parse_into_text(self, content: bytes) -> ParsedDocument:
        """解析文档内容为 markdown 文本 + 图片引用。"""

    def parse(self, content: bytes) -> ParsedDocument:
        logger.info("Parsing document with %s, bytes: %d",
                     self.__class__.__name__, len(content))
        doc = self.parse_into_text(content)
        logger.info("Extracted %d characters from %s",
                     len(doc.content), self.file_name)
        return doc
