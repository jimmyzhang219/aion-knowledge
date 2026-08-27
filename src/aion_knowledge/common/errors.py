"""统一的错误层次结构。"""

from __future__ import annotations

from typing import Any


class AionError(Exception):
    """所有 aion-knowledge 异常的基类。"""

    def __init__(self, message: str = "", details: dict[str, Any] | None = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | {self.details}"
        return self.message


class ConfigurationError(AionError):
    """无效或缺失的配置。"""


class StorageError(AionError):
    """数据库或存储操作失败。"""


class DocumentError(AionError):
    """文档处理错误。"""


class IngestionError(AionError):
    """数据摄入管道错误。"""


class RetrievalError(AionError):
    """检索管道错误。"""


class ValidationError(AionError):
    """输入校验失败。"""


class NotFoundError(AionError):
    """请求的资源未找到。"""


class UnsupportedFormatError(DocumentError):
    """不支持的文档格式。"""
