"""后处理模块基类与上下文。

定义 PostProcModule 抽象基类和 PostProcContext 数据模型。
所有后处理模块（text、vector、summarizer 等）均继承 PostProcModule，
通过 always_on / depends_on 声明启用策略与 DAG 依赖关系。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PostProcContext(BaseModel):
    """后处理模块的运行时上下文。"""
    document_id: str
    kb_id: str
    doc_name: str
    suffix: str = ""
    parser_id: str = ""
    parser_config: dict[str, Any] = Field(default_factory=dict)


class PostProcModule(ABC):
    """后处理模块基类。

    所有处理方式（含内置 text/vector）均继承此类。

    DB 连接纪律：禁止在 session 上下文内做 LLM/embedding/HTTP 等慢操作。
    先完成数据准备（无会话），再获取会话批量写入（add_all + 一次 flush），
    避免连接在慢操作期间被远端回收（connection was closed in the middle
    of operation）。违规案例见 wiki/community 两阶段化重构。
    """

    always_on: ClassVar[bool] = False
    depends_on: ClassVar[list[str]] = []

    @abstractmethod
    async def process(
        self,
        ctx: PostProcContext,
        chunks: list[dict[str, Any]],
    ) -> int:
        """处理 chunks，写入自身表，返回写入记录数。"""
        ...

    def _check_content(
        self,
        content: str | None,
        min_chars: int = 10,
        strip_markdown: bool = True,
    ) -> bool:
        """检查内容是否含有足量有效文本，避免空文档/扫描件导致 LLM 幻觉。

        各模块在调用 LLM 前主动调用此守卫，内容不足时跳过处理::

            if not self._check_content(content):
                continue

        Args:
            content: 待检查的文本内容。
            min_chars: 最小有效字符数（剥离 Markdown 并去除空白后），默认 10。
            strip_markdown: 是否先剥离 Markdown 语法再检查，默认 True。

        Returns:
            True 表示内容充足可以处理，False 表示应跳过。
        """
        if not content or not content.strip():
            logger.warning("内容为空或纯空白，跳过（min_chars=%d）", min_chars)
            return False

        if strip_markdown:
            from aion_knowledge.pipeline.cleaner.cleaner import clean_passage
            cleaned = clean_passage(content)
        else:
            cleaned = content.strip()

        if len(cleaned) < min_chars:
            logger.warning(
                "内容过少：有效字符 %d < %d，跳过",
                len(cleaned), min_chars,
            )
            return False
        return True
