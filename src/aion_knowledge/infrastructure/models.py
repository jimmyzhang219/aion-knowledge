"""跨层共享数据模型：UnifiedContext、PostProcConfig、PostProcTask。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from aion_knowledge.common.trace import get_trace_id
from aion_knowledge.common.uuid7 import uuid7
from aion_knowledge.models.enums import StrategyName

CONTEXT_SOURCES = frozenset(s.value for s in StrategyName)


class UnifiedContext(BaseModel):
    """多源数据接入的统一消息上下文。

    所有数据源在完成内容落地（S3 对象存储）后，统一封装此结构推入 Queue1。
    content 字段携带原始数据，供后续处理直接使用（无需回读 S3）。
    """
    context_id: str = Field(default_factory=lambda: str(uuid7()))  # 消息唯一标识
    source: str           # 数据来源（regular / url_import / faq / connector_*）
    kb_id: str            # 目标知识库 ID
    doc_name: str         # 文档名称（含扩展名）
    suffix: str           # 文档类型扩展名（pdf / md / csv 等）
    original_file_ref: str   # 原始文件在对象存储上的完整引用路径（s3://bucket/... 或本地路径）
    content: bytes = b""  # 新增：原始数据内容
    chunk_strategy: str = "auto"    # 切片策略（auto / no_split）
    ext_metadata: dict[str, Any] = Field(default_factory=dict)  # 扩展元数据（document_id / task_id 等）
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())  # 消息创建时间（ISO 格式）
    trace_id: str = Field(default_factory=get_trace_id)  # 请求链路追踪 ID（未显式传入时取请求上下文，无则生成）

    # 管线执行过程中填充：S3 文件目录 + 解析后图片/Markdown 在对象存储上的完整引用
    file_dir: str = ""    # S3 文件目录（key 前缀，不含 s3_prefix，如 kb-123/hash）
    image_ref_map: dict[str, str] = Field(default_factory=dict)  # {本地图片路径 → 完整对象存储引用}
    md_file_ref: str = "" # 转换后 Markdown 的完整对象存储引用


class PostProcConfig(BaseModel):
    """后处理子任务启用配置（出厂硬控层）。

    enable_* = False：该后处理出厂禁用，即使 .env 中 AION_POSTPROC_* 开启也无效。
    enable_* = True：出厂启用，最终由 .env 的 settings.postproc_* 控制。
    最终生效 = enable_* AND settings.postproc_*（AND 在 postproc_worker 调度处计算）。
    """
    enable_keyword_extract: bool = True
    enable_question_gen: bool = True
    enable_summarizer: bool = True
    enable_raptor: bool = True
    enable_graph_extract: bool = True
    enable_community: bool = True
    enable_disambiguation: bool = True
    enable_wiki: bool = True


class PostProcTask(BaseModel):
    """Queue2 消息结构。Worker1 双写完成后封装推送。"""
    document_id: str
    kb_id: str
    doc_name: str
    chunk_count: int
    postproc_config: PostProcConfig
    modules: list[str] | None = None  # 重跑白名单；None = 全部已启用模块
    suffix: str = ""
    parser_id: str = ""
    parser_config: dict[str, Any] = Field(default_factory=dict)
    trace_id: str = Field(default_factory=get_trace_id)  # 请求链路追踪 ID（未显式传入时取请求上下文，无则生成）
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
