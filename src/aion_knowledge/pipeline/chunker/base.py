"""Chunker 核心数据模型 —— 分块配置与结果定义。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChunkConfig(BaseModel):
    """分块配置。

    控制分块的全套参数，包括目标大小、重叠量、策略选择、语言提示和分隔符。
    所有分块器（HeadingSplitter / MarkerSplitter / RecursiveSplitter）共享此配置。

    策略 ``strategy`` 取值说明：

    - ``"auto"``（默认）：由 ``Chunker._select_splitters`` 基于文档画像自动选择
      最合适的策略链，详见 ``document_analysis.py``。
    - ``"heading"``：优先按 Markdown 标题层级切分，超长章节下放递归分块。
    - ``"heuristic"``：基于结构标记（编号、全大写标题、分页符等）切分，见
      ``marker_splitter.py``。
    - ``"recursive"``：纯递归分块，按 ``separators`` 优先级逐步降级拆分。
    - ``"no_split"``：整篇不切，作为一个 chunk 返回。
    """

    chunk_size: int = Field(default=512, ge=1, description="目标 chunk 大小（token 数）")
    chunk_overlap: int = Field(default=80, ge=0, description="chunk 间重叠（token 数）")
    strategy: str = Field(
        default="auto",
        pattern=r"^(auto|heading|heuristic|recursive|no_split)$",
        description=(
            "分块策略：auto=自动选择, heading=标题, "
            "heuristic=启发式, recursive=递归, no_split=不切"
        ),
    )
    languages: list[str] = Field(default=[], description="语言提示，如 ['zh'], ['en']")
    separators: list[str] = Field(
        default=["\n\n", "\n", "。", "；", "，", ""],
        description="分隔符优先级列表（递归分块用）",
    )


class ChunkResult(BaseModel):
    """单个分块结果。

    包含块内容、精确 token 数、标题面包屑路径（供检索时定位）和扩展元数据。

    语义约定：

    - ``heading_path``：如 ``"概论 > 方法 > 实验设计"``，用于检索时提供上下文
      路径前缀（与 RAPTOR 中的 summary 路径不同，此为原始标题路径）。
    - ``seq_num``：文档内分块序号，保证有序检索。
    """

    content: str = Field(description="分块文本内容")
    token_count: int = Field(ge=0, description="精确 token 数")
    heading_path: str | None = Field(default=None, description="标题面包屑路径")
    chunk_id: str = Field(description="块标识")
    seq_num: int = Field(default=0, ge=0, description="在文档中的序号")
    metadata: dict[str, Any] = Field(default_factory=dict, description="扩展元数据")
