"""检索管线上下文，贯穿所有 Stage。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aion_knowledge.retrieval.base import ChunkResult


@dataclass
class PipelineContext:
    """管线上下文：输入参数 + 各 Stage 之间的数据传递。"""
    query: str  # 当前查询；RewriteStage 启用时会被改写覆盖
    kb_id: str  # 知识库 ID
    original_query: str | None = None  # 改写前的原始查询（RewriteStage 填充），供依赖自然语言原文的检索器（如 FAQ）使用
    top_k: int = 10  # 最终返回条数上限（TopKTruncation 截断目标，也作为各检索器的召回数量）
    path_top_k: int = 20  # 单路召回条数（MultiRecallStage 传给各检索器）
    query_embedding: list[float] | None = None  # 查询向量，由上游计算后传入（向量/图/FAQ 等检索器使用）
    expansion_keywords: list[str] | None = None  # 查询扩展关键词（RewriteStage 填充，BM25/关键词/Wiki 检索器拼接使用）
    entities: list[str] | None = None  # 抽取的命名实体（RewriteStage 填充，GraphRetriever 使用）
    enabled_paths: set[str] | None = None  # 启用的召回路径名集合；None 表示全部启用
    # 以下由各 Stage 填充
    results: list[ChunkResult] = field(default_factory=list)  # RRF 融合后的候选结果（RRFFusion 写入，Reranker/MMR/TopK 依次消费）
    path_results: dict[str, list[ChunkResult]] = field(default_factory=dict)  # 各路径召回结果（MultiRecallStage 写入，key 为检索路径名）
    # 透传给 RetrieverContext（供 KG 等检索器使用）
    _llm: Any = field(default=None, repr=False, compare=False)  # 透传给 RetrieverContext 的 LLM client
    _embedder: Any = field(default=None, repr=False, compare=False)  # 透传给 RetrieverContext 的 embedder
