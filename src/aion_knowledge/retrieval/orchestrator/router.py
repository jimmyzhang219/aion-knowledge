"""检索入口 — 集成 MultiRetriever + RRFFuser。

路由流程：
  1. 初始化时根据配置注册可用的检索器
  2. search() 接收查询 → 并发检索 → RRF 融合 → 返回结果
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from aion_knowledge.common.config import settings
from aion_knowledge.retrieval.base import BaseRetriever, ChunkResult
from aion_knowledge.retrieval.orchestrator.multi_retriever import MultiRetriever
from aion_knowledge.retrieval.orchestrator.rrf_fuser import RRFFuser
from aion_knowledge.retrieval.pipeline import PipelineContext

# 检索器导入
from aion_knowledge.retrieval.search.bm25_retriever import BM25Retriever
from aion_knowledge.retrieval.search.community_retriever import CommunityRetriever
from aion_knowledge.retrieval.search.faq_retriever import FAQRetriever
from aion_knowledge.retrieval.search.graph_retriever import GraphRetriever
from aion_knowledge.retrieval.search.keyword_retriever import KeywordRetriever
from aion_knowledge.retrieval.search.question_gen_retriever import QuestionGenRetriever
from aion_knowledge.retrieval.search.raptor_retriever import RAPTORRetriever
from aion_knowledge.retrieval.search.summary_retriever import SummaryRetriever
from aion_knowledge.retrieval.search.vector_retriever import VectorRetriever
from aion_knowledge.retrieval.search.wiki_retriever import WikiRetriever

if TYPE_CHECKING:
    # 仅用于注解，避免运行时引入 langchain 依赖
    from aion_knowledge.infrastructure.embedder import Embedder
    from aion_knowledge.infrastructure.llm.client import LLMClient

logger = logging.getLogger(__name__)

# 路径名 → 配置开关名对照
# 8 个依赖后处理的路径（keyword / question_gen / ... / wiki）与 postproc_* 共用标记
_PATH_TO_CONFIG: dict[str, str] = {
    "bm25": "retrieval_path_bm25",
    "vector": "retrieval_path_vector",
    "faq": "retrieval_path_faq",
    "keyword": "postproc_keyword_extract",
    "question_gen": "postproc_question_gen",
    "summary": "postproc_summarizer",
    "raptor": "postproc_raptor",
    "graph": "postproc_graph_extract",
    "community": "postproc_community",
    "wiki": "postproc_wiki",
}

# 路径名 → 权重配置名对照
_PATH_TO_WEIGHT: dict[str, str] = {
    "bm25": "rrf_weight_bm25",
    "vector": "rrf_weight_vector",
    "keyword": "rrf_weight_keyword",
    "faq": "rrf_weight_faq",
    "question_gen": "rrf_weight_question_gen",
    "summary": "rrf_weight_summary",
    "raptor": "rrf_weight_raptor",
    "graph": "rrf_weight_graph",
    "community": "rrf_weight_community",
    "wiki": "rrf_weight_wiki",
}


class RetrievalRouter:
    """检索路由入口。

    初始化时按配置注册所有可用的检索器。
    可选的 llm/embedder 传递给需要它们的检索器（如 KG）。
    """

    def __init__(
        self,
        llm: LLMClient | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self._llm = llm
        self._embedder = embedder
        self._multi_retriever = MultiRetriever(
            timeout=settings.retrieval_path_timeout,
        )
        self._rrf_fuser = RRFFuser(
            k=settings.rrf_k,
            weights=self._build_weights(),
        )
        self._register_retrievers()

        # ── RewriteStage ──
        rewrite_llm = None
        if settings.query_rewrite_enabled:
            from aion_knowledge.infrastructure.llm.factory import create_llm
            rewrite_llm = create_llm()

        # ── Pipeline 编排 ──
        from aion_knowledge.retrieval.pipeline import RetrievalPipeline

        # stages 包未显式 re-export，直接按子模块导入
        from aion_knowledge.retrieval.pipeline.stages.mmr import MMRStage
        from aion_knowledge.retrieval.pipeline.stages.multi_recall import MultiRecallStage
        from aion_knowledge.retrieval.pipeline.stages.reranker import RerankerStage
        from aion_knowledge.retrieval.pipeline.stages.rewrite_stage import RewriteStage
        from aion_knowledge.retrieval.pipeline.stages.rrf_fusion import RRFFusionStage
        from aion_knowledge.retrieval.pipeline.stages.topk_truncate import TopKTruncateStage

        # ── 检索管线配置 ──
        # 按序执行：查询改写 → 多路召回 → RRF 融合 → 精排 → MMR 去重 → 截断
        # RewriteStage 受 query_rewrite_enabled 开关控制，关闭时传入 None 直接透传
        self._pipeline = RetrievalPipeline([
            RewriteStage(llm_client=rewrite_llm, enabled=settings.query_rewrite_enabled),
            MultiRecallStage(self._multi_retriever),
            RRFFusionStage(self._rrf_fuser),
            RerankerStage(),
            MMRStage(),
            TopKTruncateStage(),
        ])

    def _build_weights(self) -> dict[str, float]:
        """从配置读取权重，构建路径→权重的映射。"""
        weights: dict[str, float] = {}
        for path_name, config_key in _PATH_TO_WEIGHT.items():
            weights[path_name] = getattr(settings, config_key, 0.0)
        return weights

    def _register_retrievers(self) -> None:
        """根据配置启用开关注册检索器。"""
        registry: list[tuple[str, BaseRetriever]] = [
            ("bm25", BM25Retriever()),
            ("vector", VectorRetriever()),
            ("keyword", KeywordRetriever()),
            ("faq", FAQRetriever()),
            ("question_gen", QuestionGenRetriever()),
            ("summary", SummaryRetriever()),
            ("raptor", RAPTORRetriever()),
            ("graph", GraphRetriever()),
            ("community", CommunityRetriever()),
            ("wiki", WikiRetriever()),
        ]

        registered = 0
        for name, retriever in registry:
            config_key = _PATH_TO_CONFIG.get(name)
            if config_key and not getattr(settings, config_key, True):
                logger.info("Retriever %r disabled by config", name)
                continue
            self._multi_retriever.register(retriever)
            registered += 1

        logger.info("RetrievalRouter: registered %d/%d retrievers", registered, len(registry))

    async def search(
        self,
        query: str,
        kb_id: str,
        query_embedding: list[float] | None = None,
        top_k: int = 10,
        path_top_k: int = 20,
        enabled_paths: set[str] | None = None,
    ) -> tuple[list[ChunkResult], dict[str, int], dict[str, dict[str, int]]]:
        """执行多路检索管线（召回 → RRF → Reranker → MMR → 截断）。"""
        start = time.perf_counter()
        ctx = PipelineContext(
            query=query,
            kb_id=kb_id,
            top_k=top_k,
            path_top_k=path_top_k,
            query_embedding=query_embedding,
            enabled_paths=enabled_paths,
        )
        ctx._llm = self._llm
        ctx._embedder = self._embedder

        await self._pipeline.run(ctx)

        # 统计 source_breakdown / path_stats
        path_results = ctx.path_results
        fused = ctx.results
        source_breakdown: dict[str, int] = {}
        path_stats: dict[str, dict[str, int]] = {}
        for path_name, results in path_results.items():
            source_breakdown[path_name] = len(results)
            in_final = sum(1 for r in fused if path_name in r.source_paths)
            path_stats[path_name] = {"results": len(results), "in_final": in_final}

        elapsed_ms = round((time.perf_counter() - start) * 1000)
        logger.info(
            "RetrievalRouter.search 执行完成: query=%r kb_id=%s elapsed=%dms",
            query,
            kb_id,
            elapsed_ms,
        )
        return fused, source_breakdown, path_stats
