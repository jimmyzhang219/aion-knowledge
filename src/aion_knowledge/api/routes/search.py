"""搜索 API 路由 — POST /api/v1/search。"""
from __future__ import annotations

import logging
from typing import cast

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from aion_knowledge.infrastructure.embedder import create_embedder
from aion_knowledge.models.schemas import (
    PathStat,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)
from aion_knowledge.retrieval.orchestrator.router import RetrievalRouter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["search"])

# 全局检索路由实例（由应用启动时初始化）
_router: RetrievalRouter | None = None


def get_router() -> RetrievalRouter:
    """获取 RetrievalRouter 单例（带 embedder 供 GraphRetriever 使用）。"""
    global _router
    if _router is None:
        embedder = create_embedder()
        _router = RetrievalRouter(embedder=embedder)
    return _router


async def _compute_query_embedding(query: str) -> list[float] | None:
    """根据配置的嵌入提供者计算查询向量。"""
    try:
        embedder = create_embedder()
        return await embedder.embed_query(query)
    except Exception as exc:
        logger.warning("Failed to compute query embedding: %s", exc)
        return None


@router.post("/search", response_model=None)
async def search(
    req: SearchRequest,
    retrieval_router: RetrievalRouter = Depends(get_router),
) -> SearchResponse | StreamingResponse:
    """执行多路检索，可选 LLM 生成回答。"""
    # 计算查询向量
    query_embedding = await _compute_query_embedding(req.query)

    try:
        fused, source_breakdown, path_stats = await retrieval_router.search(
            query=req.query,
            kb_id=req.kb_id,
            query_embedding=query_embedding,
            top_k=req.top_k,
            path_top_k=req.path_top_k,
            enabled_paths=set(req.enabled_paths) if req.enabled_paths else None,
        )
    except Exception as exc:
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail=str(exc))

    # 上下文截断（reranker 精排已在 Pipeline 中完成）
    # 检索结果统一按 chunk 处理，不做父块装配；转为 dict 供生成器消费
    from aion_knowledge.retrieval.context import truncate_context
    context = [r.__dict__ for r in fused]
    context = truncate_context(context)

    # LLM 生成
    if req.generate_answer:
        if req.stream:
            from aion_knowledge.infrastructure.llm import create_llm
            from aion_knowledge.retrieval.generator import build_sse_response
            llm = create_llm()
            return build_sse_response(req.query, context, llm)
        else:
            from aion_knowledge.retrieval.generator import generate_answer
            # stream=False 时恒为 str，收窄联合类型
            answer = cast(str, await generate_answer(req.query, context, stream=False))
    else:
        answer = None

    # 转换为响应模型
    results = [
        SearchResultItem(
            chunk_id=r.chunk_id,
            document_id=r.document_id,
            content=r.content,
            score=r.score,
            source_paths=r.source_paths,
            chunk_type=r.chunk_type,
            metadata=r.metadata,
        )
        for r in fused
    ]

    return SearchResponse(
        results=results,
        answer=answer,
        source_breakdown=source_breakdown,
        path_stats={k: PathStat(**v) for k, v in path_stats.items()},
        total_fused=len(fused),
        query=req.query,
    )
