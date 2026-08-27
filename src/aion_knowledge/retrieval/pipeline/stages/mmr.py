"""MMRStage — 从 DB 取回已有向量 + MMR 多样性重排。"""
from __future__ import annotations

import json
from dataclasses import asdict
from math import sqrt
from typing import Any

from sqlalchemy import text

from aion_knowledge.infrastructure.db import get_session
from aion_knowledge.infrastructure.text_similarity import jaccard, tokenize
from aion_knowledge.retrieval.base import ChunkResult
from aion_knowledge.retrieval.pipeline.base import Stage
from aion_knowledge.retrieval.pipeline.context import PipelineContext
from aion_knowledge.storage.relational.vector_repo import VectorRepository

# 聚合级检索来源 → 向量所在表（chunk_id 非 chunk_text.id，不在 chunk_vector）
_AGGREGATE_TABLES: dict[str, str] = {"raptor": "chunk_raptor", "community": "chunk_community"}

# ── 相似度函数 ──────────────────────────────────────────────


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _norm(v: list[float]) -> float:
    return sqrt(_dot(v, v))


def cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine 相似度。"""
    na, nb = _norm(a), _norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return _dot(a, b) / (na * nb)


# ── MMR ─────────────────────────────────────────────────────


def mmr_diversify(
    results: list[dict[str, Any]],
    top_k: int = 10,
    mmr_lambda: float = 0.7,
    embeddings: list[list[float] | None] | None = None,
) -> list[dict[str, Any]]:
    """MMR 多样性重排序。

    MMR = λ × score(rᵢ) - (1-λ) × maxⱼ sim(rᵢ, selectedⱼ)

    每次从未选集中选择 MMR 分最高的结果加入已选集。

    Args:
        results: 候选列表（已按 score 降序）。
        top_k: 最终保留数。
        mmr_lambda: 多样性参数，越大越重相关度。
        embeddings: 与 ``results`` 等长的向量列表。传值时用 cosine_sim
                    度量冗余，否则降级为 jieba 分词 + Jaccard。

    Returns:
        多样性排序后的子集。
    """
    if not results:
        return []

    use_cosine = (
        embeddings is not None
        and len(embeddings) == len(results)
        and all(v for v in embeddings)  # 无零向量
    )

    # 剩余候选及其对应向量/token（同步维护）
    remaining: list[dict[str, Any]] = list(results)
    remaining_emb: list[list[float]] | None = (
        [e for e in embeddings if e is not None]
        if use_cosine and embeddings is not None else None
    )
    # 降级路径：入口预分词一次，避免循环内重复 jieba 分词
    remaining_tokens: list[set[str]] | None = (
        [tokenize(r.get("content", "")) for r in remaining] if not use_cosine else None
    )

    # 已选集及其对应向量/token
    selected: list[dict[str, Any]] = []
    selected_emb: list[list[float]] = []
    selected_tokens: list[set[str]] = []

    # 第一项：最高分
    selected.append(remaining.pop(0))
    if remaining_emb is not None:
        selected_emb.append(remaining_emb.pop(0))
    if remaining_tokens is not None:
        selected_tokens.append(remaining_tokens.pop(0))

    while len(selected) < top_k and remaining:
        mmr_scores = []
        for i, candidate in enumerate(remaining):
            relevance = candidate.get("score", 0)

            if use_cosine and remaining_emb is not None:
                # embedding 级语义冗余
                sim_scores = [cosine_sim(remaining_emb[i], s_emb)
                              for s_emb in selected_emb]
            elif remaining_tokens is not None:
                # 兜底：jieba 分词 + Jaccard 字面冗余
                sim_scores = [jaccard(remaining_tokens[i], s_tok)
                              for s_tok in selected_tokens]
            else:
                sim_scores = []

            redundancy = max(sim_scores) if sim_scores else 0
            mmr = mmr_lambda * relevance - (1 - mmr_lambda) * redundancy
            mmr_scores.append(mmr)

        best_idx = mmr_scores.index(max(mmr_scores))
        selected.append(remaining.pop(best_idx))
        if remaining_emb is not None:
            selected_emb.append(remaining_emb.pop(best_idx))
        if remaining_tokens is not None:
            selected_tokens.append(remaining_tokens.pop(best_idx))

    return selected


class MMRStage(Stage):
    """从 DB 取回已有 embedding → MMR 多样性重排，不做截断。

    向量来源：chunk 级走 chunk_vector.embedding；raptor/community 走各自表。
    度量 all-or-nothing：全部取到走 cosine，任一缺失走 jaccard（mmr_diversify 内置判定）。
    """

    def __init__(self, mmr_lambda: float = 0.7) -> None:
        self._mmr_lambda = mmr_lambda

    async def run(self, ctx: PipelineContext) -> None:
        if not ctx.results:
            return

        embeddings = await self._collect_embeddings(ctx.results)
        dicts = [asdict(r) for r in ctx.results]
        diverse = mmr_diversify(dicts, len(dicts), self._mmr_lambda, embeddings)
        ctx.results = [ChunkResult(**d) for d in diverse]

    async def _collect_embeddings(self, results: list[ChunkResult]) -> list[list[float] | None] | None:
        """从 DB 取回已有向量，按 results 顺序对齐。

        先一条 SQL 查 chunk_vector（覆盖所有 chunk 级候选）；
        未命中的（raptor/community）按 source_paths 查对应表。
        返回与 results 等长的列表，未取到的位置为 None（mmr_diversify 据此降级 jaccard）。
        """
        if not results:
            return None
        emb_map: dict[str, list[float]] = {}
        async with get_session() as session:
            emb_map.update(
                await VectorRepository(session).fetch_embeddings([r.chunk_id for r in results])
            )
            # 聚合级候选：表名来自模块常量白名单，非用户输入，可安全拼接
            for source, table in _AGGREGATE_TABLES.items():
                ids = [r.chunk_id for r in results
                       if r.chunk_id not in emb_map and source in r.source_paths]
                if ids:
                    rows = await session.execute(
                        text(f"SELECT id::text AS chunk_id, embedding FROM {table} "
                             f"WHERE id = ANY(CAST(:ids AS uuid[])) AND embedding IS NOT NULL"),
                        {"ids": ids},
                    )
                    emb_map.update({str(r.chunk_id): json.loads(str(r.embedding)) for r in rows})
        return [emb_map.get(r.chunk_id) for r in results]
