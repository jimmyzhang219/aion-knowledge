"""Tests for MMR — Maximal Marginal Relevance diversity."""
from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

from aion_knowledge.retrieval.base import ChunkResult
from aion_knowledge.retrieval.pipeline.context import PipelineContext
from aion_knowledge.retrieval.pipeline.stages.mmr import MMRStage, cosine_sim, mmr_diversify


class TestCosineSim:
    def test_identical(self):
        assert cosine_sim([1.0, 0.0], [1.0, 0.0]) == 1.0

    def test_orthogonal(self):
        assert cosine_sim([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_opposite(self):
        assert cosine_sim([1.0, 0.0], [-1.0, 0.0]) == -1.0

    def test_zero_vector(self):
        assert cosine_sim([0.0, 0.0], [1.0, 0.0]) == 0.0
        assert cosine_sim([0.0, 0.0], [0.0, 0.0]) == 0.0


class TestMMRDiversify:
    def test_returns_top_k(self):
        results = [{"content": "A" * 10, "score": 0.9 - i * 0.1} for i in range(5)]
        diverse = mmr_diversify(results, top_k=3, mmr_lambda=0.7)
        assert len(diverse) == 3

    def test_first_item_is_highest_score(self):
        results = [{"content": "A" * 10, "score": s} for s in [0.9, 0.8, 0.7]]
        diverse = mmr_diversify(results, top_k=3, mmr_lambda=0.7)
        assert diverse[0]["score"] == 0.9

    def test_empty_input(self):
        assert mmr_diversify([], top_k=10) == []

    def test_less_than_top_k(self):
        results = [{"content": "A", "score": 0.9}]
        diverse = mmr_diversify(results, top_k=10)
        assert len(diverse) == 1

    def test_with_embedding_cosine(self):
        """embedding cosine 去重应优先选向量差异大的结果。"""
        results = [
            {"content": "苹果 香蕉 橘子", "score": 0.9},
            {"content": "苹果 香蕉 橘子 西瓜", "score": 0.8},
            {"content": "电脑 鼠标 键盘", "score": 0.7},
        ]
        # 向量与内容一致：前三项高度相似，第三项差异大
        embeddings = [
            [1.0, 0.0, 0.0],  # 与 #1 相似
            [0.9, 0.1, 0.0],  # 与 #0 相似
            [0.0, 1.0, 0.0],  # 与 #0/#1 差异大
        ]
        diverse = mmr_diversify(results, top_k=3, mmr_lambda=0.5, embeddings=embeddings)
        assert len(diverse) == 3
        # 第一项应为 score 最高的 #0
        assert diverse[0]["score"] == 0.9
        # 第二项应为向量差异最大的 #2（score 最低但 diversity 高）
        assert diverse[1]["content"] == "电脑 鼠标 键盘"
        # 第三项是最后剩下的 #1（与 #0 冗余高，排最后）
        assert diverse[2]["content"] == "苹果 香蕉 橘子 西瓜"

    def test_embedding_fallback_to_token_jaccard_on_mismatch(self):
        """embeddings 长度不匹配时降级为 jieba-Jaccard（按 content 分词）。"""
        results = [{"content": "苹果香蕉", "score": 0.9}]
        diverse = mmr_diversify(results, embeddings=[[1.0], [2.0]])  # 长度不匹配
        assert len(diverse) == 1

    def test_fallback_uses_jieba_tokenization(self):
        """降级路径用 jieba 分词驱动多样性（非 whitespace split）。

        #1 与 #0 共享"苹果"等词（jieba jaccard>0），冗余高 →
        MMR 将不共享词的 #2 提前到第二位。若误用 split()，#1/#0 零重叠
        → #1 按 score 排第二，测试失败。
        """
        results = [
            {"content": "苹果很好吃", "score": 0.9},
            {"content": "苹果不好吃", "score": 0.8},
            {"content": "汽车维修保养", "score": 0.7},
        ]
        diverse = mmr_diversify(results, top_k=3, mmr_lambda=0.5)
        assert diverse[1]["content"] == "汽车维修保养"


def _result(cid: str, content: str, score: float, source: str = "vector") -> ChunkResult:
    return ChunkResult(
        chunk_id=cid, kb_id="k", document_id="d", content=content,
        score=score, source_paths=[source],
    )


def _session_with_rows(rows_by_call: list[list]) -> AsyncMock:
    """构造 mock session：execute 第 i 次调用返回第 i 组 rows。"""
    session = AsyncMock()
    results = []
    for rows in rows_by_call:
        r = MagicMock()
        r.__iter__.return_value = list(rows)
        results.append(r)
    session.execute = AsyncMock(side_effect=results)
    return session


class TestMMRStage:
    async def test_chunk_level_uses_db_embeddings(self, monkeypatch):
        """chunk 级候选走 chunk_vector 取回向量，cosine 路径，不调 embedder。"""
        results = [_result("c1", "苹果", 0.9), _result("c2", "电脑", 0.8)]
        ctx = PipelineContext(query="q", kb_id="k", results=results)

        row1, row2 = MagicMock(), MagicMock()
        row1.chunk_id, row1.embedding = "c1", "[1.0,0.0]"
        row2.chunk_id, row2.embedding = "c2", "[0.0,1.0]"
        session = _session_with_rows([[row1, row2]])

        @asynccontextmanager
        async def fake_session():
            yield session
        monkeypatch.setattr("aion_knowledge.retrieval.pipeline.stages.mmr.get_session", fake_session)

        await MMRStage(mmr_lambda=0.5).run(ctx)

        assert len(ctx.results) == 2
        assert ctx.results[0].chunk_id == "c1"
        # chunk 级候选全部命中 chunk_vector，仅 1 次 DB 查询（无聚合表回退查询）
        assert session.execute.call_count == 1

    async def test_mixed_chunk_and_raptor_uses_each_table(self, monkeypatch):
        """chunk + raptor 混合：chunk 级走 chunk_vector，raptor 走 chunk_raptor，全 cosine。"""
        results = [_result("c1", "苹果", 0.9, "vector"), _result("r1", "摘要", 0.8, "raptor")]
        ctx = PipelineContext(query="q", kb_id="k", results=results)

        chunk_row = MagicMock()
        chunk_row.chunk_id, chunk_row.embedding = "c1", "[1.0,0.0]"
        raptor_row = MagicMock()
        raptor_row.chunk_id, raptor_row.embedding = "r1", "[0.0,1.0]"
        session = _session_with_rows([[chunk_row], [raptor_row]])

        @asynccontextmanager
        async def fake_session():
            yield session
        monkeypatch.setattr("aion_knowledge.retrieval.pipeline.stages.mmr.get_session", fake_session)

        await MMRStage(mmr_lambda=0.5).run(ctx)

        assert {r.chunk_id for r in ctx.results} == {"c1", "r1"}
        second_sql = session.execute.call_args_list[1].args[0].text
        assert "chunk_raptor" in second_sql

    async def test_missing_embedding_falls_back_to_jaccard(self, monkeypatch):
        """任一候选向量缺失 → 整批降级 jaccard（mmr_diversify 内置 all-or-nothing）。"""
        results = [
            _result("c1", "苹果很好吃", 0.9),
            _result("c2", "苹果不好吃", 0.8),
            _result("c3", "汽车维修保养", 0.7),
        ]
        ctx = PipelineContext(query="q", kb_id="k", results=results)

        session = _session_with_rows([[]])

        @asynccontextmanager
        async def fake_session():
            yield session
        monkeypatch.setattr("aion_knowledge.retrieval.pipeline.stages.mmr.get_session", fake_session)

        await MMRStage(mmr_lambda=0.5).run(ctx)

        assert ctx.results[1].chunk_id == "c3"
