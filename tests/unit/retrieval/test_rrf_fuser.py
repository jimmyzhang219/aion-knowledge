"""RRF 融合器单元测试。"""
from __future__ import annotations

from aion_knowledge.retrieval.base import ChunkResult
from aion_knowledge.retrieval.orchestrator.rrf_fuser import RRFFuser


class TestRRFFuser:
    """RRF 融合功能测试。"""

    def test_single_path_no_fusion(self):
        """单路检索返回该路原始顺序。"""
        fuser = RRFFuser(k=60, weights={"vector": 1.0})
        results = {
            "vector": [
                ChunkResult(chunk_id="a", kb_id="kb1", document_id="d1", content="A", score=0.9, source_paths=["vector"]),
                ChunkResult(chunk_id="b", kb_id="kb1", document_id="d1", content="B", score=0.8, source_paths=["vector"]),
            ],
        }
        fused = fuser.fuse(results, top_k=5)
        assert len(fused) == 2
        assert fused[0].chunk_id == "a"
        assert fused[0].score > fused[1].score

    def test_two_paths_rrf_fusion(self):
        """双路 RRF 融合，同一 chunk 在两路中排名不同应融合得分。"""
        fuser = RRFFuser(k=60, weights={"vector": 0.7, "bm25": 0.3})
        results = {
            "vector": [
                ChunkResult(chunk_id="a", kb_id="kb1", document_id="d1", content="A", score=0.9, source_paths=["vector"]),
                ChunkResult(chunk_id="b", kb_id="kb1", document_id="d1", content="B", score=0.8, source_paths=["vector"]),
            ],
            "bm25": [
                ChunkResult(chunk_id="b", kb_id="kb1", document_id="d1", content="B", score=0.7, source_paths=["bm25"]),
                ChunkResult(chunk_id="a", kb_id="kb1", document_id="d1", content="A", score=0.6, source_paths=["bm25"]),
            ],
        }
        fused = fuser.fuse(results, top_k=5)
        assert len(fused) == 2
        # a: 0.7/61 + 0.3/62 = 0.01632
        # b: 0.7/62 + 0.3/61 = 0.01621
        # a 略高（vector 权重更大）
        assert fused[0].chunk_id == "a"
        assert fused[0].score > fused[1].score

    def test_top_k_truncation(self):
        """top_k 应正确截断。"""
        fuser = RRFFuser(k=60, weights={"vector": 1.0})
        results = {
            "vector": [
                ChunkResult(chunk_id=f"c{i}", kb_id="kb1", document_id="d1", content=str(i), score=1.0 - i * 0.1, source_paths=["vector"])
                for i in range(5)
            ],
        }
        fused = fuser.fuse(results, top_k=3)
        assert len(fused) == 3

    def test_empty_input(self):
        """空输入返回空列表。"""
        fuser = RRFFuser(k=60, weights={"vector": 1.0})
        assert fuser.fuse({}, top_k=10) == []

    def test_weight_not_normalized(self):
        """RRF 对非归一化权重仍然有效。"""
        fuser = RRFFuser(k=60, weights={"a": 2.0, "b": 1.0})
        results = {
            "a": [ChunkResult(chunk_id="x", kb_id="kb1", document_id="d1", content="X", score=1.0, source_paths=["a"])],
            "b": [ChunkResult(chunk_id="y", kb_id="kb1", document_id="d1", content="Y", score=1.0, source_paths=["b"])],
        }
        fused = fuser.fuse(results, top_k=10)
        assert len(fused) == 2

    def test_threshold_filter(self):
        """threshold_ratio 应过滤掉低于最高分指定比例的结果。"""
        fuser = RRFFuser(k=60, weights={"path": 1.0})
        results = {
            "path": [
                ChunkResult(chunk_id=f"c{i}", kb_id="kb1", document_id="d1",
                            content=str(i), score=1.0 - i * 0.1)
                for i in range(10)
            ],
        }
        # RRF(k=60) 分数 = 1/(60+rank)，rank=1..10 → [1/61..1/70]
        # 为了产生差异，用 threshold_ratio=0.95
        fused = fuser.fuse(results, top_k=10, threshold_ratio=0.95)
        # 阈值 ≈ 0.95 * max_score，只有前几项保留
        assert len(fused) in (3, 4)
        # 验证 threshold_ratio=None 时不启用过滤
        fused_all = fuser.fuse(results, top_k=10, threshold_ratio=None)
        assert len(fused_all) == 10

    def test_source_paths_populated(self):
        """融合后的 source_paths 应列出所有命中该 chunk 的路径。"""
        fuser = RRFFuser(k=60, weights={"a": 1.0, "b": 1.0, "c": 1.0})
        results = {
            "a": [ChunkResult(chunk_id="x", kb_id="kb1", document_id="d1", content="X", score=1.0, source_paths=["a"])],
            "b": [ChunkResult(chunk_id="x", kb_id="kb1", document_id="d1", content="X", score=0.9, source_paths=["b"])],
            "c": [ChunkResult(chunk_id="y", kb_id="kb1", document_id="d1", content="Y", score=1.0, source_paths=["c"])],
        }
        fused = fuser.fuse(results, top_k=10)
        result_x = next(r for r in fused if r.chunk_id == "x")
        assert set(result_x.source_paths) == {"a", "b"}
