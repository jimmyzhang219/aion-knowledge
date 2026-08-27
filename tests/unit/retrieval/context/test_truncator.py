"""Tests for truncator — context truncation."""
from __future__ import annotations

from aion_knowledge.retrieval.context.truncator import truncate_context


class TestTruncateContext:
    """上下文截断测试。"""

    def test_under_limit_returns_all(self):
        """上下文未超限时全部保留。"""
        ctx = [{"content": "short text", "score": 0.9}]
        result = truncate_context(ctx, max_tokens=200)
        assert len(result) == 1

    def test_over_limit_drops_lowest(self):
        """超限时丢弃低分结果。"""
        # ~391 tokens
        long_text = "this is a long document with many words about various topics and concepts " * 30
        ctx = [
            {"content": long_text, "score": 0.9},
            {"content": long_text, "score": 0.1},
        ]
        result = truncate_context(ctx, max_tokens=500)
        assert len(result) == 1
        assert result[0]["score"] == 0.9

    def test_single_over_limit_keeps_one(self):
        """即使单条超限也至少保留一条。"""
        word = "this word repeated exceeds the token limit easily "
        ctx = [
            {"content": word * 300, "score": 0.5},
        ]
        result = truncate_context(ctx, max_tokens=100)
        assert len(result) == 1

    def test_empty_input(self):
        """空输入不崩溃。"""
        assert truncate_context([], max_tokens=100) == []

    def test_keeps_ordered_by_score(self):
        """保留的结果按 score 降序排列。"""
        ctx = [
            {"content": "a", "score": 0.3},
            {"content": "b", "score": 0.9},
            {"content": "c", "score": 0.6},
        ]
        result = truncate_context(ctx, max_tokens=4)
        assert result[0]["score"] == 0.9
        assert result[1]["score"] == 0.6


class TestTruncateContextDynamic:
    """max_tokens=None 的动态截断路径测试。"""

    def test_dynamic_returns_all_within_budget(self):
        ctx = [{"content": "hello world", "score": 0.9}]
        result = truncate_context(ctx, max_tokens=None)
        assert len(result) == 1

    def test_explicit_override_still_works(self):
        """显式传 max_tokens 时沿用旧路径，不触发动态计算。"""
        ctx = [{"content": "short", "score": 0.9}]
        result = truncate_context(ctx, max_tokens=200)
        assert len(result) == 1
