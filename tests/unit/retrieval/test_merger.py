"""Tests for context merger — FAQ direct answer channel."""
from __future__ import annotations

from aion_knowledge.retrieval.context.merger import merge_context


class TestFaqDirectAnswer:
    def test_high_score_faq_direct(self):
        results = [
            {
                "chunk_type": "faq",
                "score": 0.9,
                "content": "Q: 如何重置密码？\nA: 进入设置页面",
                "metadata": {},
            },
        ]
        context = merge_context(results, query="重置密码")
        faq_items = [c for c in context if c.get("type") == "faq_direct"]
        assert len(faq_items) == 1
        assert "如何重置密码" in faq_items[0]["content"]

    def test_low_score_faq_not_direct(self):
        results = [
            {
                "chunk_type": "faq",
                "score": 0.5,
                "content": "Q: test\nA: answer",
                "metadata": {},
            },
        ]
        context = merge_context(results, query="test")
        faq_items = [c for c in context if c.get("type") == "faq_direct"]
        assert len(faq_items) == 0

    def test_non_faq_not_affected(self):
        results = [
            {"chunk_type": "text", "content": "normal doc content", "score": 0.95},
        ]
        context = merge_context(results, query="test")
        faq_items = [c for c in context if c.get("type") == "faq_direct"]
        assert len(faq_items) == 0
        assert any(c.get("type") == "chunk" for c in context)
