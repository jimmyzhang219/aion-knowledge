"""Tests for infrastructure.text_similarity — jieba 分词 + 集合相似度。"""
from __future__ import annotations

from aion_knowledge.infrastructure.text_similarity import dice, jaccard, tokenize


class TestTokenize:
    def test_chinese_segmented_into_words(self):
        # 中文按词分词，"苹果" 应作为一个独立 token
        tokens = tokenize("我喜欢苹果")
        assert "苹果" in tokens
        assert "我" in tokens

    def test_segments_into_multiple_tokens(self):
        tokens = tokenize("我喜欢吃苹果")
        assert len(tokens) > 1

    def test_empty_string_returns_empty_set(self):
        assert tokenize("") == set()


class TestJaccard:
    def test_identical_sets(self):
        assert jaccard({"a", "b"}, {"a", "b"}) == 1.0

    def test_disjoint_sets(self):
        assert jaccard({"a"}, {"b"}) == 0.0

    def test_partial_overlap(self):
        # |∩|=1, |∪|=3 → 1/3
        assert jaccard({"a", "b"}, {"a", "c"}) == 1 / 3

    def test_empty_set_returns_zero(self):
        assert jaccard(set(), {"a"}) == 0.0
        assert jaccard({"a"}, set()) == 0.0
        assert jaccard(set(), set()) == 0.0


class TestDice:
    def test_identical_sets(self):
        assert dice({"a", "b"}, {"a", "b"}) == 1.0

    def test_disjoint_sets(self):
        assert dice({"a"}, {"b"}) == 0.0

    def test_partial_overlap(self):
        # 2|∩|/(|A|+|B|) = 2*1/(2+2) = 0.5
        assert dice({"a", "b"}, {"a", "c"}) == 0.5

    def test_empty_set_returns_zero(self):
        assert dice(set(), {"a"}) == 0.0
        assert dice({"a"}, set()) == 0.0
