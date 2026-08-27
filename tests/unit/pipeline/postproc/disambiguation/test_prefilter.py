"""Disambiguation 预过滤/候选对生成测试。"""
from __future__ import annotations

from aion_knowledge.pipeline.postproc.disambiguation.processor import (
    DisambiguationModule,
    _generate_candidates,
    _is_cjk,
    _is_similar,
    _levenshtein,
)

module = DisambiguationModule()


def test_is_cjk():
    assert _is_cjk("苹果") is True
    assert _is_cjk("OpenAI") is False
    assert _is_cjk("") is False
    assert _is_cjk("123") is False
    assert _is_cjk("テスト") is False  # 日文假名不属于 CJK 统一表意文字


def test_levenshtein():
    assert _levenshtein("", "") == 0
    assert _levenshtein("a", "") == 1
    assert _levenshtein("", "a") == 1
    assert _levenshtein("abc", "abc") == 0
    assert _levenshtein("OpenAI", "0penAI") == 1
    assert _levenshtein("Apple", "Microsoft") == 9
    assert _levenshtein("ABC", "abc") == 3  # 大小写敏感


def test_is_similar_exact_match():
    assert _is_similar("Apple Inc.", "Apple Inc.") is True


def test_is_similar_edit_distance_within_threshold():
    assert _is_similar("OpenAI", "0penAI") is True  # 编辑距离 1


def test_is_similar_edit_distance_beyond_threshold():
    assert _is_similar("Apple", "Microsoft") is False  # 编辑距离 > 3


def test_is_similar_cjk_jaccard():
    """CJK 用字符集合重叠度。"""
    assert _is_similar("苹果公司", "苹果") is True
    assert _is_similar("阿里巴巴", "腾讯公司") is False


def test_generate_candidates_groups_by_type():
    entities = [
        {"entity_name": "Apple", "entity_type": "ORG"},
        {"entity_name": "App1e", "entity_type": "ORG"},
        {"entity_name": "Microsoft", "entity_type": "ORG"},
        {"entity_name": "iPhone", "entity_type": "PRODUCT"},
        {"entity_name": "iPhone15", "entity_type": "PRODUCT"},
    ]
    cands = _generate_candidates(entities)
    names_in_cands = {(a, b) for a, b in cands}
    assert ("App1e", "Apple") in names_in_cands
    for a, b in cands:
        type_a = next(e["entity_type"] for e in entities if e["entity_name"] == a)
        type_b = next(e["entity_type"] for e in entities if e["entity_name"] == b)
        assert type_a == type_b, f"{a}({type_a}) vs {b}({type_b}) 类型不同"


def test_generate_candidates_empty_list():
    assert _generate_candidates([]) == []
