"""文本相似度工具 —— jieba 分词 + 集合相似度（Jaccard/Dice），业务无关。

`jaccard`/`dice` 接受 ``set`` 而非 ``str``，是纯集合运算，与集合来源（词级/字符级/n-gram）
解耦，便于各处复用。`tokenize` 是独立的中文分词能力。
"""
from __future__ import annotations

import jieba


def tokenize(text: str) -> set[str]:
    """jieba 精确分词，返回去重 token 集合。空串返回空集。"""
    if not text:
        return set()
    return set(jieba.cut(text))


def jaccard(a: set[str], b: set[str]) -> float:
    """集合 Jaccard 相似度 = |A∩B| / |A∪B|；任一空集返回 0.0。"""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def dice(a: set[str], b: set[str]) -> float:
    """集合 Dice 相似度 = 2|A∩B| / (|A|+|B|)；任一空集返回 0.0。"""
    if not a or not b:
        return 0.0
    return 2 * len(a & b) / (len(a) + len(b))
