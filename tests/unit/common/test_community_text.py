"""build_community_text 拼接函数测试。"""
from __future__ import annotations

from aion_knowledge.common.community_text import build_community_text


def test_full_report_concatenation():
    """title + summary + 每条 finding 的 summary: explanation 依次拼接。"""
    text = build_community_text(
        "沉浸式VR技术栈",
        "该社区围绕 VR 大空间技术",
        [{"summary": "硬件", "explanation": "头显与定位"},
         {"summary": "软件", "explanation": "渲染引擎"}],
    )
    assert text == "沉浸式VR技术栈\n该社区围绕 VR 大空间技术\n硬件: 头显与定位\n软件: 渲染引擎"


def test_empty_parts_skipped():
    """空片段跳过，不产生多余空行。"""
    text = build_community_text("", "仅摘要", [])
    assert text == "仅摘要"


def test_empty_findings_and_title():
    """全部为空返回空串（调用方据此跳过嵌入）。"""
    assert build_community_text("", "", []) == ""


def test_findings_none_is_handled():
    """findings 为 None（JSONB 列可能为 NULL）不崩溃。"""
    assert build_community_text("标题", "摘要", None) == "标题\n摘要"


def test_finding_missing_explanation_key():
    """findings 元素缺 explanation key 不崩溃，等价空串。"""
    assert build_community_text("标题", "摘要", [{"summary": "硬件"}]) == "标题\n摘要\n硬件: "


def test_empty_summary_skipped():
    """summary 为空跳过，不产生多余空行。"""
    assert build_community_text("标题", "", []) == "标题"
