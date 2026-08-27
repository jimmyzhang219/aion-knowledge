"""WikiModule 测试 — KB 级页面池管线（MAP 文档级 / REDUCE 跨文档合并 / REFINE）。"""
from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from aion_knowledge.pipeline.postproc.base import PostProcContext
from aion_knowledge.pipeline.postproc.wiki.processor import WikiModule

TEST_KB_UUID = uuid.uuid4()
CHUNK_UUID = uuid.uuid4()


@pytest.mark.asyncio
async def test_compose_full_text_orders_and_joins():
    """MAP 输入应为按 seq_num 排序拼接的文档原文（含图片/表格内容）。"""
    chunks = [
        {"seq_num": 2, "content": "表格内容"},
        {"seq_num": 1, "content": "第一段"},
        {"seq_num": 3, "content": "图片 VLM 描述"},
    ]
    assert WikiModule._compose_full_text(chunks) == "第一段\n\n表格内容\n\n图片 VLM 描述"


@pytest.mark.asyncio
async def test_extract_candidates_document_level_single_call():
    """MAP 应整篇文档一次 LLM 调用提取候选（不再逐 chunk）。"""
    module = WikiModule()
    llm = MagicMock()
    llm.generate_structured = AsyncMock(return_value={
        "candidates": [{"term": "机器学习", "type": "concept", "reason": "核心概念"}],
    })
    chunks = [{"chunk_uuid": str(CHUNK_UUID), "content": "机器学习是人工智能的分支。", "seq_num": 1}]
    result = await module._extract_candidates(llm, chunks)
    assert result == [{"term": "机器学习", "type": "concept", "reason": "核心概念"}]
    llm.generate_structured.assert_awaited_once()


@pytest.mark.asyncio
async def test_merge_with_existing_slug_hit_and_judge():
    """REDUCE：slug 规则命中 → merge；未命中 LLM 判同 → merge；否则 new。"""
    module = WikiModule()
    llm = MagicMock()
    existing = [{"slug": "machine_learning", "title": "机器学习"}]

    async def fake_judge(llm, term, existing):
        return "machine_learning" if term == "AI" else None

    with patch.object(module, "_judge_same", side_effect=fake_judge):
        merged, new = await module._merge_with_existing(
            llm,
            [
                {"term": "Machine Learning", "type": "concept"},
                {"term": "AI", "type": "concept"},
                {"term": "深度学习", "type": "concept"},
            ],
            existing,
        )
    assert [m["existing_slug"] for m in merged] == ["machine_learning", "machine_learning"]
    assert len(new) == 1 and new[0]["term"] == "深度学习"


@pytest.mark.asyncio
async def test_process_merges_and_creates():
    """process 主流程：new 页面插入；merge 更新已有行引用。"""
    module = WikiModule()
    ctx = PostProcContext(document_id="d1", kb_id=str(TEST_KB_UUID), doc_name="t.md")
    chunks = [{"chunk_uuid": str(CHUNK_UUID), "content": "机器学习是人工智能的一个分支。", "seq_num": 1}]

    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock()
    # execute 按调用顺序：查已有页面 → []；merge 目标 SELECT → 命中行；UPDATE/checkpoint 结果未用
    merge_row = SimpleNamespace(
        first=lambda: SimpleNamespace(
            source_refs=["d2"], chunk_refs=["c_old"], payload={"source_terms": ["旧概念"]},
        ),
    )
    mock_session.execute = AsyncMock(side_effect=[[], merge_row, MagicMock(), MagicMock()])
    mock_session.add = Mock()
    mock_session.flush = AsyncMock()

    with (
        patch.object(module, "_extract_candidates",
                     AsyncMock(return_value=[{"term": "机器学习", "type": "concept", "reason": ""}])),
        patch.object(module, "_merge_with_existing",
                     AsyncMock(return_value=(
                         [{"term": "旧概念", "type": "concept", "existing_slug": "old"}],
                         [{"term": "机器学习", "type": "concept", "reason": ""}],
                     ))),
        patch.object(module, "_generate_page",
                     AsyncMock(return_value={"slug": "machine_learning", "content": "## 概述\n机器学习是..."})),
        patch("aion_knowledge.pipeline.postproc.wiki.processor.get_llm_client_for_module"),
        patch("aion_knowledge.infrastructure.db.get_session", return_value=mock_session),
    ):
        count = await module.process(ctx, chunks)
        assert count == 2  # 1 new + 1 merge


@pytest.mark.asyncio
async def test_judge_same_rejects_hallucinated_slug():
    """LLM 返回的 existing_slug 不在已有页面集合时视为未命中。"""
    module = WikiModule()
    llm = MagicMock()
    llm.generate_structured = AsyncMock(return_value={
        "is_same": True, "existing_slug": "ghost_slug",
    })
    existing = [{"slug": "machine_learning", "title": "机器学习"}]
    assert await module._judge_same(llm, "AI", existing) is None


@pytest.mark.asyncio
async def test_extract_out_links():
    """应从 content 抽取 [[slug]] 与 [[slug|显示名]]，去重保序。"""
    content = "见 [[machine_learning|机器学习]] 与 [[deep_learning]]，重复 [[machine_learning]]。"
    assert WikiModule._extract_out_links(content) == ["machine_learning", "deep_learning"]


@pytest.mark.asyncio
async def test_process_updates_in_links_for_out_links():
    """页面写入后，被链接页面（白名单内）的 in_links 应追加本页 slug。"""
    module = WikiModule()
    ctx = PostProcContext(document_id="d1", kb_id=str(TEST_KB_UUID), doc_name="t.md")
    chunks = [{"chunk_uuid": str(CHUNK_UUID), "content": "机器学习是人工智能的一个分支。", "seq_num": 1}]

    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock()
    # 已有 deep_learning 页面：白名单含 deep_learning；本页 slug 由 term 派生（机器学习）
    mock_session.execute = AsyncMock(return_value=[
        SimpleNamespace(page_slug="deep_learning", page_title="深度学习"),
    ])
    mock_session.add = Mock()
    mock_session.flush = AsyncMock()

    with (
        patch.object(module, "_extract_candidates",
                     AsyncMock(return_value=[{"term": "机器学习", "type": "concept", "reason": ""}])),
        patch.object(module, "_merge_with_existing", AsyncMock(return_value=([], [
            {"term": "机器学习", "type": "concept", "reason": ""},
        ]))),
        patch.object(module, "_generate_page",
                     AsyncMock(return_value={"slug": "machine_learning",
                                             "content": "见 [[deep_learning|深度学习]] 与 [[ghost_slug]]"})),
        patch("aion_knowledge.pipeline.postproc.wiki.processor.get_llm_client_for_module"),
        patch("aion_knowledge.infrastructure.db.get_session", return_value=mock_session),
    ):
        await module.process(ctx, chunks)
    # 存储 slug 恒由 term 派生：LLM 返回的 "machine_learning" 不落库
    added = mock_session.add.call_args.args[0]
    assert added.page_slug == "机器学习"
    # TextClause 的 repr 不含 SQL 文本，须经 .text 取 SQL 判断
    in_link_calls = [c for c in mock_session.execute.call_args_list
                     if "in_links" in c.args[0].text and "array_append" in c.args[0].text]
    assert len(in_link_calls) == 1  # ghost_slug 被白名单过滤，仅 deep_learning 一个目标
    # params 以位置参数传入（execute(sql, params)），位于 args[1]
    assert in_link_calls[0].args[1]["link_slug"] == "机器学习"  # 本页 slug（term 派生）
    assert in_link_calls[0].args[1]["target"] == "deep_learning"


@pytest.mark.asyncio
async def test_process_dedupes_same_slug_within_batch():
    """同文档内 slug 相同的候选应合并为一页（防唯一约束冲突），source_terms 累积。"""
    module = WikiModule()
    ctx = PostProcContext(document_id="d1", kb_id=str(TEST_KB_UUID), doc_name="t.md")
    chunks = [{"chunk_uuid": str(CHUNK_UUID), "content": "AI 与 Ai 是同一概念。", "seq_num": 1}]

    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock()
    mock_session.execute = AsyncMock(return_value=[])  # 无已有页面
    mock_session.add = Mock()
    mock_session.flush = AsyncMock()

    with (
        patch.object(module, "_extract_candidates",
                     AsyncMock(return_value=[
                         {"term": "AI", "type": "concept", "reason": ""},
                         {"term": "Ai", "type": "concept", "reason": ""},
                     ])),
        patch.object(module, "_merge_with_existing", AsyncMock(return_value=([], [
            {"term": "AI", "type": "concept", "reason": ""},
            {"term": "Ai", "type": "concept", "reason": ""},
        ]))),
        patch.object(module, "_generate_page",
                     AsyncMock(return_value={"slug": "ai", "content": "## 概述\nAI"})),
        patch("aion_knowledge.pipeline.postproc.wiki.processor.get_llm_client_for_module"),
        patch("aion_knowledge.infrastructure.db.get_session", return_value=mock_session),
    ):
        count = await module.process(ctx, chunks)
    assert count == 1  # 两候选合并为一页
    added = mock_session.add.call_args[0][0]
    assert added.page_slug == "ai"
    assert added.payload["source_terms"] == ["AI", "Ai"]
