"""UnifiedContext / PostProcTask trace_id 字段行为。"""
from __future__ import annotations

from aion_knowledge.common.trace import reset_trace_id, set_trace_id
from aion_knowledge.infrastructure.models import PostProcConfig, PostProcTask, UnifiedContext


def _ctx(**kw) -> UnifiedContext:
    return UnifiedContext(
        source=kw.get("source", "regular"),
        kb_id=kw.get("kb_id", "kb-1"),
        doc_name=kw.get("doc_name", "d.md"),
        suffix=kw.get("suffix", "md"),
        original_file_ref=kw.get("original_file_ref", "s3://x"),
        **{k: v for k, v in kw.items() if k not in ("source", "kb_id", "doc_name", "suffix", "original_file_ref")},
    )


def test_unified_context_trace_id_generated_when_unset():
    """无请求上下文：trace_id 自动生成 uuid7。"""
    ctx = _ctx()
    assert len(ctx.trace_id) == 36


def test_unified_context_trace_id_picks_up_request_value():
    """请求上下文内构造：trace_id 取 contextvar 值。"""
    token = set_trace_id("req-trace-1")
    try:
        ctx = _ctx()
        assert ctx.trace_id == "req-trace-1"
    finally:
        reset_trace_id(token)


def test_postproc_task_trace_id_generated_when_unset():
    """PostProcTask 无上下文时自动生成。"""
    task = PostProcTask(
        document_id="d-1", kb_id="kb-1", doc_name="t.md", chunk_count=1,
        postproc_config=PostProcConfig(),
    )
    assert len(task.trace_id) == 36


def test_modules_default_none():
    """重跑白名单默认 None = 全部已启用模块（保持向后兼容）。"""
    task = PostProcTask(
        document_id="d1", kb_id="kb1", doc_name="t.md", chunk_count=2,
        postproc_config=PostProcConfig(),
    )
    assert task.modules is None


def test_modules_explicit():
    """显式传入白名单。"""
    task = PostProcTask(
        document_id="d1", kb_id="kb1", doc_name="t.md", chunk_count=2,
        postproc_config=PostProcConfig(),
        modules=["raptor", "wiki"],
    )
    assert task.modules == ["raptor", "wiki"]
