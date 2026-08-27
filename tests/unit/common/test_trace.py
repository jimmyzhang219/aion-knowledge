"""common.trace 模块测试 — contextvar + trace_id 生成与注入。"""
from __future__ import annotations

from aion_knowledge.common.trace import (
    get_trace_id,
    reset_trace_id,
    set_trace_id,
    trace_id_var,
)


def test_get_trace_id_generates_uuid_when_unset():
    """contextvar 未设置时生成 uuid7 字符串。"""
    token = trace_id_var.set("")  # 确保起始为空，避免前序用例污染
    try:
        tid = get_trace_id()
        assert isinstance(tid, str)
        assert len(tid) == 36  # uuid7 字符串长度
    finally:
        trace_id_var.reset(token)


def test_get_trace_id_returns_set_value():
    """set 后 get 返回同一值，reset 后恢复生成行为。"""
    token = set_trace_id("my-trace-123")
    try:
        assert get_trace_id() == "my-trace-123"
    finally:
        reset_trace_id(token)
    cleanup = trace_id_var.set("")  # 避免 get_trace_id() 写回污染后续用例
    try:
        tid = get_trace_id()
        assert tid != "my-trace-123"
        assert len(tid) == 36
    finally:
        trace_id_var.reset(cleanup)


def test_get_trace_id_stable_within_context():
    """同一上下文内多次调用 get_trace_id 返回同一值（幂等：首次生成后写回 contextvar）。"""
    token = trace_id_var.set("")  # 确保起始为空
    try:
        tid1 = get_trace_id()
        tid2 = get_trace_id()
        assert tid1 == tid2
        assert len(tid1) == 36
    finally:
        trace_id_var.reset(token)


def test_set_trace_id_empty_string_triggers_generation():
    """set_trace_id("") 后再 get_trace_id 不返回空字符串，而是生成新 uuid7。"""
    token = set_trace_id("")
    try:
        tid = get_trace_id()
        assert tid != ""
        assert len(tid) == 36
    finally:
        reset_trace_id(token)
