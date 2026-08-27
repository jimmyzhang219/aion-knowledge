"""logger 配置测试 — 日志记录自动携带 trace_id。"""
from __future__ import annotations

import logging

import pytest

from aion_knowledge.common.logger import _LOG_FORMAT, get_logger
from aion_knowledge.common.trace import reset_trace_id, set_trace_id


def test_log_format_contains_trace_id() -> None:
    """日志格式 _LOG_FORMAT 包含 %(trace_id)s 占位符。"""
    assert "%(trace_id)s" in _LOG_FORMAT


def test_log_record_carries_trace_id(caplog: pytest.LogCaptureFixture) -> None:
    """请求上下文内：日志记录带 trace_id 值。"""
    caplog.set_level(logging.INFO)
    token = set_trace_id("t-42")
    try:
        get_logger("test.trace").info("hello")
    finally:
        reset_trace_id(token)
    assert caplog.records[-1].trace_id == "t-42"


def test_log_record_defaults_to_dash(caplog: pytest.LogCaptureFixture) -> None:
    """无请求上下文：日志记录 trace_id 为 '-'。"""
    caplog.set_level(logging.INFO)
    get_logger("test.trace").info("world")
    assert caplog.records[-1].trace_id == "-"


async def test_middleware_access_log_carries_trace_id(caplog: pytest.LogCaptureFixture) -> None:
    """双中间件 app：AccessLog 日志 record.trace_id == 请求 X-Trace-ID 值。

    回归：任务 2 修复中间件注册顺序（TraceID 最外层）后，
    AccessLog._log 在 trace_id 未 reset 前执行，日志必须带请求值。
    """
    import httpx
    from fastapi import FastAPI

    from aion_knowledge.api.middleware import AccessLogMiddleware, TraceIDMiddleware

    app = FastAPI()

    @app.get("/hello")
    async def hello() -> dict:
        return {"ok": True}

    app.add_middleware(AccessLogMiddleware)
    app.add_middleware(TraceIDMiddleware)

    caplog.set_level(logging.INFO, logger="aion_knowledge.api.middleware")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        r = await client.get("/hello", headers={"X-Trace-ID": "access-trace-7"})
        assert r.status_code == 200
        assert r.headers.get("X-Trace-ID") == "access-trace-7"

    records = [rec for rec in caplog.records if rec.name == "aion_knowledge.api.middleware"]
    assert records, "应产生 AccessLog 日志"
    assert records[-1].trace_id == "access-trace-7"
