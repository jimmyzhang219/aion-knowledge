"""AccessLogMiddleware 访问日志测试。"""
from __future__ import annotations

import logging

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from aion_knowledge.api.middleware import AccessLogMiddleware, TraceIDMiddleware
from aion_knowledge.common.trace import get_trace_id

_LOG_NAME = "aion_knowledge.api.middleware"


@pytest.fixture
def client() -> httpx.AsyncClient:
    """仅挂 AccessLogMiddleware 的最小应用，避免触发 DB/LLM 初始化。"""
    app = FastAPI()

    @app.post("/json")
    async def json_ep() -> dict:
        return {"ok": True, "msg": "你好"}

    @app.post("/stream")
    async def stream_ep():
        async def gen():
            for i in range(3):
                yield f"chunk{i}\n"

        return StreamingResponse(gen())

    @app.post("/sse")
    async def sse_ep():
        async def gen():
            for i in range(3):
                yield f"chunk{i}\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    app.add_middleware(AccessLogMiddleware)
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    )


async def _log_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    records = [r for r in caplog.records if r.name == _LOG_NAME]
    assert records, "未产生访问日志"
    return [r.getMessage() for r in records]


async def _request_line(caplog: pytest.LogCaptureFixture) -> str:
    lines = await _log_lines(caplog)
    req = [line for line in lines if line.startswith("REQUEST ")]
    assert req, f"缺请求日志: {lines}"
    return req[-1]


async def _response_line(caplog: pytest.LogCaptureFixture) -> str:
    lines = await _log_lines(caplog)
    resp = [line for line in lines if line.startswith("RESPONSE ")]
    assert resp, f"缺响应日志: {lines}"
    return resp[-1]


async def test_json_response_logged_with_body_and_elapsed(
    client: httpx.AsyncClient, caplog: pytest.LogCaptureFixture
) -> None:
    """JSON 响应：REQUEST 带 req_body，RESPONSE 带 resp_body / status / 耗时。"""
    caplog.set_level(logging.INFO, logger=_LOG_NAME)
    r = await client.post("/json", json={"q": "test"})
    assert r.status_code == 200
    assert r.json() == {"ok": True, "msg": "你好"}

    req = await _request_line(caplog)
    assert "REQUEST POST /json" in req
    assert 'req_body={"q":"test"}' in req
    assert "trace_id=" in req

    resp = await _response_line(caplog)
    assert "RESPONSE POST /json" in resp
    assert "status=200" in resp
    assert "ms" in resp
    assert 'resp_body={"ok":true' in resp
    assert "你好" in resp


async def test_streaming_response_logged_after_completion(
    client: httpx.AsyncClient, caplog: pytest.LogCaptureFixture
) -> None:
    """流式响应：RESPONSE 在流发送完毕后打出，包含响应体与耗时，流内容完整送达。"""
    caplog.set_level(logging.INFO, logger=_LOG_NAME)
    r = await client.post("/stream", json={"q": "test"})
    assert r.status_code == 200
    assert r.text == "chunk0\nchunk1\nchunk2\n"

    req = await _request_line(caplog)
    assert "REQUEST POST /stream" in req

    resp = await _response_line(caplog)
    assert "RESPONSE POST /stream" in resp
    assert "status=200" in resp
    assert "ms" in resp
    assert "resp_body=chunk0" in resp
    assert "chunk2" in resp


async def test_sse_response_logged_before_return_and_stream_end(
    client: httpx.AsyncClient, caplog: pytest.LogCaptureFixture
) -> None:
    """SSE 流式：RESPONSE（状态+耗时，无 body）在返回前打出，流结束补记 STREAM-END。"""
    caplog.set_level(logging.INFO, logger=_LOG_NAME)
    r = await client.post("/sse", json={"q": "test"})
    assert r.status_code == 200
    assert r.text == "chunk0\nchunk1\nchunk2\n"

    lines = await _log_lines(caplog)
    resp = [line for line in lines if line.startswith("RESPONSE POST /sse")]
    end = [line for line in lines if line.startswith("STREAM-END POST /sse")]
    assert len(resp) == 1, lines
    assert len(end) == 1, lines
    # RESPONSE 必须先于 STREAM-END 打出（返回前 → 流结束）
    assert lines.index(resp[0]) < lines.index(end[0]), lines
    assert "status=200" in resp[0]
    assert "ms" in resp[0]
    assert "streaming" in resp[0]
    assert "resp_body" not in resp[0]
    assert "resp_body=chunk0" in end[0]
    assert "chunk2" in end[0]


@pytest.fixture
def error_client() -> httpx.AsyncClient:
    """处理器抛未捕获异常的测试应用。"""
    app = FastAPI()

    @app.post("/boom")
    async def boom_ep() -> dict:
        raise RuntimeError("handler exploded")

    app.add_middleware(AccessLogMiddleware)
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    )


async def test_handler_exception_logs_request_and_500_response(
    error_client: httpx.AsyncClient, caplog: pytest.LogCaptureFixture
) -> None:
    """处理器未捕获异常：仍有 REQUEST + status=500 的 RESPONSE 两条日志。"""
    caplog.set_level(logging.INFO, logger=_LOG_NAME)
    r = await error_client.post("/boom", json={"q": "x"})
    assert r.status_code == 500

    req = await _request_line(caplog)
    assert "REQUEST POST /boom" in req
    assert 'req_body={"q":"x"}' in req

    resp = await _response_line(caplog)
    assert "RESPONSE POST /boom" in resp
    assert "status=500" in resp
    assert "ms" in resp


# TraceIDMiddleware 测试：X-Trace-ID 生成/透传/回显，X-Request-ID 移除。


@pytest.fixture
def trace_client() -> httpx.AsyncClient:
    """仅挂 TraceIDMiddleware 的最小应用。"""
    app = FastAPI()

    @app.get("/show")
    async def show() -> dict:
        # handler 内直接读 contextvar，验证请求内可见
        return {"tid": get_trace_id()}

    app.add_middleware(TraceIDMiddleware)
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    )


async def test_trace_id_generated_and_echoed(trace_client: httpx.AsyncClient):
    """不带 header：生成 uuid7 并回显，handler 内可见。"""
    r = await trace_client.get("/show")
    assert r.status_code == 200
    tid = r.headers.get("X-Trace-ID")
    assert tid and len(tid) == 36
    assert r.json()["tid"] == tid


async def test_trace_id_passthrough(trace_client: httpx.AsyncClient):
    """带 X-Trace-ID：透传同一值并回显。"""
    r = await trace_client.get("/show", headers={"X-Trace-ID": "caller-trace-1"})
    assert r.status_code == 200
    assert r.headers.get("X-Trace-ID") == "caller-trace-1"
    assert r.json()["tid"] == "caller-trace-1"


async def test_legacy_x_request_id_ignored(trace_client: httpx.AsyncClient):
    """旧 X-Request-ID 头不再回显、不作为 trace_id 来源。"""
    r = await trace_client.get("/show", headers={"X-Request-ID": "legacy"})
    assert r.status_code == 200
    assert "X-Request-ID" not in r.headers
    assert r.headers.get("X-Trace-ID") != "legacy"
