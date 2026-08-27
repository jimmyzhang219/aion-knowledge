"""中间件注册。"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterable, MutableMapping
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import (
    BaseHTTPMiddleware,
    RequestResponseEndpoint,
    _StreamingResponse,
)
from starlette.requests import Request
from starlette.responses import Response

from aion_knowledge.common.trace import get_trace_id, reset_trace_id, set_trace_id
from aion_knowledge.common.uuid7 import uuid7

logger = logging.getLogger(__name__)


class AccessLogMiddleware(BaseHTTPMiddleware):
    """请求/响应访问日志：收到请求时记 REQUEST，响应返回前记 RESPONSE。

    请求日志在读取请求体后、进入处理器前打出（即使处理器后续异常也保证有条目）；
    普通响应在响应体发送完毕时打出——starlette 1.x 的 BaseHTTPMiddleware 把所有
    响应包成内部 _StreamingResponse，没有 .body 属性，响应体只能在 body_iterator
    迭代时拿到，因此用 tee 转发分片并累计，待响应发送完毕再打日志，此时才有完整
    响应体与真实耗时。SSE 流式响应（text/event-stream）则在返回前先记一条
    RESPONSE（状态 + 首字节耗时），流结束由 tee 补记 STREAM-END（完整响应体 +
    总耗时）。处理器抛未捕获异常时补记 status=500 后继续上抛。
    """

    _BODY_TRUNCATE_LEN = 4096

    @staticmethod
    def _should_log_body(request: Request) -> bool:
        """是否应记录 body（跳过文件上传等二进制请求）。"""
        ct = request.headers.get("content-type", "")
        if request.method not in ("POST", "PUT", "PATCH"):
            return False
        if "multipart/form-data" in ct:
            return False
        return True

    @staticmethod
    def _safe_decode(body: bytes) -> str:
        return body.decode("utf-8", errors="replace")

    def _log_request(self, request: Request, req_body: bytes, trace_id: str) -> None:
        parts = [
            f"REQUEST {request.method} {request.url.path}",
            f"trace_id={trace_id}",
        ]
        if req_body:
            body = self._safe_decode(req_body)[: self._BODY_TRUNCATE_LEN]
            parts.append(f"req_body={body}")
        logger.info(" | ".join(parts))

    def _log_response(
        self,
        request: Request,
        response: Response | None,
        captured: list[bytes],
        start: float,
        trace_id: str,
        status_code: int | None = None,
        streaming: bool = False,
    ) -> None:
        elapsed_ms = round((time.perf_counter() - start) * 1000)
        if status_code is None:
            # 显式传 status_code 的调用（异常路径 response=None）不走到这里
            assert response is not None
            status_code = response.status_code
        parts = [
            f"RESPONSE {request.method} {request.url.path}",
            f"trace_id={trace_id}",
            f"status={status_code}",
            f"{elapsed_ms}ms",
        ]
        if streaming:
            parts.append("streaming")
        resp_body = b"".join(captured)[: self._BODY_TRUNCATE_LEN]
        if resp_body:
            parts.append(f"resp_body={self._safe_decode(resp_body)}")
        logger.info(" | ".join(parts))

    def _log_stream_end(
        self,
        request: Request,
        captured: list[bytes],
        start: float,
        trace_id: str,
    ) -> None:
        """SSE 流结束日志：完整响应体与总耗时（含流式发送时间）。"""
        elapsed_ms = round((time.perf_counter() - start) * 1000)
        parts = [
            f"STREAM-END {request.method} {request.url.path}",
            f"trace_id={trace_id}",
            f"{elapsed_ms}ms",
        ]
        resp_body = b"".join(captured)[: self._BODY_TRUNCATE_LEN]
        if resp_body:
            parts.append(f"resp_body={self._safe_decode(resp_body)}")
        logger.info(" | ".join(parts))

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        start = time.perf_counter()
        # TraceIDMiddleware（最外层）已先注入 trace_id，此处读取用于两条日志的关联
        trace_id = get_trace_id()

        # 读取请求体（非二进制请求）
        req_body = b""
        if self._should_log_body(request):
            try:
                req_body = await request.body()
            except Exception:
                pass
        # 请求日志：收到请求即记录
        self._log_request(request, req_body, trace_id)

        # starlette 1.x 的 BaseHTTPMiddleware 把所有响应包成内部 _StreamingResponse，
        # 没有 .body 属性，响应体只能在 body_iterator 迭代时拿到。因此用 tee 转发
        # 分片并累计，待响应发送完毕后再打日志——此时才有完整响应体与真实耗时。
        captured: list[bytes] = []
        captured_len = 0

        try:
            response = await call_next(request)
        except Exception:
            # 处理器未捕获异常：ServerErrorMiddleware 将返回 500，补记响应日志后继续上抛
            self._log_response(request, None, captured, start, trace_id, status_code=500)
            raise

        if isinstance(response, _StreamingResponse):
            # SSE 流式响应：响应头已就绪、流尚未发送，先记一条 RESPONSE（返回前）。
            # 客户端可能长时间挂流，不等待流结束；非 SSE（JSON 等）则等 body 拿到再记。
            is_sse = "text/event-stream" in response.headers.get("content-type", "")
            if is_sse:
                self._log_response(request, response, captured, start, trace_id, streaming=True)

            # 先保存原始迭代器：tee 生成器体在首次迭代时才求值，
            # 若直接引用 response.body_iterator 会取到被替换后的 tee 自身
            original = response.body_iterator

            async def tee() -> AsyncIterable[bytes | str | memoryview | MutableMapping[str, Any]]:
                nonlocal captured_len
                try:
                    async for chunk in original:
                        if captured_len < self._BODY_TRUNCATE_LEN:
                            if isinstance(chunk, (bytes, bytearray, memoryview)):
                                b = bytes(chunk)
                            elif isinstance(chunk, str):
                                b = chunk.encode("utf-8", errors="replace")
                            else:
                                b = None  # pathsend 等控制消息不累计
                            if b is not None:
                                captured.append(b)
                                captured_len += len(b)
                        yield chunk
                finally:
                    # 客户端中断/异常时也保证有一条日志（此时响应体为部分内容）
                    if is_sse:
                        self._log_stream_end(request, captured, start, trace_id)
                    else:
                        self._log_response(request, response, captured, start, trace_id)

            response.body_iterator = tee()
        else:
            # 退化分支：响应体拿不到，只记状态与耗时
            self._log_response(request, response, captured, start, trace_id)

        return response


class TraceIDMiddleware(BaseHTTPMiddleware):
    """为每个请求注入 trace_id：X-Trace-ID 优先，未传则生成 uuid7。

    通过 contextvar 向后续链路（handler / 日志 Filter / 队列任务构造）传递，
    响应头回显 X-Trace-ID。直接替换旧的 X-Request-ID 体系。
    """

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        trace_id = request.headers.get("X-Trace-ID") or str(uuid7())
        token = set_trace_id(trace_id)
        try:
            response = await call_next(request)
        finally:
            reset_trace_id(token)
        response.headers["X-Trace-ID"] = trace_id
        return response


def register_middleware(app: FastAPI) -> None:
    """注册全局中间件。顺序：后注册的先执行。"""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # 后注册者先执行（Starlette 洋葱模型）。
    # TraceIDMiddleware 为最外层：先 set_trace_id → call_next → AccessLog → handler，
    # AccessLog._log 时 trace_id 仍在上下文，finally 后 reset + 回显 X-Trace-ID。
    app.add_middleware(AccessLogMiddleware)
    app.add_middleware(TraceIDMiddleware)
