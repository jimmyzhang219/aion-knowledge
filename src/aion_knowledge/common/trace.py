"""请求链路追踪上下文 — trace_id 的生成与注入。

实现要点：
- ``trace_id_var`` 为 contextvar，请求入口（中间件 / MCP 工具）set，
  队列 worker 消费消息时 set（worker 是独立 asyncio task，不继承请求上下文）。
- ``get_trace_id()`` 未设置时生成 uuid7，保证任何路径都有值。
- 日志注入由 ``logger.py`` 的 LogRecord 工厂负责（在每条日志记录创建时从
  ``trace_id_var`` 读取并注入 ``record.trace_id``，无需 Filter）。
"""

from __future__ import annotations

from contextvars import ContextVar, Token

from aion_knowledge.common.uuid7 import uuid7

trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")


def get_trace_id() -> str:
    """读取当前上下文 trace_id；未设置则生成 uuid7（幂等：已设置不重复生成）。"""
    current = trace_id_var.get()
    if current:
        return current
    generated = str(uuid7())
    trace_id_var.set(generated)
    return generated


def set_trace_id(value: str) -> Token[str]:
    """注入 trace_id，返回 reset token。"""
    return trace_id_var.set(value)


def reset_trace_id(token: Token[str]) -> None:
    """恢复之前的 trace_id 状态。"""
    trace_id_var.reset(token)
