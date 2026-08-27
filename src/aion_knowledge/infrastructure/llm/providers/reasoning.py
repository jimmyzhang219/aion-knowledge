"""OpenAI 兼容模型的 Reasoning 支持。

LangChain 的 ``ChatOpenAI._create_chat_result`` 不会从 API 响应中提取
非标准字段（如 ``reasoning_content``），导致 thinking 内容丢失。

此模块提供 ``ReasoningOpenAI`` 类，在 ``_create_chat_result`` 中
从原始响应中捕获 ``reasoning_content`` 注入 ``additional_kwargs``，
使得下游可以正常消费 thinking 内容。
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI


class ReasoningOpenAI(ChatOpenAI):
    """ChatOpenAI 子类：从 API 响应中捕获 reasoning_content。

    适用于所有通过 OpenAI 兼容 API 返回 ``reasoning_content`` 字段的模型
    （如 Qwen、DeepSeek 等），无需为每个 provider 编写独立子类。

    作为通用基类，接受 ``thinking_budget`` 参数但默认不注入——具体参数格式
    （如阿里云 DashScope 的 ``extra_body``）由子类（``ChatMaaS`` 等）决定。
    """

    def __init__(self, **kwargs: Any) -> None:
        kwargs.pop("thinking_budget", None)
        super().__init__(**kwargs)

    def _create_chat_result(
        self,
        response: Any,
        generation_info: dict[str, Any] | None = None,
    ) -> Any:
        result = super()._create_chat_result(response, generation_info)

        # 从原始响应中提取 reasoning_content，注入 additional_kwargs
        response_dict = response if isinstance(response, dict) else response.model_dump()
        for i, choice in enumerate(response_dict.get("choices") or []):
            if i < len(result.generations):
                msg = result.generations[i].message
                if isinstance(msg, AIMessage):
                    rc = choice.get("message", {}).get("reasoning_content")
                    if rc:
                        msg.additional_kwargs["reasoning_content"] = rc

        return result
