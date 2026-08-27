"""阿里云百炼（DashScope）统一接口的 LLM 实现。

阿里云百炼托管了 Qwen / DeepSeek / GLM / Kimi 等多个模型，它们共用同一套
OpenAI 兼容接口（DashScope），思考参数通过非标准字段 ``extra_body`` 传递：

- ``enable_thinking``：开启/关闭思考（hybrid 模型）
- ``thinking_budget``：思考 token 上限（Qwen3.x / GLM / Kimi 支持；
  DeepSeek-v4-pro 不支持，但服务端会静默忽略，不报错）

此类负责请求侧的参数翻译；响应侧的 ``reasoning_content`` 提取由基类
``ReasoningOpenAI`` 完成。
"""

from __future__ import annotations

from typing import Any

from .reasoning import ReasoningOpenAI


class ChatMaaS(ReasoningOpenAI):
    """阿里云百炼（DashScope）统一接口实现。

    将通用的 ``thinking_budget`` 参数翻译为 DashScope 的 ``extra_body`` 格式。
    """

    def __init__(self, **kwargs: Any) -> None:
        budget = kwargs.pop("thinking_budget", 0)
        if budget > 0:
            extra_body = dict(kwargs.get("extra_body") or {})
            extra_body.setdefault("enable_thinking", True)
            extra_body["thinking_budget"] = budget
            kwargs["extra_body"] = extra_body
        super().__init__(**kwargs)
