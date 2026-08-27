"""LLM 工厂 — 创建 LLMClient 实例。

整合原 src/aion_knowledge/llm/factory.py（LangChain 模型选择）
与 src/aion_knowledge/pipeline/llm/dispatcher.py（provider 路由）。
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from aion_knowledge.common.config import settings
from aion_knowledge.infrastructure.llm.client import LLMClient
from aion_knowledge.infrastructure.llm.providers import ChatMaaS, ChatQwen, ReasoningOpenAI

# ── Qwen 默认参数 ──
_QWEN_DEFAULTS: dict[str, Any] = {
    "reasoning_effort": "high",
}


def _get_llm_class(model: str, provider: str) -> type[BaseChatModel]:
    """根据模型名称和服务提供商返回对应的实现类。

    - ``qwen*``：``ChatQwen``（video_url 多模态格式）
    - 阿里云百炼（``alicloud``）其他模型：``ChatMaaS``（DashScope 统一接口）
    - 其他 provider：``ReasoningOpenAI``（通用 OpenAI 兼容）
    """
    if model.startswith("qwen"):
        return ChatQwen
    if provider == "alicloud":
        return ChatMaaS
    return ReasoningOpenAI


def create_llm(
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    provider: str | None = None,
    **extra_kwargs: Any,
) -> LLMClient:
    """创建 LLMClient 实例。

    Args:
        model: 模型名称。默认 settings.llm_model。
        api_key: API Key。默认 settings.llm_api_key。
        base_url: API 地址。默认 settings.llm_base_url。
        provider: 服务提供商。默认 settings.llm_provider。
                  可选: "openai", "alicloud", "zhipu", "deepseek", "ollama" 等。
        **extra_kwargs: 透传给 LLM 构造函数。

    Returns:
        LLMClient 实例。
    """
    model_name = model or settings.llm_model
    resolved_provider = (provider or settings.llm_provider.value).lower()

    if resolved_provider == "ollama":
        from langchain_ollama import ChatOllama  # noqa: PLC0415

        langchain_model: BaseChatModel = ChatOllama(
            model=model_name,
            base_url=base_url or settings.llm_base_url or "http://localhost:11434",
        )
    else:
        cls = _get_llm_class(model_name, resolved_provider)

        kwargs: dict[str, Any] = {
            "model": model_name,
            "api_key": api_key or settings.llm_api_key,
            "base_url": base_url or settings.llm_base_url or None,
            "timeout": settings.llm_request_timeout,
        }

        # 全局 thinking 开关；调用方显式传 reasoning_effort（如 VLM 视觉模型
        # 传 None 禁用思考）时不注入任何思考参数
        if settings.llm_enable_thinking and "reasoning_effort" not in extra_kwargs:
            kwargs["max_completion_tokens"] = settings.llm_max_completion_tokens or None
            kwargs["reasoning_effort"] = "high"
            # 通用的思考预算（token 上限）；具体参数格式由 provider 实现各自适配
            if settings.llm_thinking_budget > 0:
                kwargs["thinking_budget"] = settings.llm_thinking_budget

        if model_name.startswith("qwen"):
            for k, v in _QWEN_DEFAULTS.items():
                kwargs.setdefault(k, v)

        kwargs.update(extra_kwargs)
        langchain_model = cls(**kwargs)

    return LLMClient(langchain_model)


def get_vlm_client() -> LLMClient:
    """获取 VLM 多模态客户端（用于图片描述等视觉任务）。

    使用 AION_VLM_* 独立配置，修复了之前在 dispatcher.py 中重复定义导致
    AION_VLM_PROVIDER 被忽略的 bug。

    VL 模型通常不支持 reasoning/thinking，因此显式传入
    reasoning_effort=None 覆盖全局的 llm_enable_thinking 设置。
    """
    return create_llm(
        model=settings.vlm_model,
        api_key=settings.vlm_api_key or None,
        base_url=settings.vlm_base_url or None,
        provider=settings.vlm_provider,
        reasoning_effort=None,
    )


def get_llm_client_for_module(module_name: str) -> LLMClient:
    """获取指定后处理模块专用的 LLM 客户端。

    优先读取 per-module 配置覆盖，未覆盖则使用 create_llm() 默认。
    """
    overrides = settings.postproc_module_llm.get(module_name, {})
    return create_llm(
        model=overrides.get("llm_model"),
        api_key=overrides.get("llm_api_key"),
        base_url=overrides.get("llm_base_url"),
        provider=overrides.get("llm_provider"),
    )
