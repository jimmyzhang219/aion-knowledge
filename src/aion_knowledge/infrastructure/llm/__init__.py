"""LLM 基础设施模块 — 基于 LangChain 的 LLM 统一调用层。"""

from aion_knowledge.infrastructure.llm.client import LLMClient
from aion_knowledge.infrastructure.llm.factory import (
    create_llm,
    get_llm_client_for_module,
    get_vlm_client,
)

__all__ = [
    "create_llm",
    "get_llm_client_for_module",
    "get_vlm_client",
    "LLMClient",
]
