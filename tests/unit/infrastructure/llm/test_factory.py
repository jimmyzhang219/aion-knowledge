"""LLM 工厂测试。"""

from __future__ import annotations

import sys
from unittest.mock import Mock

from aion_knowledge.common.config import LLMProvider
from aion_knowledge.infrastructure.llm.client import LLMClient
from aion_knowledge.infrastructure.llm.factory import (
    _get_llm_class,
    create_llm,
    get_llm_client_for_module,
    get_vlm_client,
)
from aion_knowledge.infrastructure.llm.providers import ChatMaaS, ChatQwen, ReasoningOpenAI

# ── Ollama 模拟 ──

_ollama_mock = Mock()
_ollama_mock.ChatOllama = Mock()


def _mock_ollama_module(monkeypatch):
    """将 langchain_ollama 替换为模拟模块（避免安装依赖）。"""
    monkeypatch.setitem(sys.modules, "langchain_ollama", _ollama_mock)


class TestGetLLMClass:
    """_get_llm_class 按 provider + model 路由。"""

    def test_qwen_returns_chat_qwen(self):
        assert _get_llm_class("qwen3.5-plus", "alicloud") is ChatQwen

    def test_non_qwen_on_alicloud_returns_chat_maas(self):
        # 走阿里云百炼的 glm/deepseek/kimi 等统一路由到 ChatMaaS
        assert _get_llm_class("glm-5.2", "alicloud") is ChatMaaS
        assert _get_llm_class("deepseek-v4-pro", "alicloud") is ChatMaaS

    def test_non_alicloud_returns_reasoning_openai(self):
        # 走各自官方 API 的模型：通用 OpenAI 兼容
        assert _get_llm_class("gpt-4o", "openai") is ReasoningOpenAI
        assert _get_llm_class("deepseek-v4", "deepseek") is ReasoningOpenAI
        assert _get_llm_class("claude-opus", "anthropic") is ReasoningOpenAI


class TestCreateLLM:
    """create_llm 工厂函数测试。"""

    def test_default_provider_returns_llm_client(self, monkeypatch):
        monkeypatch.setattr("aion_knowledge.infrastructure.llm.factory.settings.llm_model", "gpt-4o")
        monkeypatch.setattr("aion_knowledge.infrastructure.llm.factory.settings.llm_api_key", "sk-test")
        monkeypatch.setattr("aion_knowledge.infrastructure.llm.factory.settings.llm_provider", LLMProvider.OPENAI)
        monkeypatch.setattr("aion_knowledge.infrastructure.llm.factory.settings.llm_request_timeout", 60)
        monkeypatch.setattr("aion_knowledge.infrastructure.llm.factory.settings.llm_enable_thinking", False)
        client = create_llm()
        assert isinstance(client, LLMClient)

    def test_provider_ollama_returns_llm_client(self, monkeypatch):
        _mock_ollama_module(monkeypatch)
        monkeypatch.setattr("aion_knowledge.infrastructure.llm.factory.settings.llm_model", "llama3")
        monkeypatch.setattr("aion_knowledge.infrastructure.llm.factory.settings.llm_base_url", "http://localhost:11434")
        client = create_llm(provider="ollama")
        assert isinstance(client, LLMClient)

    def test_vlm_client_uses_vlm_config(self, monkeypatch):
        monkeypatch.setattr("aion_knowledge.infrastructure.llm.factory.settings.vlm_model", "qwen-vl-plus")
        monkeypatch.setattr("aion_knowledge.infrastructure.llm.factory.settings.vlm_api_key", "sk-vlm")
        monkeypatch.setattr("aion_knowledge.infrastructure.llm.factory.settings.vlm_provider", "alicloud")
        monkeypatch.setattr("aion_knowledge.infrastructure.llm.factory.settings.llm_api_key", "sk-default")
        monkeypatch.setattr("aion_knowledge.infrastructure.llm.factory.settings.llm_request_timeout", 60)
        monkeypatch.setattr("aion_knowledge.infrastructure.llm.factory.settings.llm_enable_thinking", False)
        monkeypatch.setattr("aion_knowledge.infrastructure.llm.factory.settings.llm_model", "gpt-4o")
        monkeypatch.setattr("aion_knowledge.infrastructure.llm.factory.settings.llm_provider", LLMProvider.OPENAI)
        client = get_vlm_client()
        assert isinstance(client, LLMClient)

    def test_per_module_override(self, monkeypatch):
        _mock_ollama_module(monkeypatch)
        monkeypatch.setattr(
            "aion_knowledge.infrastructure.llm.factory.settings.postproc_module_llm",
            {"summarizer": {"llm_provider": "ollama", "llm_model": "llama3"}},
        )
        monkeypatch.setattr("aion_knowledge.infrastructure.llm.factory.settings.llm_base_url", "")
        client = get_llm_client_for_module("summarizer")
        assert isinstance(client, LLMClient)

    def test_unknown_module_uses_global(self, monkeypatch):
        monkeypatch.setattr("aion_knowledge.infrastructure.llm.factory.settings.llm_model", "gpt-4o")
        monkeypatch.setattr("aion_knowledge.infrastructure.llm.factory.settings.llm_api_key", "sk-test")
        monkeypatch.setattr("aion_knowledge.infrastructure.llm.factory.settings.llm_provider", LLMProvider.OPENAI)
        monkeypatch.setattr("aion_knowledge.infrastructure.llm.factory.settings.llm_request_timeout", 60)
        monkeypatch.setattr("aion_knowledge.infrastructure.llm.factory.settings.llm_enable_thinking", False)
        monkeypatch.setattr(
            "aion_knowledge.infrastructure.llm.factory.settings.postproc_module_llm",
            {},
        )
        client = get_llm_client_for_module("nonexistent")
        assert isinstance(client, LLMClient)

    def test_thinking_budget_passed_via_extra_body(self, monkeypatch):
        """llm_thinking_budget>0 时通过 extra_body 透传给 provider。

        阿里云 DashScope 的 enable_thinking / thinking_budget 非 OpenAI 标准
        参数，须走 extra_body。"""
        monkeypatch.setattr("aion_knowledge.infrastructure.llm.factory.settings.llm_model", "qwen3.5-plus")
        monkeypatch.setattr("aion_knowledge.infrastructure.llm.factory.settings.llm_api_key", "sk-test")
        monkeypatch.setattr("aion_knowledge.infrastructure.llm.factory.settings.llm_provider", LLMProvider.ALI_CLOUD)
        monkeypatch.setattr("aion_knowledge.infrastructure.llm.factory.settings.llm_request_timeout", 60)
        monkeypatch.setattr("aion_knowledge.infrastructure.llm.factory.settings.llm_enable_thinking", True)
        monkeypatch.setattr("aion_knowledge.infrastructure.llm.factory.settings.llm_thinking_budget", 32000)
        monkeypatch.setattr("aion_knowledge.infrastructure.llm.factory.settings.llm_max_completion_tokens", 64000)
        client = create_llm()
        extra_body = getattr(client._model, "extra_body", None) or {}
        assert extra_body.get("thinking_budget") == 32000
        assert extra_body.get("enable_thinking") is True

    def test_vlm_client_never_receives_thinking_params(self, monkeypatch):
        """VLM 客户端即使全局开启 thinking 也不注入思考参数。

        get_vlm_client 传 reasoning_effort=None 显式禁用思考（VL 模型不支持
        reasoning），thinking_budget / enable_thinking 不得注入，否则
        DashScope 对 VL 模型返回 400（invalid_parameter_error）。"""
        monkeypatch.setattr("aion_knowledge.infrastructure.llm.factory.settings.vlm_model", "qwen3-vl-plus")
        monkeypatch.setattr("aion_knowledge.infrastructure.llm.factory.settings.vlm_api_key", "sk-vlm")
        monkeypatch.setattr("aion_knowledge.infrastructure.llm.factory.settings.vlm_provider", "alicloud")
        monkeypatch.setattr("aion_knowledge.infrastructure.llm.factory.settings.llm_api_key", "sk-test")
        monkeypatch.setattr("aion_knowledge.infrastructure.llm.factory.settings.llm_request_timeout", 60)
        monkeypatch.setattr("aion_knowledge.infrastructure.llm.factory.settings.llm_enable_thinking", True)
        monkeypatch.setattr("aion_knowledge.infrastructure.llm.factory.settings.llm_thinking_budget", 32000)
        monkeypatch.setattr("aion_knowledge.infrastructure.llm.factory.settings.llm_max_completion_tokens", 64000)
        client = get_vlm_client()
        extra_body = getattr(client._model, "extra_body", None) or {}
        assert "thinking_budget" not in extra_body
        assert "enable_thinking" not in extra_body

    def test_thinking_budget_zero_not_passed(self, monkeypatch):
        """llm_thinking_budget=0 时不注入 thinking_budget（保持默认不限制）。"""
        monkeypatch.setattr("aion_knowledge.infrastructure.llm.factory.settings.llm_model", "qwen3.5-plus")
        monkeypatch.setattr("aion_knowledge.infrastructure.llm.factory.settings.llm_api_key", "sk-test")
        monkeypatch.setattr("aion_knowledge.infrastructure.llm.factory.settings.llm_provider", LLMProvider.ALI_CLOUD)
        monkeypatch.setattr("aion_knowledge.infrastructure.llm.factory.settings.llm_request_timeout", 60)
        monkeypatch.setattr("aion_knowledge.infrastructure.llm.factory.settings.llm_enable_thinking", True)
        monkeypatch.setattr("aion_knowledge.infrastructure.llm.factory.settings.llm_thinking_budget", 0)
        monkeypatch.setattr("aion_knowledge.infrastructure.llm.factory.settings.llm_max_completion_tokens", 64000)
        client = create_llm()
        extra_body = getattr(client._model, "extra_body", None) or {}
        assert "thinking_budget" not in extra_body
