"""LLM provider 实现测试（特定模型/平台的个性化适配）。

分层：
  - ReasoningOpenAI：通用基类（提取 reasoning_content + 接受 thinking_budget 默认忽略）
  - ChatMaaS：阿里云百炼（DashScope）统一接口（thinking_budget → extra_body）
  - ChatQwen：Qwen 特有（video_url 多模态，继承 ChatMaaS）
"""

from __future__ import annotations

from aion_knowledge.infrastructure.llm.providers import (
    ChatMaaS,
    ChatQwen,
    ReasoningOpenAI,
)


class TestReasoningOpenAI:
    """通用基类：接受 thinking_budget 参数但默认不注入（格式由子类决定）。"""

    def test_ignores_thinking_budget(self):
        """非 MaaS 路径收到 thinking_budget 时不报错、不注入。"""
        m = ReasoningOpenAI(model="gpt-4o", api_key="sk-x", thinking_budget=32000)
        extra_body = getattr(m, "extra_body", None) or {}
        assert "thinking_budget" not in extra_body


class TestChatMaaS:
    """阿里云百炼（DashScope）统一接口：thinking_budget → extra_body。"""

    def test_translates_thinking_budget_to_extra_body(self):
        m = ChatMaaS(model="glm-5.2", api_key="sk-x", thinking_budget=32000)
        assert m.extra_body == {"enable_thinking": True, "thinking_budget": 32000}

    def test_zero_budget_not_injected(self):
        m = ChatMaaS(model="glm-5.2", api_key="sk-x", thinking_budget=0)
        extra_body = getattr(m, "extra_body", None) or {}
        assert "thinking_budget" not in extra_body
        assert "enable_thinking" not in extra_body


class TestChatQwen:
    """ChatQwen 继承 ChatMaaS，自动获得 thinking_budget 翻译。"""

    def test_inherits_maas_thinking_budget(self):
        m = ChatQwen(model="qwen3.5-plus", api_key="sk-x", thinking_budget=32000)
        assert m.extra_body == {"enable_thinking": True, "thinking_budget": 32000}

    def test_default_reasoning_effort_kept(self):
        m = ChatQwen(model="qwen3.5-plus", api_key="sk-x")
        assert m.reasoning_effort == "high"
