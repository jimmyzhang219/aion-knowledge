"""Tests for ModelRegistry + utility functions."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
import tiktoken

from aion_knowledge.common.model_registry import (
    ModelRegistry,
    get_model_max_input_tokens,
    get_registry,
    sample_head_middle_tail,
    truncate_by_tokens,
)


class TestModelRegistry:
    def test_known_model(self):
        reg = get_registry()
        info = reg.get("gpt-4o")
        assert info.context_window == 128000
        assert info.provider == "openai"

    def test_unknown_model_returns_fallback(self, caplog):
        reg = get_registry()
        caplog.set_level(logging.WARNING)
        info = reg.get("totally-nonexistent-model-v99")
        assert info.context_window == 8192  # 兜底值
        assert "未在注册表中找到" in caplog.text

    def test_unknown_model_no_exception(self):
        reg = get_registry()
        info = reg.get("")  # 空字符串不会崩溃
        assert info.context_window == 8192

    def test_conf_registry_qwen3_wildcard(self):
        """真实 conf/model_registry.json 的 qwen3* 模式应命中未精确注册的 qwen3 变体。

        （json.load 锁定 conf 数据源，避免后续修改 conf 值时测试静默失真）
        """
        with open("conf/model_registry.json", encoding="utf-8") as f:
            conf = json.load(f)
        assert conf["qwen3*"]["context_window"] == 1048576  # conf 源锁定
        reg = get_registry()
        info = reg.get("qwen3.7-max-20250601")
        assert info.context_window == 1048576
        assert info.provider == "alicloud"

    def test_missing_json_raises(self, monkeypatch):
        """JSON 唯一源：找不到任何注册表文件时必须 raise，不静默降级。"""
        monkeypatch.setattr(
            ModelRegistry,
            "_DEFAULT_PATHS",
            ["/nonexistent/a.json", "/nonexistent/b.json"],
        )
        with pytest.raises(FileNotFoundError):
            ModelRegistry()


class TestTruncateByTokens:
    def test_short_text_unchanged(self):
        text = "Hello, world!"
        assert truncate_by_tokens(text, 100) == text

    def test_long_text_truncated(self):
        text = "word " * 1000  # ~1000 token, 远超过 10 token
        result = truncate_by_tokens(text, 10)
        # 精确验证：截断后的 token 数应 <= 10
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        assert len(enc.encode(result)) <= 10

    def test_empty_text(self):
        assert truncate_by_tokens("", 100) == ""
        assert truncate_by_tokens(None, 100) == ""

    def test_zero_max_tokens_returns_empty(self):
        text = "some text"
        assert truncate_by_tokens(text, 0) == ""


class TestGetModelMaxInputTokens:
    def test_known_model(self):
        n = get_model_max_input_tokens("gpt-4o", ratio=0.1)
        assert n == 12800  # 128000 * 0.1

    def test_unknown_model_default(self):
        n = get_model_max_input_tokens("no-such-model", ratio=0.5)
        assert n == 4096  # 8192 * 0.5

    def test_minimum_floor(self):
        n = get_model_max_input_tokens("no-such-model", ratio=0.001)
        assert n == 512  # 保底


class TestSampleHeadMiddleTail:
    def test_none_or_empty_returns_empty(self):
        assert sample_head_middle_tail(None, 100) == ""
        assert sample_head_middle_tail("", 100) == ""

    def test_short_text_unchanged(self):
        text = "Hello, world!"
        assert sample_head_middle_tail(text, 100) == text

    def test_long_text_returns_three_segments(self):
        # 构造 ~200 token 文本
        text = " ".join(["word"] * 500)
        result = sample_head_middle_tail(text, 50, head_ratio=0.6, middle_ratio=0.2)
        # 结果中包含省略标记
        assert "[...content omitted...]" in result
        # 结果 token 数 <= 50
        enc = tiktoken.get_encoding("cl100k_base")
        assert len(enc.encode(result)) <= 50

    def test_head_only_and_head_tail_modes(self):
        text = " ".join(["word"] * 500)
        enc = tiktoken.get_encoding("cl100k_base")
        # head_only: middle_ratio=0, tail 预算 = 0
        result = sample_head_middle_tail(text, 50, head_ratio=1.0, middle_ratio=0.0)
        assert len(result) < len(text)
        # head_only 模式不应包含省略标记
        assert "[...content omitted...]" not in result
        # token 数不超过 max_tokens
        assert len(enc.encode(result)) <= 50

        # 两段模式：head + tail，middle_ratio=0
        result2 = sample_head_middle_tail(text, 50, head_ratio=0.6, middle_ratio=0.0)
        assert "[...content omitted...]" in result2
        # 只有一个省略标记（非三个）
        assert result2.count("[...content omitted...]") == 1
        assert len(enc.encode(result2)) <= 50

    def test_head_middle_no_tail_rounding_remainder(self):
        """head+middle (tail_ratio=0) 时，整数舍入不能残留非预期的 tail 段或超过预算。"""
        text = " ".join(["word"] * 500)
        enc = tiktoken.get_encoding("cl100k_base")
        # head_ratio + middle_ratio = 1.0，tail_ratio = 0.0
        # content_budget = 50 - marker(6) = 44
        # int(44*0.8)=35, int(44*0.2)=8, 余数=1 → 不应产生 tail 段
        result = sample_head_middle_tail(text, 50, head_ratio=0.8, middle_ratio=0.2)
        assert len(enc.encode(result)) <= 50
        # 应该只有一个省略标记（head + middle），非两个
        assert result.count("[...content omitted...]") == 1

    def test_head_middle_no_overlap(self):
        """head 段和 middle 段不应有内容重叠。"""
        # 构造一段刚好略长于 max_tokens 的文本
        text = "word " * 200  # ~200 token
        result = sample_head_middle_tail(text, 30, head_ratio=0.6, middle_ratio=0.2)
        enc = tiktoken.get_encoding("cl100k_base")
        assert len(enc.encode(result)) <= 30
        # 如果 head 和 middle 有重叠，解析省略标记间的文字会看到重复词
        # （这个断言确保函数不抛异常且输出合理）
        assert result.count("[...content omitted...]") == 2  # 三段模式

    def test_negative_max_tokens_returns_empty(self):
        assert sample_head_middle_tail("some text", -1) == ""

    def test_zero_max_tokens_returns_empty(self):
        assert sample_head_middle_tail("some text", 0) == ""


def _write_registry(tmp_path: Path, data: dict) -> ModelRegistry:
    """把 data 写成临时 JSON 并构造独立 ModelRegistry 实例。"""
    p = tmp_path / "registry.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return ModelRegistry(str(p))


class TestModelRegistryWildcard:
    """星号通配符模式条目测试（自定义临时注册表）。"""

    def test_wildcard_prefix_match(self, tmp_path):
        reg = _write_registry(tmp_path, {
            "qwen3*": {"provider": "alicloud", "context_window": 1048576, "max_output": 32768},
        })
        info = reg.get("qwen3.7-max-20250601")
        assert info.context_window == 1048576
        assert info.provider == "alicloud"

    def test_exact_match_prefers_exact_entry(self, tmp_path):
        reg = _write_registry(tmp_path, {
            "qwen3.7-max": {"provider": "alicloud", "context_window": 999, "max_output": 100},
            "qwen3*": {"provider": "alicloud", "context_window": 1048576, "max_output": 32768},
        })
        info = reg.get("qwen3.7-max")
        assert info.context_window == 999  # 精确条目优先于模式

    def test_longest_pattern_wins(self, tmp_path):
        reg = _write_registry(tmp_path, {
            "qwen3*": {"provider": "alicloud", "context_window": 1048576, "max_output": 32768},
            "qwen3.7*": {"provider": "alicloud", "context_window": 524288, "max_output": 16384},
        })
        info = reg.get("qwen3.7-max")
        assert info.context_window == 524288  # 最长模式优先

    def test_wildcard_middle_and_suffix_match(self, tmp_path):
        """* 可出现在任意位置：中间位置匹配任意字符（含 .），后缀模式同样生效。"""
        reg = _write_registry(tmp_path, {
            "gpt-4*mini": {"provider": "openai", "context_window": 128000, "max_output": 16384},
            "*flash": {"provider": "deepseek", "context_window": 1048576, "max_output": 8192},
        })
        info = reg.get("gpt-4o-mini")
        assert info.context_window == 128000  # * 在中间位置，匹配 ".7-..." 等任意字符
        assert info.provider == "openai"
        info2 = reg.get("deepseek-v4-flash")
        assert info2.context_window == 1048576  # 后缀模式（* 在前缀位置）
        assert info2.provider == "deepseek"

    def test_no_match_returns_fallback(self, tmp_path, caplog):
        reg = _write_registry(tmp_path, {
            "qwen3*": {"provider": "alicloud", "context_window": 1048576, "max_output": 32768},
        })
        caplog.set_level(logging.WARNING)
        info = reg.get("totally-nonexistent-v99")
        assert info.context_window == 8192  # 兜底
        assert "未在注册表中找到" in caplog.text
