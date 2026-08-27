"""配置管理测试。"""

import os

from aion_knowledge.common.config import EmbeddingProvider, LLMProvider, LogLevel, Settings


def test_settings_defaults(monkeypatch) -> None:
    for key in list(os.environ):
        if key.startswith("AION_"):
            monkeypatch.delenv(key, raising=False)
    settings = Settings(_env_file=None)
    assert settings.log_level == LogLevel.INFO
    assert settings.embedding_dimensions == 1536
    assert settings.db_url.startswith("postgresql+asyncpg")


def test_settings_env_override(monkeypatch) -> None:
    monkeypatch.setenv("AION_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("AION_EMBEDDING_DIMENSIONS", "1024")
    settings = Settings()
    assert settings.log_level == LogLevel.DEBUG
    assert settings.embedding_dimensions == 1024


def test_settings_embedding_provider_enum() -> None:
    settings = Settings(embedding_provider="ollama", _env_file=None)
    assert settings.embedding_provider == EmbeddingProvider.OLLAMA


def test_settings_llm_provider_alicloud() -> None:
    settings = Settings(llm_provider="alicloud", _env_file=None)
    assert settings.llm_provider == LLMProvider.ALI_CLOUD


def test_settings_llm_thinking() -> None:
    settings = Settings(
        llm_enable_thinking=True,
        llm_thinking_budget=81920,
        llm_max_completion_tokens=128000,
        _env_file=None,
    )
    assert settings.llm_enable_thinking is True
    assert settings.llm_thinking_budget == 81920
    assert settings.llm_max_completion_tokens == 128000


# ── 文档解析配置 ──────────────────────────────────────────────────────────


def test_parser_concurrency_defaults_are_conservative() -> None:
    s = Settings(_env_file=None)
    assert s.markitdown_max_workers == 1
    assert s.pdf_render_max_workers == 1
    assert s.pdf_render_dpi == 200
    assert s.pdf_jpeg_quality == 85


def test_parser_concurrency_env_override(monkeypatch) -> None:
    monkeypatch.setenv("AION_MARKITDOWN_MAX_WORKERS", "3")
    monkeypatch.setenv("AION_PDF_RENDER_MAX_WORKERS", "2")
    monkeypatch.setenv("AION_PDF_RENDER_DPI", "180")
    monkeypatch.setenv("AION_PDF_JPEG_QUALITY", "90")
    s = Settings()
    assert s.markitdown_max_workers == 3
    assert s.pdf_render_max_workers == 2
    assert s.pdf_render_dpi == 180
    assert s.pdf_jpeg_quality == 90


def test_pdf_force_scanned_default() -> None:
    s = Settings(_env_file=None)
    assert s.pdf_force_scanned is False


def test_pdf_force_scanned_env(monkeypatch) -> None:
    monkeypatch.setenv("AION_PDF_FORCE_SCANNED", "true")
    s = Settings()
    assert s.pdf_force_scanned is True
