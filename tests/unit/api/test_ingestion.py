"""Ingestion API 路由测试。"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from aion_knowledge.api import create_app
from aion_knowledge.ingestion.kb_guard import KnowledgeBaseNotFoundError

_KB_ID = "00000000-0000-0000-0000-000000000001"


@pytest.fixture
def client() -> TestClient:
    app = create_app()
    return TestClient(app)


@pytest.fixture
def mock_strategy():
    strategy = AsyncMock()
    strategy.execute = AsyncMock(return_value={"status": "queued"})
    return strategy


def test_upload_no_split_preserved(client: TestClient, mock_strategy) -> None:
    """验证 no_split 保持原值传递。"""
    with patch("aion_knowledge.api.routes.regular.get_strategy",
               return_value=mock_strategy):
        client.post(
            f"/api/v1/knowledge/{_KB_ID}/documents/upload",
            data={"chunk_strategy": "no_split"},
            files={"file": ("test.md", b"# hello", "text/markdown")},
        )
    _, kwargs = mock_strategy.execute.call_args
    assert kwargs["chunk_strategy"] == "no_split"


def test_upload_auto_normalized(client: TestClient, mock_strategy) -> None:
    """验证非 no_split 值被归一化为 auto。"""
    with patch("aion_knowledge.api.routes.regular.get_strategy",
               return_value=mock_strategy):
        client.post(
            f"/api/v1/knowledge/{_KB_ID}/documents/upload",
            data={"chunk_strategy": "heading"},
            files={"file": ("test.md", b"# hello", "text/markdown")},
        )
    _, kwargs = mock_strategy.execute.call_args
    assert kwargs["chunk_strategy"] == "auto"


def test_upload_default_is_auto(client: TestClient, mock_strategy) -> None:
    """验证不传 chunk_strategy 时默认 auto。"""
    with patch("aion_knowledge.api.routes.regular.get_strategy",
               return_value=mock_strategy):
        client.post(
            f"/api/v1/knowledge/{_KB_ID}/documents/upload",
            files={"file": ("test.md", b"# hello", "text/markdown")},
        )
    _, kwargs = mock_strategy.execute.call_args
    assert kwargs["chunk_strategy"] == "auto"


def test_upload_kb_not_found_404(client: TestClient, mock_strategy) -> None:
    """上传文档：kb_id 不存在 → 404。"""
    mock_strategy.execute = AsyncMock(
        side_effect=KnowledgeBaseNotFoundError(f"KnowledgeBase {_KB_ID} not found")
    )
    with patch("aion_knowledge.api.routes.regular.get_strategy",
               return_value=mock_strategy):
        resp = client.post(
            f"/api/v1/knowledge/{_KB_ID}/documents/upload",
            files={"file": ("test.md", b"# hello", "text/markdown")},
        )
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"]


def test_faq_import_kb_not_found_404(client: TestClient, mock_strategy) -> None:
    """FAQ 导入：kb_id 不存在 → 404。"""
    mock_strategy.execute = AsyncMock(
        side_effect=KnowledgeBaseNotFoundError(f"KnowledgeBase {_KB_ID} not found")
    )
    with patch("aion_knowledge.api.routes.faq.get_strategy",
               return_value=mock_strategy):
        resp = client.post(
            f"/api/v1/knowledge/{_KB_ID}/faq/import",
            files={"file": ("faq.csv", b"question,answer\nq,a\n", "text/csv")},
        )
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"]


def test_api_direct_kb_not_found_404(client: TestClient, mock_strategy) -> None:
    """API 直投：kb_id 不存在 → 404。"""
    mock_strategy.execute = AsyncMock(
        side_effect=KnowledgeBaseNotFoundError(f"KnowledgeBase {_KB_ID} not found")
    )
    with patch("aion_knowledge.api.routes.api_direct.get_strategy",
               return_value=mock_strategy):
        resp = client.post(
            f"/api/v1/knowledge/{_KB_ID}/api-direct",
            json={"content": "hello", "doc_name": "t.txt"},
        )
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"]
