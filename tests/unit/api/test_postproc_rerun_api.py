"""后处理重跑 API 路由测试。"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from aion_knowledge.api import create_app
from aion_knowledge.infrastructure.models import PostProcConfig, PostProcTask

_KB = "00000000-0000-0000-0000-000000000001"
_DOC = "00000000-0000-0000-0000-0000000000d1"

_URL = f"/api/v1/knowledge/{_KB}/documents/{_DOC}/postproc/run"


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def _task() -> PostProcTask:
    return PostProcTask(
        document_id=_DOC, kb_id=_KB, doc_name="t.md", chunk_count=2,
        modules=["raptor"], postproc_config=PostProcConfig(),
    )


def test_rerun_ok(client):
    """201：入队成功返回 queued。"""
    with patch(
        "aion_knowledge.api.routes.regular.enqueue_postproc_rerun",
        new=AsyncMock(return_value=_task()),
    ) as mock_enqueue:
        resp = client.post(_URL, json={"modules": ["raptor"]})
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "queued"
    assert body["document_id"] == _DOC
    assert body["modules"] == ["raptor"]
    mock_enqueue.assert_awaited_once_with(_KB, _DOC, ["raptor"])


def test_rerun_module_error_400(client):
    """400：模块校验失败透传 detail。"""
    from aion_knowledge.ingestion.postproc_rerun import ModuleValidationError

    async def _raise(*args, **kwargs):
        raise ModuleValidationError("模块未启用（.env 门控或出厂硬控未开）: ['raptor']")

    with patch("aion_knowledge.api.routes.regular.enqueue_postproc_rerun",
               new=_raise):
        resp = client.post(_URL, json={"modules": ["raptor"]})
    assert resp.status_code == 400
    assert "未启用" in resp.json()["detail"]


def test_rerun_not_found_404(client):
    """404：文档不存在透传 detail。"""
    from aion_knowledge.ingestion.postproc_rerun import DocumentNotFoundError

    async def _raise(*args, **kwargs):
        raise DocumentNotFoundError(f"Document {_DOC} not found in kb {_KB}")

    with patch("aion_knowledge.api.routes.regular.enqueue_postproc_rerun",
               new=_raise):
        resp = client.post(_URL, json={"modules": ["raptor"]})
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"]


def test_rerun_missing_modules_422(client):
    """422：缺 modules 字段由 Pydantic 拒绝。"""
    resp = client.post(_URL, json={})
    assert resp.status_code == 422
