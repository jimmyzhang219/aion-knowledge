"""enqueue_postproc_rerun 单元测试（全 mock，不触库）。"""
from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aion_knowledge.infrastructure.models import PostProcConfig
from aion_knowledge.ingestion.postproc_rerun import (
    DocumentNotFoundError,
    ModuleValidationError,
    enqueue_postproc_rerun,
    validate_modules,
)

_KB = "00000000-0000-0000-0000-000000000001"
_DOC = "00000000-0000-0000-0000-0000000000d1"


@pytest.fixture
def mock_db(monkeypatch):
    """mock get_session：async with 返回 AsyncMock session。"""
    fake_session = AsyncMock()
    cm = AsyncMock()
    cm.__aenter__.return_value = fake_session
    monkeypatch.setattr(
        "aion_knowledge.ingestion.postproc_rerun.get_session", lambda: cm
    )
    return fake_session


def _enable_all_env(monkeypatch):
    """确定性打开所有 .env 门控。"""
    from aion_knowledge.common.config import settings

    for name in ("keyword_extract", "question_gen", "summarizer", "raptor",
                 "graph_extract", "community", "disambiguation", "wiki"):
        monkeypatch.setattr(settings, f"postproc_{name}", True)


# ------------------------------------------------------------------ #
# validate_modules
# ------------------------------------------------------------------ #


class TestValidateModules:
    @pytest.mark.asyncio
    async def test_empty_rejected(self):
        with pytest.raises(ModuleValidationError) as excinfo:
            validate_modules([])
        assert "不能为空" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_unknown_rejected(self):
        with pytest.raises(ModuleValidationError) as excinfo:
            validate_modules(["nonexistent_module"])
        assert "未知模块" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_first_batch_rejected(self):
        with pytest.raises(ModuleValidationError) as excinfo:
            validate_modules(["text"])
        assert "首批模块" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_factory_gated_rejected(self, monkeypatch):
        """出厂硬控关闭时拒绝（monkeypatch 强制 enable_graph_extract=False）。"""
        from aion_knowledge.infrastructure.models import PostProcConfig

        _enable_all_env(monkeypatch)

        class _GatedConfig(PostProcConfig):
            enable_graph_extract: bool = False

        monkeypatch.setattr(
            "aion_knowledge.infrastructure.models.PostProcConfig", _GatedConfig
        )
        with pytest.raises(ModuleValidationError) as excinfo:
            validate_modules(["graph_extract"])
        assert "未启用" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_env_gated_rejected(self, monkeypatch):
        """.env 门控关闭：拒绝。"""
        _enable_all_env(monkeypatch)
        from aion_knowledge.common.config import settings
        monkeypatch.setattr(settings, "postproc_raptor", False)
        with pytest.raises(ModuleValidationError) as excinfo:
            validate_modules(["raptor"])
        assert "未启用" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_valid_passes(self, monkeypatch):
        _enable_all_env(monkeypatch)
        validate_modules(["raptor", "wiki"])  # 不抛异常


# ------------------------------------------------------------------ #
# enqueue_postproc_rerun
# ------------------------------------------------------------------ #


class TestEnqueuePostprocRerun:
    def _patch_queue(self, monkeypatch):
        queue = SimpleNamespace(put=AsyncMock())
        monkeypatch.setattr(
            "aion_knowledge.ingestion.postproc_rerun.postproc_queue", queue
        )
        return queue

    def _patch_doc_repo(self, monkeypatch, doc):
        async def _fake_get(session, doc_id):
            return doc

        monkeypatch.setattr(
            "aion_knowledge.ingestion.postproc_rerun.get_document_by_id", _fake_get
        )

    def _patch_count(self, monkeypatch, count: int):
        class _FakeChunkRepo:
            def __init__(self, session):
                pass

            async def count_by_document(self, document_id):
                return count

        monkeypatch.setattr(
            "aion_knowledge.ingestion.postproc_rerun.ChunkRepository", _FakeChunkRepo
        )

    @pytest.mark.asyncio
    async def test_invalid_document_id(self, monkeypatch):
        queue = self._patch_queue(monkeypatch)
        with pytest.raises(DocumentNotFoundError) as excinfo:
            await enqueue_postproc_rerun(_KB, "not-a-uuid", ["raptor"])
        assert "invalid document id" in str(excinfo.value)
        queue.put.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_invalid_kb_id(self, monkeypatch):
        queue = self._patch_queue(monkeypatch)
        with pytest.raises(DocumentNotFoundError) as excinfo:
            await enqueue_postproc_rerun("not-a-kb", _DOC, ["raptor"])
        assert "invalid kb id" in str(excinfo.value)
        queue.put.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_document_not_found(self, monkeypatch, mock_db):
        _enable_all_env(monkeypatch)
        queue = self._patch_queue(monkeypatch)
        self._patch_doc_repo(monkeypatch, None)
        with pytest.raises(DocumentNotFoundError):
            await enqueue_postproc_rerun(_KB, _DOC, ["raptor"])
        queue.put.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_kb_mismatch(self, monkeypatch, mock_db):
        _enable_all_env(monkeypatch)
        queue = self._patch_queue(monkeypatch)
        doc = SimpleNamespace(
            id=uuid.UUID(_DOC),
            kb_id=uuid.UUID("00000000-0000-0000-0000-000000000002"),
            doc_name="t.md",
            suffix="md",
        )
        self._patch_doc_repo(monkeypatch, doc)
        with pytest.raises(DocumentNotFoundError):
            await enqueue_postproc_rerun(_KB, _DOC, ["raptor"])
        queue.put.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_module_validation_error(self, monkeypatch, mock_db):
        queue = self._patch_queue(monkeypatch)
        doc = SimpleNamespace(
            id=uuid.UUID(_DOC),
            kb_id=uuid.UUID(_KB),
            doc_name="t.md",
            suffix="md",
        )
        self._patch_doc_repo(monkeypatch, doc)
        self._patch_count(monkeypatch, 3)
        with pytest.raises(ModuleValidationError):
            await enqueue_postproc_rerun(_KB, _DOC, ["text"])
        queue.put.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_success(self, monkeypatch, mock_db):
        _enable_all_env(monkeypatch)
        queue = self._patch_queue(monkeypatch)
        doc = SimpleNamespace(
            id=uuid.UUID(_DOC),
            kb_id=uuid.UUID(_KB),
            doc_name="t.md",
            suffix="md",
        )
        self._patch_doc_repo(monkeypatch, doc)
        self._patch_count(monkeypatch, 3)

        task = await enqueue_postproc_rerun(_KB, _DOC, ["raptor"])

        assert task.document_id == _DOC
        assert task.kb_id == _KB
        assert task.doc_name == "t.md"
        assert task.suffix == "md"
        assert task.chunk_count == 3
        assert task.modules == ["raptor"]
        assert task.postproc_config == PostProcConfig()
        queue.put.assert_awaited_once_with(task)

    @pytest.mark.asyncio
    async def test_kb_id_uppercase(self, monkeypatch, mock_db):
        """kb_id 传大写 UUID 字符串：UUID 比较不受大小写影响。"""
        _enable_all_env(monkeypatch)
        queue = self._patch_queue(monkeypatch)
        kb_lower = "00000000-0000-0000-0000-0000000000ab"
        doc = SimpleNamespace(
            id=uuid.UUID(_DOC),
            kb_id=uuid.UUID(kb_lower),
            doc_name="t.md",
            suffix="md",
        )
        self._patch_doc_repo(monkeypatch, doc)
        self._patch_count(monkeypatch, 3)

        task = await enqueue_postproc_rerun(kb_lower.upper(), _DOC, ["raptor"])

        assert task.kb_id == kb_lower.upper()
        queue.put.assert_awaited_once_with(task)

    @pytest.mark.asyncio
    async def test_kb_id_uuid_object(self, monkeypatch, mock_db):
        """kb_id 传 uuid.UUID 对象：正常命中文档。"""
        _enable_all_env(monkeypatch)
        queue = self._patch_queue(monkeypatch)
        doc = SimpleNamespace(
            id=uuid.UUID(_DOC),
            kb_id=uuid.UUID(_KB),
            doc_name="t.md",
            suffix="md",
        )
        self._patch_doc_repo(monkeypatch, doc)
        self._patch_count(monkeypatch, 3)

        task = await enqueue_postproc_rerun(uuid.UUID(_KB), _DOC, ["raptor"])

        assert task.kb_id == _KB
        queue.put.assert_awaited_once_with(task)
