"""ORM 模型定义测试。"""

from sqlalchemy import inspect

from aion_knowledge.models.enums import DocumentStatus, IngestionTaskStatus
from aion_knowledge.models.orm import (
    Base,
    IngestionTask,
    KnowledgeDocument,
)
from aion_knowledge.pipeline.postproc.text.orm import ChunkText


def test_all_models_registered() -> None:
    """所有预期模型应已注册到 Base.metadata。"""
    table_names = Base.metadata.tables.keys()
    expected = {
        "doc_knowledge_documents",
        "chunk_text",
        "task_ingestion_tasks",
    }
    assert expected.issubset(table_names)


def test_knowledge_document_defaults() -> None:
    """列默认值应在 schema 中配置。"""
    mapper = inspect(KnowledgeDocument)
    assert mapper.columns["status"].default.arg == DocumentStatus.pending
    assert callable(mapper.columns["tags"].default.arg)
    assert mapper.columns["tags"].default.arg(None) == []
    assert mapper.columns["source_label"].default.arg == ""


def test_chunk_text_defaults() -> None:
    """列默认值应在 schema 中配置。"""
    mapper = inspect(ChunkText)
    assert mapper.columns["chunk_type"].default.arg == "text"
    assert callable(mapper.columns["chunk_metadata"].default.arg)
    assert mapper.columns["chunk_metadata"].default.arg(None) == {}
    assert callable(mapper.columns["image_refs"].default.arg)
    assert mapper.columns["image_refs"].default.arg(None) == []
    assert callable(mapper.columns["keywords"].default.arg)
    assert mapper.columns["keywords"].default.arg(None) == []


def test_ingestion_task_defaults() -> None:
    """列默认值应在 schema 中配置。"""
    mapper = inspect(IngestionTask)
    assert mapper.columns["status"].default.arg == IngestionTaskStatus.pending
    assert mapper.columns["retry_count"].default.arg == 0
