"""Pydantic 模型测试。"""

import uuid

import pytest
from pydantic import ValidationError

from aion_knowledge.models.schemas import (
    KnowledgeBaseCreate,
    KnowledgeBaseResponse,
    KnowledgeDocumentCreate,
    SearchRequest,
    SearchResponse,
)


def test_knowledge_document_create_valid() -> None:
    data = {
        "kb_id": str(uuid.uuid4()),
        "doc_name": "test.pdf",
        "suffix": "pdf",
        "size": 1024,
        "creator": "tester",
    }
    doc = KnowledgeDocumentCreate(**data)
    assert doc.doc_name == "test.pdf"
    assert doc.size == 1024


def test_knowledge_document_create_invalid_size() -> None:
    with pytest.raises(ValidationError):
        KnowledgeDocumentCreate(
            kb_id=uuid.uuid4(),
            doc_name="test.pdf",
            suffix="pdf",
            size=-1,
            creator="tester",
        )


def test_search_request_valid() -> None:
    kb_id = str(uuid.uuid4())
    req = SearchRequest(query="hello", kb_id=kb_id, top_k=20, path_top_k=20)
    assert req.query == "hello"
    assert req.top_k == 20
    assert req.path_top_k == 20


def test_search_request_empty_query() -> None:
    with pytest.raises(ValidationError):
        SearchRequest(query="", kb_id=str(uuid.uuid4()))


def test_search_response() -> None:
    resp = SearchResponse(results=[], query="test")
    assert resp.total_fused == 0
    assert resp.answer is None


def test_knowledge_base_create_valid() -> None:
    data = {"name": "项目 Alpha", "tags": ["研发", "核心"]}
    kb = KnowledgeBaseCreate(**data)
    assert kb.name == "项目 Alpha"
    assert kb.tags == ["研发", "核心"]
    assert kb.description == ""


def test_knowledge_base_create_name_required() -> None:
    with pytest.raises(ValidationError):
        KnowledgeBaseCreate()


def test_knowledge_base_response_from_attributes() -> None:
    from datetime import datetime, timezone

    data = {
        "id": uuid.uuid4(),
        "name": "test",
        "tags": [],
        "description": "",
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    resp = KnowledgeBaseResponse(**data)
    assert resp.name == "test"
