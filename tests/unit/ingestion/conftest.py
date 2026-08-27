"""Ingestion 层测试辅助夹具。"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest


@pytest.fixture
def sample_unified_context_dict() -> dict:
    return {
        "context_id": str(uuid.uuid4()),
        "source": "regular",
        "kb_id": str(uuid.uuid4()),
        "doc_name": "test_doc.pdf",
        "suffix": "pdf",
        "original_file_ref": "s3://bucket/docs/test_ref.pdf",
        "content": b"",
        "chunk_strategy": "auto",
        "ext_metadata": {"tags": ["test"]},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def sample_postproc_config_dict() -> dict:
    return {
        "enable_keyword_extract": True,
        "enable_question_gen": False,
        "enable_summarizer": True,
        "enable_graph_extract": False,
        "enable_community": False,
        "enable_disambiguation": False,
        "enable_wiki": False,
        "enable_raptor": False,
    }


@pytest.fixture
def sample_postproc_task_dict(sample_postproc_config_dict) -> dict:
    return {
        "document_id": str(uuid.uuid4()),
        "kb_id": str(uuid.uuid4()),
        "doc_name": "test_doc.pdf",
        "chunk_count": 10,
        "postproc_config": sample_postproc_config_dict,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
