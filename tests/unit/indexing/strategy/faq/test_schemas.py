"""Tests for FAQ pydantic models."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from aion_knowledge.indexing.strategy.faq.schemas import (
    FAQChunk,
    FAQChunkMetadata,
    FAQEntry,
    FAQImportResult,
)


class TestFAQChunkMetadata:
    def test_minimal(self):
        m = FAQChunkMetadata(standard_question="Q1", answers=["A1"])
        assert m.standard_question == "Q1"
        assert m.answers == ["A1"]
        assert m.similar_questions == []
        assert m.negative_questions == []
        assert m.answer_strategy == "all"
        assert m.version == 1

    def test_standard_question_required(self):
        with pytest.raises(ValidationError):
            FAQChunkMetadata(answers=["A1"])

    def test_answers_required(self):
        with pytest.raises(ValidationError):
            FAQChunkMetadata(standard_question="Q1")


class TestFAQEntry:
    def test_minimal(self):
        e = FAQEntry(standard_question="Q1", answers=["A1"])
        assert e.standard_question == "Q1"
        assert e.answers == ["A1"]
        assert e.tags == []
        assert e.enabled is True

    def test_standard_question_required(self):
        with pytest.raises(ValidationError):
            FAQEntry(answers=["A1"])

    def test_answers_required(self):
        with pytest.raises(ValidationError):
            FAQEntry(standard_question="Q1")


class TestFAQChunk:
    def test_minimal(self):
        c = FAQChunk(
            chunk_id="id1",
            kb_id="kb1",
            document_id="doc1",
            content="Q: Q1\nA: A1",
            metadata=FAQChunkMetadata(standard_question="Q1", answers=["A1"]),
        )
        assert c.chunk_type == "faq"
        assert c.tags == []


class TestFAQImportResult:
    def test_defaults(self):
        r = FAQImportResult(mode="append")
        assert r.mode == "append"
        assert r.total == 0
        assert r.inserted == 0
        assert r.error_details == []

    def test_full(self):
        r = FAQImportResult(
            mode="replace", total=10, inserted=8, updated=1, skipped=1, errors=0,
        )
        assert r.total == 10
