"""FAQ 数据模型：FAQChunkMetadata、FAQEntry、FAQChunk、FAQImportResult。"""

from __future__ import annotations

from pydantic import BaseModel


class FAQChunkMetadata(BaseModel):
    """存储于 chunk_text.metadata 中的 FAQ 结构化数据。"""
    standard_question: str
    similar_questions: list[str] = []
    negative_questions: list[str] = []
    answers: list[str]
    answer_strategy: str = "all"    # "all" | "random"
    version: int = 1
    source: str = ""


class FAQEntry(BaseModel):
    """API 传输对象 — 一条 FAQ 条目。"""
    standard_question: str
    similar_questions: list[str] = []
    negative_questions: list[str] = []
    answers: list[str]
    answer_strategy: str = "all"
    tags: list[str] = []
    enabled: bool = True


class FAQChunk(BaseModel):
    """数据库读写对象 — 一条 FAQ chunk 记录。"""
    chunk_id: str
    kb_id: str
    document_id: str
    chunk_type: str = "faq"
    content: str
    metadata: FAQChunkMetadata
    tags: list[str] = []


class FAQImportResult(BaseModel):
    """FAQ 导入操作的结果统计。"""
    mode: str
    total: int = 0
    inserted: int = 0
    updated: int = 0
    skipped: int = 0
    errors: int = 0
    error_details: list[str] = []
