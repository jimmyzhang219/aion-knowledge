"""删除功能 ORM 字段测试：deleted / enabled 定义与默认值。"""
from __future__ import annotations

from aion_knowledge.models.orm import KnowledgeBase, KnowledgeDocument


def test_kb_has_deleted_column():
    col = KnowledgeBase.__table__.columns["deleted"]
    assert col.default.arg is False          # 默认未删除
    assert col.server_default is not None


def test_doc_has_deleted_and_enabled_columns():
    doc = KnowledgeDocument.__table__.columns
    assert "deleted" in doc
    assert "enabled" in doc
    assert doc["deleted"].default.arg is False    # 默认未删除
    assert doc["enabled"].default.arg is True     # 默认启用
    assert doc["deleted"].server_default is not None
    assert doc["enabled"].server_default is not None
