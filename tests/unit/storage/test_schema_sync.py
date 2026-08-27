"""schema 同步操作过滤器 _is_additive 单元测试。"""

from alembic.operations.ops import (
    AddColumnOp,
    AlterColumnOp,
    BulkInsertOp,
    CreateForeignKeyOp,
    CreateIndexOp,
    CreateTableCommentOp,
    CreateTableOp,
    CreateUniqueConstraintOp,
    DropColumnOp,
    DropConstraintOp,
    DropIndexOp,
    DropTableCommentOp,
    DropTableOp,
    ExecuteSQLOp,
    ModifyTableOps,
    RenameTableOp,
)
from sqlalchemy import Column, Integer, MetaData, String, Table

from aion_knowledge.infrastructure.db import _collect_additive, _is_additive


def _additive_ops():
    """白名单：新增/修改类操作，应被应用。"""
    return [
        CreateTableOp("t1", [Column("id", Integer, primary_key=True)]),
        CreateIndexOp("ix_t1_name", "t1", ["name"]),
        AddColumnOp("t1", Column("name", String(50))),
        CreateUniqueConstraintOp("uq_t1_name", "t1", ["name"]),
        CreateForeignKeyOp("fk_t1_a", "t1", "t2", ["a"], ["id"]),
        CreateTableCommentOp("t1", "注释"),
        AlterColumnOp("t1", "name", modify_type=String(100)),
    ]


def _destructive_ops():
    """黑名单：删除/改名/原始 SQL 等，应被跳过。"""
    return [
        DropTableOp("t1"),
        DropIndexOp("ix_t1_name"),
        DropColumnOp("t1", "name"),
        DropConstraintOp("uq_t1_name", "t1"),
        DropTableCommentOp("t1"),
        RenameTableOp("t1", "t2"),
        ExecuteSQLOp("SELECT 1"),
        BulkInsertOp(Table("t1", MetaData()), [{"a": 1}]),
    ]


def test_additive_ops_accepted():
    assert all(_is_additive(op) for op in _additive_ops())


def test_destructive_ops_rejected():
    assert not any(_is_additive(op) for op in _destructive_ops())


def test_collect_additive_expands_container():
    container = ModifyTableOps("t1", [AddColumnOp("t1", Column("c", Integer))])
    assert _collect_additive([container]) == [container.ops[0]]

    mixed = ModifyTableOps(
        "t1", [AddColumnOp("t1", Column("c", Integer)), DropColumnOp("t1", "x")]
    )
    assert _collect_additive([mixed]) == [mixed.ops[0]]
