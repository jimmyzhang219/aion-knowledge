"""schema 进程内同步集成测试：init_db() 的 ORM↔DB diff 同步路径。

运行: cd /Users/jimmy/VSCodeProjects/aion-knowledge && python -m pytest tests/integration/test_schema_sync.py -v -s
"""
from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import Column, Integer, String, Table
from sqlalchemy import text as sql_text

from aion_knowledge.infrastructure.db import _engine, init_db
from aion_knowledge.models.orm import Base

# 测试用临时表（统一前缀，避免与业务表混淆）
_TABLE = "sync_test_scratch"
_EXTRA_TABLE = "sync_test_extra"

pytestmark = pytest.mark.asyncio(loop_scope="session")


def _register(table_name: str, columns: list) -> None:
    """向 Base.metadata 注册一个临时表（_sync_orm_schema 用同一 metadata 做 diff）。"""
    Table(table_name, Base.metadata, *columns)


async def _cleanup() -> None:
    """清理：从 metadata 移除注册 + 删除库里临时表（幂等，开头/结尾各调一次）。"""
    for table_name in (_TABLE, _EXTRA_TABLE):
        table = Base.metadata.tables.get(table_name)
        if table is not None:
            Base.metadata.remove(table)
    async with _engine.begin() as conn:
        for table_name in (_TABLE, _EXTRA_TABLE):
            await conn.execute(sql_text(f"DROP TABLE IF EXISTS {table_name}"))


async def _table_exists(conn, table_name: str) -> bool:
    row = await conn.execute(sql_text(
        f"SELECT to_regclass('public.{table_name}') IS NOT NULL"))
    return bool(row.scalar())


async def test_sync_creates_missing_table():
    """库里没有的表，同步后应被创建。"""
    await _cleanup()
    try:
        _register(_TABLE, [
            Column("id", Integer, primary_key=True),
            Column("name", String(50)),
        ])
        await init_db()
        async with _engine.begin() as conn:
            assert await _table_exists(conn, _TABLE) is True
    finally:
        await _cleanup()


async def test_sync_adds_missing_column():
    """库里已有表缺列，同步后应补上该列。"""
    await _cleanup()
    try:
        async with _engine.begin() as conn:
            await conn.execute(sql_text(
                f"CREATE TABLE {_TABLE} (id INTEGER PRIMARY KEY, name VARCHAR(50))"))
        _register(_TABLE, [
            Column("id", Integer, primary_key=True),
            Column("name", String(50)),
            Column("extra", String(100)),
        ])
        await init_db()
        async with _engine.begin() as conn:
            has_col = (await conn.execute(sql_text(
                "SELECT EXISTS (SELECT 1 FROM information_schema.columns "
                f"WHERE table_name = '{_TABLE}' AND column_name = 'extra')"))).scalar()
        assert has_col is True
    finally:
        await _cleanup()


async def test_sync_keeps_db_only_tables():
    """ORM 里不存在的表，同步后应原样保留（不删除语义）。"""
    await _cleanup()
    try:
        async with _engine.begin() as conn:
            await conn.execute(sql_text(f"CREATE TABLE {_EXTRA_TABLE} (id INTEGER)"))
        await init_db()
        async with _engine.begin() as conn:
            assert await _table_exists(conn, _EXTRA_TABLE) is True
    finally:
        await _cleanup()


async def test_sync_rebuilds_partial_schema():
    """表不全：库里只有部分表时，缺失的表应被补建，已有表不受影响。"""
    await _cleanup()
    try:
        async with _engine.begin() as conn:
            await conn.execute(sql_text(
                f"CREATE TABLE {_TABLE} (id INTEGER PRIMARY KEY, name VARCHAR(50))"))
        # ORM 侧注册两张表，库里只有其中一张
        _register(_TABLE, [
            Column("id", Integer, primary_key=True),
            Column("name", String(50)),
        ])
        _register(_EXTRA_TABLE, [
            Column("id", Integer, primary_key=True),
            Column("flag", String(20)),
        ])
        await init_db()
        async with _engine.begin() as conn:
            assert await _table_exists(conn, _TABLE) is True
            assert await _table_exists(conn, _EXTRA_TABLE) is True
            # 已有表保持原样（没被重建/动过结构）
            has_name = (await conn.execute(sql_text(
                "SELECT EXISTS (SELECT 1 FROM information_schema.columns "
                f"WHERE table_name = '{_TABLE}' AND column_name = 'name')"))).scalar()
            assert has_name is True
    finally:
        await _cleanup()


async def test_sync_fixes_wrong_column_type():
    """字段不对：库里列类型与 ORM 不一致时，应被修正为 ORM 声明类型。"""
    await _cleanup()
    try:
        async with _engine.begin() as conn:
            await conn.execute(sql_text(
                f"CREATE TABLE {_TABLE} (id INTEGER PRIMARY KEY, value INTEGER)"))
        # ORM 侧 value 声明为 VARCHAR(100)，库里却是 INTEGER
        _register(_TABLE, [
            Column("id", Integer, primary_key=True),
            Column("value", String(100)),
        ])
        await init_db()
        async with _engine.begin() as conn:
            typ = (await conn.execute(sql_text(
                "SELECT data_type, character_maximum_length FROM information_schema.columns "
                f"WHERE table_name = '{_TABLE}' AND column_name = 'value'"))).first()
        assert (typ.data_type, typ.character_maximum_length) == ("character varying", 100)
    finally:
        await _cleanup()


async def test_sync_leaves_no_migration_artifacts():
    """同步后不应有 alembic_version 表，仓库也不应再有 alembic 迁移文件。"""
    await _cleanup()
    try:
        await init_db()
        async with _engine.begin() as conn:
            assert await _table_exists(conn, "alembic_version") is False

        repo_root = Path(__file__).resolve().parents[2]
        assert not (repo_root / "migrations").exists()
        assert not (repo_root / "alembic.ini").exists()
    finally:
        await _cleanup()


async def test_sync_failure_blocks_startup_and_rolls_back():
    """任一同步操作失败 → init_db 抛异常（阻断启动）且整体回滚。"""
    await _cleanup()
    try:
        async with _engine.begin() as conn:
            await conn.execute(sql_text(
                f"CREATE TABLE {_TABLE} (id INTEGER PRIMARY KEY, name VARCHAR(50))"))
            await conn.execute(sql_text(
                f"INSERT INTO {_TABLE} (id, name) VALUES (1, 'a'), (2, 'a')"))
        # ORM 侧 name 声明 unique：加唯一约束遇重复值必失败
        _register(_TABLE, [
            Column("id", Integer, primary_key=True),
            Column("name", String(50), unique=True),
        ])
        with pytest.raises(Exception) as exc_info:
            await init_db()
        assert "duplicate" in str(exc_info.value).lower()

        # 整体回滚：数据仍在，唯一约束未被加上
        async with _engine.begin() as conn:
            rows = (await conn.execute(
                sql_text(f"SELECT count(*) FROM {_TABLE}"))).scalar()
            has_constraint = (await conn.execute(sql_text(
                f"SELECT count(*) FROM pg_constraint WHERE conrelid = '{_TABLE}'::regclass"
                " AND contype = 'u'"))).scalar()
        assert rows == 2
        assert has_constraint == 0
    finally:
        await _cleanup()
