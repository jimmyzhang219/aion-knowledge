"""测试根夹具：隔离 DB 连接池 + 清空 trace 上下文，避免跨测试污染。"""

import pytest
import pytest_asyncio

from aion_knowledge.common.trace import trace_id_var
from aion_knowledge.infrastructure.db import dispose_engine


@pytest_asyncio.fixture(autouse=True)
async def _isolate_db_pool() -> None:
    """每个测试开始前清空连接池。

    pytest-asyncio 每测试独立事件循环，池中连接属于上个测试的 loop；
    复用触发 asyncpg "attached to a different loop" / "another operation is in progress"。
    跨 loop dispose 已验证安全（SQLAlchemy 池清理路径不依赖原 loop）。
    空池时是 no-op，不触库的 700+ 测试开销≈0。
    """
    await dispose_engine()


@pytest.fixture(autouse=True)
def _reset_trace_id_var() -> None:
    """每个测试结束后清空主线程 trace_id_var。

    TestClient 经 anyio 门户线程跑应用，线程内 contextvar 改动会写回主线程，
    导致后续同步测试（如 test_logger）读到残留 trace_id；清空后由各请求重新生成。
    """
    yield
    trace_id_var.set("")
