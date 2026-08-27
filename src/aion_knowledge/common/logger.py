"""日志配置 — 模块加载时立即生效。

实现要点：
- LogRecord 工厂在每条日志记录创建时注入 ``trace_id``（contextvar 读取，无上下文时为 "-"）。
  工厂在 ``Logger.makeRecord()`` 阶段执行，早于 handler / formatter / filter，
  因此 caplog 等第三方 handler 与 ``logging.info()`` 顶层调用路径均能拿到 trace_id。
"""

from __future__ import annotations

import logging
import sys

from aion_knowledge.common.config import settings
from aion_knowledge.common.trace import trace_id_var

_LOG_FORMAT = "%(asctime)s  %(levelname)-8s  [%(trace_id)s]  %(name)s  %(filename)s:%(lineno)d  %(message)s"

# 保存原始 LogRecord 工厂，链式包装避免覆盖其他库的定制工厂
_original_record_factory = logging.getLogRecordFactory()


def _trace_record_factory(*args: object, **kwargs: object) -> logging.LogRecord:
    """LogRecord 工厂：在每条日志记录创建时注入 trace_id。"""
    record = _original_record_factory(*args, **kwargs)
    record.trace_id = trace_id_var.get() or "-"
    return record


logging.setLogRecordFactory(_trace_record_factory)


# 在模块加载时立即配置日志（比 uvicorn dictConfig 早）
logging.basicConfig(
    level=settings.log_level.value,
    format=_LOG_FORMAT,
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stderr,
    force=True,
)


def setup_logging() -> None:
    """重新配置日志（幂等，force=True 解决被 uvicorn dictConfig 覆盖的问题）。

    历史：Alembic 迁移的 ``fileConfig()`` 曾执行 ``disable_existing_loggers=True``，
    已改为进程内 schema 同步后无此副作用；下方循环保留作防御，确保 logger 可用。
    """
    logging.basicConfig(
        level=settings.log_level.value,
        format=_LOG_FORMAT,
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
        force=True,
    )
    # 重新启用被 Alembic fileConfig disable 的 logger
    for name in logging.root.manager.loggerDict:
        logging.getLogger(name).disabled = False


def get_logger(name: str) -> logging.Logger:
    """获取指定模块名的日志记录器。"""
    return logging.getLogger(name)
