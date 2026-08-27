"""Aion Knowledge API — ``python -m aion_knowledge`` 入口。"""

from __future__ import annotations

import asyncio
import logging
import os

import uvicorn

from aion_knowledge.api import create_app
from aion_knowledge.common.logger import setup_logging

app = create_app()


def main() -> None:
    # 在 uvicorn 启动前配置日志，避免 uvicorn dictConfig 覆盖根 logger
    setup_logging()
    logging.getLogger().setLevel(logging.INFO)

    config = uvicorn.Config(
        "aion_knowledge.__main__:app",
        host="0.0.0.0",
        port=19531,
        reload=False,
        log_config=None,
        # 默认 None 会无限期等待进行中的请求/连接排空，PyCharm Stop 第一次点击无效
        # （要第二次强杀）。设有限超时：超时后取消任务强制退出，一次点击即可停止。
        timeout_graceful_shutdown=10,
    )
    server = uvicorn.Server(config)
    try:
        if os.getenv("ASYNCIO_DEBUGGER_ENV") == "True":
            # PyCharm 调试器注入 ASYNCIO_DEBUGGER_ENV 后会把 asyncio.run 替换为不支持
            # loop_factory 的版本（pydevd_nest_asyncio），而 uvicorn 0.30+ 在 Python 3.12
            # 下调用 asyncio.run(..., loop_factory=...) 会直接 TypeError。调试模式下改用
            # serve() 直跑（不带 loop_factory 参数），Debug 才能正常启动。
            asyncio.run(server.serve())
        else:
            server.run()
    except KeyboardInterrupt:
        # 停止信号（PyCharm Stop / Ctrl+C）时 asyncio.Runner 会把任务取消转成
        # KeyboardInterrupt，属正常退出路径，无需打印 traceback。
        logging.getLogger().info("收到停止信号，正常退出")


if __name__ == "__main__":
    main()
