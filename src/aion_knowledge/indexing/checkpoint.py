"""Checkpoint 持久化与恢复机制。

用于 Worker 崩溃恢复：启动时扫描状态为 processing/finalizing 的记录，
根据 Checkpoint 决定回滚或重试。
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = "/tmp/aion_checkpoints"


@dataclass
class Checkpoint:
    """单个文档的处理断点。"""
    document_id: str
    component: str          # parser / chunker / dual_write / postproc
    status: str             # processing / completed / failed
    retry_count: int = 0
    progress_pct: float = 0.0
    error_info: dict[str, Any] | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def increment_retry(self) -> None:
        self.retry_count += 1
        self.updated_at = time.time()

    def mark_done(self) -> None:
        self.status = "completed"
        self.progress_pct = 100.0
        self.updated_at = time.time()


class CheckpointManager:
    """文件系统持久化的 Checkpoint 管理器。"""

    def __init__(self, directory: str = CHECKPOINT_DIR) -> None:
        self._dir = directory
        os.makedirs(self._dir, exist_ok=True)

    def _path(self, document_id: str) -> str:
        return os.path.join(self._dir, f"{document_id}.json")

    async def save(self, document_id: str, cp: Checkpoint) -> None:
        cp.updated_at = time.time()
        with open(self._path(document_id), "w") as f:
            f.write(json.dumps({
                "document_id": cp.document_id,
                "component": cp.component,
                "status": cp.status,
                "retry_count": cp.retry_count,
                "progress_pct": cp.progress_pct,
                "error_info": cp.error_info,
                "created_at": cp.created_at,
                "updated_at": cp.updated_at,
            }))

    async def load(self, document_id: str) -> Checkpoint | None:
        path = self._path(document_id)
        if not os.path.isfile(path):
            return None
        with open(path) as f:
            data = json.load(f)
        return Checkpoint(**data)

    async def delete(self, document_id: str) -> None:
        path = self._path(document_id)
        if os.path.isfile(path):
            os.remove(path)

    async def list_stale(self, max_age_seconds: float = 300) -> list[str]:
        """列出超过 max_age_seconds 未更新的 checkpoint 文档 ID。"""
        now = time.time()
        stale: list[str] = []
        if not os.path.isdir(self._dir):
            return stale
        for fname in os.listdir(self._dir):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(self._dir, fname)
            try:
                with open(path) as f:
                    data = json.load(f)
                if now - data.get("updated_at", 0) > max_age_seconds:
                    stale.append(data["document_id"])
            except (json.JSONDecodeError, KeyError):
                logger.warning("Corrupted checkpoint file: %s", path)
                continue
        return stale
