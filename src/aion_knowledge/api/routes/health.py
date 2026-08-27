"""健康检查端点。"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["system"])


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    timestamp: str = ""


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """服务健康检查。"""
    return HealthResponse(timestamp=datetime.now(timezone.utc).isoformat())
