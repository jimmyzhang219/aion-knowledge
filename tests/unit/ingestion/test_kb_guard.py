"""kb_guard 校验守卫测试。"""
from unittest.mock import AsyncMock, patch

import pytest

from aion_knowledge.ingestion.kb_guard import KnowledgeBaseNotFoundError, ensure_kb_exists

_VALID_UUID = "00000000-0000-0000-0000-000000000001"


@pytest.mark.asyncio
async def test_kb_exists_passes():
    """知识库存在时校验通过。"""
    with patch(
        "aion_knowledge.ingestion.kb_guard.KnowledgeBaseRepo.get_by_id",
        new_callable=AsyncMock, return_value=object(),
    ):
        await ensure_kb_exists(_VALID_UUID)  # 不抛异常即通过


@pytest.mark.asyncio
async def test_kb_not_found_raises():
    """知识库不存在时抛 KnowledgeBaseNotFoundError。"""
    with patch(
        "aion_knowledge.ingestion.kb_guard.KnowledgeBaseRepo.get_by_id",
        new_callable=AsyncMock, return_value=None,
    ):
        with pytest.raises(KnowledgeBaseNotFoundError, match="not found"):
            await ensure_kb_exists(_VALID_UUID)


@pytest.mark.asyncio
async def test_invalid_uuid_raises():
    """非法 UUID 字符串归入同一异常（消息含 invalid kb id）。"""
    with pytest.raises(KnowledgeBaseNotFoundError, match="invalid kb id"):
        await ensure_kb_exists("not-a-uuid")
