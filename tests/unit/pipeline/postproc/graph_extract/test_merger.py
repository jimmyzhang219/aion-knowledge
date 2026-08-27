"""KBGraphMerger 测试。"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from aion_knowledge.pipeline.postproc.graph_extract.merger import KBGraphMerger


class TestKBGraphMerger:
    @pytest.mark.asyncio
    async def test_merge_document(self):
        """验证合并文档实体到 Neo4j。"""
        merger = KBGraphMerger()

        entities = {
            "马云": {
                "type": "PERSON",
                "descriptions": ["阿里巴巴创始人"],
                "source_chunks": ["c1"],
            },
            "阿里巴巴": {
                "type": "ORGANIZATION",
                "descriptions": ["电商平台"],
                "source_chunks": ["c1"],
            },
        }
        relations = {
            ("马云", "阿里巴巴", "founder"): {
                "source": "马云", "target": "阿里巴巴",
                "type": "founder", "weight": 8,
                "descriptions": ["创始人"],
            },
        }

        with patch("aion_knowledge.pipeline.postproc.graph_extract.merger.neo4j_add_graph", AsyncMock(return_value=None)):
            with patch.object(merger, "_update_metadata_v2", AsyncMock(return_value=None)):
                result = await merger.merge_document(
                    kb_id="kb1", doc_id="d1",
                    entities=entities, relations=relations,
                )
                assert result is None

    @pytest.mark.asyncio
    async def test_merge_document_empty(self):
        """空实体/关系集合不应报错。"""
        merger = KBGraphMerger()
        with patch("aion_knowledge.pipeline.postproc.graph_extract.merger.neo4j_add_graph", AsyncMock(return_value=None)):
            with patch.object(merger, "_update_metadata_v2", AsyncMock(return_value=None)):
                result = await merger.merge_document(
                    kb_id="kb1", doc_id="d1",
                    entities={}, relations={},
                )
                assert result is None

    @pytest.mark.asyncio
    async def test_update_kb_graph_stats_refreshes_all_counts(self):
        """统计刷新：entity/relation/doc 来自 Neo4j，community 来自 PG 社区表，一并 upsert。"""
        from aion_knowledge.pipeline.postproc.graph_extract import merger as merger_mod

        stats = {"entity_count": 132, "relation_count": 84, "doc_count": 1}

        class _FakeCtx:
            def __init__(self) -> None:
                self.session = AsyncMock()

            async def __aenter__(self) -> AsyncMock:
                return self.session

            async def __aexit__(self, *args) -> bool:
                return False

        with patch.object(merger_mod, "neo4j_get_stats", AsyncMock(return_value=stats)):
            with patch.object(merger_mod.GraphMetadataRepository, "count_communities",
                              AsyncMock(return_value=15)):
                with patch.object(merger_mod.GraphMetadataRepository, "upsert_stats",
                                  AsyncMock()) as mock_upsert:
                    with patch.object(merger_mod, "get_session", return_value=_FakeCtx()):
                        await merger_mod.update_kb_graph_stats("kb1")

        mock_upsert.assert_awaited_once_with(
            kb_id="kb1", entity_count=132, relation_count=84,
            doc_count=1, community_count=15,
        )
