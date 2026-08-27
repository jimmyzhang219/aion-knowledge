"""CommunityModule 测试（改造后版本）。"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aion_knowledge.pipeline.postproc.base import PostProcContext
from aion_knowledge.pipeline.postproc.community.processor import CommunityModule


class TestCommunityModule:
    @pytest.mark.asyncio
    async def test_process_with_kb_graph(self):
        module = CommunityModule()
        ctx = PostProcContext(document_id="d1", kb_id="kb1", doc_name="t.md")
        chunks = [{"chunk_uuid": "c1", "content": "test"}]

        with patch(
            "aion_knowledge.infrastructure.graph.load_kb_graph",
            AsyncMock(return_value=(
                [{"entity_name": "A", "entity_type": "ORG"},
                 {"entity_name": "B", "entity_type": "ORG"},
                 {"entity_name": "C", "entity_type": "ORG"}],
                [{"source_entity": "A", "target_entity": "B", "relation_type": "link", "weight": 1.0},
                 {"source_entity": "B", "target_entity": "C", "relation_type": "link", "weight": 1.0}],
            )),
        ):
            with patch("aion_knowledge.pipeline.postproc.community.processor.get_llm_client_for_module"):
                with patch("aion_knowledge.pipeline.postproc.community.checkpoint.CommunityCheckpointManager") as mock_cpm:
                    mock_cpm_instance = AsyncMock()
                    mock_cpm_instance.should_skip = AsyncMock(return_value=False)
                    mock_cpm.return_value = mock_cpm_instance
                    with patch.object(module, "_generate_and_save_reports", AsyncMock(return_value=3)):
                        with patch(
                            "aion_knowledge.pipeline.postproc.graph_extract.merger.update_kb_graph_stats",
                            AsyncMock(return_value=None),
                        ):
                            count = await module.process(ctx, chunks)
                            assert count == 3

    @pytest.mark.asyncio
    async def test_process_refreshes_kb_stats(self):
        """写入社区后应刷新 graph_metadata 统计（community_count 随之更新）。"""
        module = CommunityModule()
        ctx = PostProcContext(document_id="d1", kb_id="kb1", doc_name="t.md")
        chunks = [{"chunk_uuid": "c1", "content": "test"}]

        with patch(
            "aion_knowledge.infrastructure.graph.load_kb_graph",
            AsyncMock(return_value=(
                [{"entity_name": "A", "entity_type": "ORG"},
                 {"entity_name": "B", "entity_type": "ORG"},
                 {"entity_name": "C", "entity_type": "ORG"}],
                [{"source_entity": "A", "target_entity": "B", "relation_type": "link", "weight": 1.0},
                 {"source_entity": "B", "target_entity": "C", "relation_type": "link", "weight": 1.0}],
            )),
        ):
            with patch("aion_knowledge.pipeline.postproc.community.processor.get_llm_client_for_module"):
                with patch("aion_knowledge.pipeline.postproc.community.checkpoint.CommunityCheckpointManager") as mock_cpm:
                    mock_cpm_instance = AsyncMock()
                    mock_cpm_instance.should_skip = AsyncMock(return_value=False)
                    mock_cpm.return_value = mock_cpm_instance
                    with patch.object(module, "_generate_and_save_reports", AsyncMock(return_value=3)):
                        with patch(
                            "aion_knowledge.pipeline.postproc.graph_extract.merger.update_kb_graph_stats",
                            AsyncMock(),
                        ) as mock_refresh:
                            count = await module.process(ctx, chunks)
                            assert count == 3
                            mock_refresh.assert_awaited_once_with("kb1")

    @pytest.mark.asyncio
    async def test_process_stats_refresh_failure_does_not_break(self):
        """统计刷新失败应记录警告但不影响社区写入主流程。"""
        module = CommunityModule()
        ctx = PostProcContext(document_id="d1", kb_id="kb1", doc_name="t.md")
        chunks = [{"chunk_uuid": "c1", "content": "test"}]

        with patch(
            "aion_knowledge.infrastructure.graph.load_kb_graph",
            AsyncMock(return_value=(
                [{"entity_name": "A", "entity_type": "ORG"},
                 {"entity_name": "B", "entity_type": "ORG"},
                 {"entity_name": "C", "entity_type": "ORG"}],
                [{"source_entity": "A", "target_entity": "B", "relation_type": "link", "weight": 1.0},
                 {"source_entity": "B", "target_entity": "C", "relation_type": "link", "weight": 1.0}],
            )),
        ):
            with patch("aion_knowledge.pipeline.postproc.community.processor.get_llm_client_for_module"):
                with patch("aion_knowledge.pipeline.postproc.community.checkpoint.CommunityCheckpointManager") as mock_cpm:
                    mock_cpm_instance = AsyncMock()
                    mock_cpm_instance.should_skip = AsyncMock(return_value=False)
                    mock_cpm.return_value = mock_cpm_instance
                    with patch.object(module, "_generate_and_save_reports", AsyncMock(return_value=3)):
                        with patch(
                            "aion_knowledge.pipeline.postproc.graph_extract.merger.update_kb_graph_stats",
                            AsyncMock(side_effect=RuntimeError("db down")),
                        ):
                            count = await module.process(ctx, chunks)
                            assert count == 3

    @pytest.mark.asyncio
    async def test_fallback_no_kb_graph(self):
        module = CommunityModule()
        ctx = PostProcContext(document_id="d1", kb_id="kb1", doc_name="t.md")
        chunks = [{"chunk_uuid": "c1", "content": "实体A和实体B有合作关系。"}]

        with patch(
            "aion_knowledge.infrastructure.graph.load_kb_graph",
            AsyncMock(return_value=([], [])),
        ):
            with patch.object(module, "_fallback_process", AsyncMock(return_value=2)):
                count = await module.process(ctx, chunks)
                assert count == 2

    @pytest.mark.asyncio
    async def test_empty_chunks(self):
        module = CommunityModule()
        ctx = PostProcContext(document_id="d1", kb_id="kb1", doc_name="t.md")
        assert await module.process(ctx, []) == 0

    def test_depends_on_updated(self):
        assert "disambiguation" in CommunityModule.depends_on
        assert "text" in CommunityModule.depends_on
        assert CommunityModule.always_on is False

    def test_module_factory(self):
        from aion_knowledge.pipeline.postproc.community.processor import module
        assert isinstance(module(), CommunityModule)


class _TrackingSession:
    """模拟 get_session：跟踪会话活跃状态，供断言「LLM 调用期间无活动连接」。"""

    active = False
    enter_count = 0

    def __init__(self):
        self.added: list = []

    async def __aenter__(self):
        type(self).active = True
        type(self).enter_count += 1
        return self

    async def __aexit__(self, *exc):
        type(self).active = False

    def add_all(self, rows):
        self.added.extend(rows)

    async def flush(self):
        pass


_KB_UUID = "00000000-0000-0000-0000-000000000001"
_CHUNK_UUID = "00000000-0000-0000-0000-000000000002"


@pytest.mark.asyncio
async def test_generate_and_save_reports_no_active_session_during_llm():
    """LLM 报告生成期间不得有活动会话；报告收集完成后一次批量写入。"""
    _TrackingSession.enter_count = 0
    _TrackingSession.active = False
    module = CommunityModule()
    communities = [
        {"id": "c1", "level": 0, "members": ["A", "B"]},
        {"id": "c2", "level": 1, "members": ["C"]},
    ]

    async def slow_generate_report(llm, prompt, members, relations):
        assert not _TrackingSession.active, "LLM 调用期间存在活动 DB 会话"
        await asyncio.sleep(0.01)
        return {"title": "t", "summary": "s", "findings": [{"summary": "f", "explanation": "e"}], "rating": 5}

    tracking = _TrackingSession()
    with (
        patch.object(module, "_generate_report", side_effect=slow_generate_report),
        # get_session 在 processor 模块顶部 import，须 patch 其模块命名空间
        patch("aion_knowledge.pipeline.postproc.community.processor.get_session",
              return_value=tracking),
    ):
        inserted = await module._generate_and_save_reports(
            AsyncMock(), _KB_UUID, communities, []
        )
        assert inserted == 2
        assert _TrackingSession.enter_count == 1, "会话应只打开 1 次"
        assert len(tracking.added) == 2


@pytest.mark.asyncio
async def test_fallback_process_no_active_session_during_llm():
    """fallback 路径同样遵守：LLM 提取/生成期间无活动会话。"""
    _TrackingSession.enter_count = 0
    _TrackingSession.active = False
    module = CommunityModule()
    ctx = PostProcContext(document_id="d1", kb_id=_KB_UUID, doc_name="t.md")
    # 单条 chunk 提取 2 实体 + 1 关系，构成 2 节点连通图 → 1 个社区 → 1 条报告
    chunks = [{"chunk_uuid": _CHUNK_UUID, "content": "阿里巴巴是一家位于杭州的公司。"}]

    async def slow_extract(llm, content, entity_types=None, max_gleanings=None):
        assert not _TrackingSession.active, "LLM 调用期间存在活动 DB 会话"
        await asyncio.sleep(0.01)
        return (
            [{"name": "阿里巴巴", "type": "organization"},
             {"name": "杭州", "type": "location"}],
            [{"source": "阿里巴巴", "target": "杭州", "type": "located_in"}],
        )

    async def slow_generate_report(llm, prompt, members, relations):
        assert not _TrackingSession.active, "LLM 调用期间存在活动 DB 会话"
        await asyncio.sleep(0.01)
        return {"title": "t", "summary": "s", "findings": [], "rating": 5}

    tracking = _TrackingSession()
    with (
        patch("aion_knowledge.pipeline.postproc.graph_extract.extractor.extract_entities_with_gleaning",
              side_effect=slow_extract),
        patch.object(module, "_generate_report", side_effect=slow_generate_report),
        patch("aion_knowledge.pipeline.postproc.community.processor.get_llm_client_for_module"),
        patch("aion_knowledge.pipeline.postproc.community.processor.get_session",
              return_value=tracking),
    ):
        count = await module._fallback_process(ctx, chunks)
        assert count == 1
        assert _TrackingSession.enter_count == 1, "会话应只打开 1 次"


@pytest.mark.asyncio
async def test_generate_and_save_reports_embeds_batch_and_writes():
    """主路径：报告收集完后批量嵌入（1 次调用），写入的 embedding 与 embedder 输出一致。"""
    module = CommunityModule()
    communities = [
        {"id": "c1", "level": 0, "members": ["A", "B"]},
        {"id": "c2", "level": 1, "members": ["C"]},
    ]

    async def fake_generate(llm, prompt, members, relations):
        return {"title": f"t-{members[0]}", "summary": "s",
                "findings": [{"summary": "f", "explanation": "e"}], "rating": 5}

    embed_provider = AsyncMock()
    embed_provider.embed_documents = AsyncMock(return_value=[[0.1] * 1024, [0.2] * 1024])

    tracking = _TrackingSession()
    with (
        patch.object(module, "_generate_report", side_effect=fake_generate),
        patch("aion_knowledge.pipeline.postproc.community.processor.create_embedder",
              return_value=embed_provider),
        patch("aion_knowledge.pipeline.postproc.community.processor.get_session",
              return_value=tracking),
    ):
        inserted = await module._generate_and_save_reports(
            AsyncMock(), _KB_UUID, communities, []
        )

    assert inserted == 2
    # 批量：一次 embed_documents 调用，入参为拼接文本列表
    embed_provider.embed_documents.assert_awaited_once()
    texts = embed_provider.embed_documents.await_args.args[0]
    assert texts == ["t-A\ns\nf: e", "t-C\ns\nf: e"]
    assert len(tracking.added) == 2
    embeddings = [row.embedding for row in tracking.added]
    assert embeddings == [[0.1] * 1024, [0.2] * 1024]


@pytest.mark.asyncio
async def test_generate_and_save_reports_embed_failure_keeps_null():
    """嵌入失败不阻断入库：报告仍写入，embedding 为 None，不抛异常。"""
    module = CommunityModule()
    communities = [{"id": "c1", "level": 0, "members": ["A", "B"]}]

    async def fake_generate(llm, prompt, members, relations):
        return {"title": "t", "summary": "s", "findings": [], "rating": 5}

    embed_provider = AsyncMock()
    embed_provider.embed_documents = AsyncMock(side_effect=RuntimeError("ollama down"))

    tracking = _TrackingSession()
    with (
        patch.object(module, "_generate_report", side_effect=fake_generate),
        patch("aion_knowledge.pipeline.postproc.community.processor.create_embedder",
              return_value=embed_provider),
        patch("aion_knowledge.pipeline.postproc.community.processor.get_session",
              return_value=tracking),
    ):
        inserted = await module._generate_and_save_reports(
            AsyncMock(), _KB_UUID, communities, []
        )

    assert inserted == 1
    assert tracking.added[0].embedding is None


@pytest.mark.asyncio
async def test_fallback_process_writes_embedding():
    """回退路径同样生成摘要向量。"""
    module = CommunityModule()
    ctx = PostProcContext(document_id="d1", kb_id=_KB_UUID, doc_name="t.md")
    chunks = [{"chunk_uuid": _CHUNK_UUID, "content": "阿里巴巴是一家位于杭州的公司。"}]

    async def fake_extract(llm, content, entity_types=None, max_gleanings=None):
        return (
            [{"name": "阿里巴巴", "type": "organization"},
             {"name": "杭州", "type": "location"}],
            [{"source": "阿里巴巴", "target": "杭州", "type": "located_in"}],
        )

    async def fake_generate(llm, prompt, members, relations):
        return {"title": "t", "summary": "s", "findings": [], "rating": 5}

    embed_provider = AsyncMock()
    embed_provider.embed_documents = AsyncMock(return_value=[[0.3] * 1024])

    tracking = _TrackingSession()
    with (
        patch("aion_knowledge.pipeline.postproc.graph_extract.extractor.extract_entities_with_gleaning",
              side_effect=fake_extract),
        patch.object(module, "_generate_report", side_effect=fake_generate),
        patch("aion_knowledge.pipeline.postproc.community.processor.get_llm_client_for_module"),
        patch("aion_knowledge.pipeline.postproc.community.processor.create_embedder",
              return_value=embed_provider),
        patch("aion_knowledge.pipeline.postproc.community.processor.get_session",
              return_value=tracking),
    ):
        count = await module._fallback_process(ctx, chunks)

    assert count == 1
    assert tracking.added[0].embedding == [0.3] * 1024


@pytest.mark.asyncio
async def test_fallback_process_refreshes_kb_stats():
    """fallback 路径写完社区后同样刷新 graph_metadata 统计（community_count 随之更新）。"""
    module = CommunityModule()
    ctx = PostProcContext(document_id="d1", kb_id=_KB_UUID, doc_name="t.md")
    chunks = [{"chunk_uuid": _CHUNK_UUID, "content": "阿里巴巴是一家位于杭州的公司。"}]

    async def fake_extract(llm, content, entity_types=None, max_gleanings=None):
        return (
            [{"name": "阿里巴巴", "type": "organization"},
             {"name": "杭州", "type": "location"}],
            [{"source": "阿里巴巴", "target": "杭州", "type": "located_in"}],
        )

    async def fake_generate(llm, prompt, members, relations):
        return {"title": "t", "summary": "s", "findings": [], "rating": 5}

    tracking = _TrackingSession()
    with (
        patch("aion_knowledge.pipeline.postproc.graph_extract.extractor.extract_entities_with_gleaning",
              side_effect=fake_extract),
        patch.object(module, "_generate_report", side_effect=fake_generate),
        patch("aion_knowledge.pipeline.postproc.community.processor.get_llm_client_for_module"),
        patch("aion_knowledge.pipeline.postproc.community.processor.get_session",
              return_value=tracking),
        patch("aion_knowledge.pipeline.postproc.graph_extract.merger.update_kb_graph_stats",
              AsyncMock()) as mock_refresh,
    ):
        count = await module._fallback_process(ctx, chunks)

    assert count == 1
    mock_refresh.assert_awaited_once_with(str(_KB_UUID))


@pytest.mark.asyncio
async def test_generate_and_save_reports_embed_partial_none():
    """embed_documents 返回部分 None 元素：None 对应报告的 embedding 为 None，其余正常。"""
    module = CommunityModule()
    communities = [
        {"id": "c1", "level": 0, "members": ["A", "B"]},
        {"id": "c2", "level": 1, "members": ["C"]},
    ]

    async def fake_generate(llm, prompt, members, relations):
        return {"title": f"t-{members[0]}", "summary": "s", "findings": [], "rating": 5}

    embed_provider = AsyncMock()
    embed_provider.embed_documents = AsyncMock(return_value=[[0.1] * 1024, None])

    tracking = _TrackingSession()
    with (
        patch.object(module, "_generate_report", side_effect=fake_generate),
        patch("aion_knowledge.pipeline.postproc.community.processor.create_embedder",
              return_value=embed_provider),
        patch("aion_knowledge.pipeline.postproc.community.processor.get_session",
              return_value=tracking),
    ):
        inserted = await module._generate_and_save_reports(
            AsyncMock(), _KB_UUID, communities, []
        )

    assert inserted == 2
    embeddings = [row.embedding for row in tracking.added]
    assert embeddings == [[0.1] * 1024, None]


@pytest.mark.asyncio
async def test_generate_and_save_reports_embed_empty_list():
    """embed_documents 返回空列表：全部 embedding 为 None，报告仍入库。"""
    module = CommunityModule()
    communities = [{"id": "c1", "level": 0, "members": ["A", "B"]}]

    async def fake_generate(llm, prompt, members, relations):
        return {"title": "t", "summary": "s", "findings": [], "rating": 5}

    embed_provider = AsyncMock()
    embed_provider.embed_documents = AsyncMock(return_value=[])

    tracking = _TrackingSession()
    with (
        patch.object(module, "_generate_report", side_effect=fake_generate),
        patch("aion_knowledge.pipeline.postproc.community.processor.create_embedder",
              return_value=embed_provider),
        patch("aion_knowledge.pipeline.postproc.community.processor.get_session",
              return_value=tracking),
    ):
        inserted = await module._generate_and_save_reports(
            AsyncMock(), _KB_UUID, communities, []
        )

    assert inserted == 1
    assert tracking.added[0].embedding is None


class _NullEmbeddingSession:
    """模拟 get_session：execute 返回 NULL embedding 的存量记录行。"""

    def __init__(self, null_rows: list[dict]):
        self._null_rows = null_rows
        self._updated: list[str] = []
        self.commits = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass

    async def execute(self, stmt, params=None):
        # 查询语句返回存量行；UPDATE 语句记录 id
        if stmt.text.startswith("SELECT"):
            return _Rows(self._null_rows)
        self._updated.append(params["id"])
        return MagicMock()

    async def commit(self):
        self.commits += 1


class _Rows:
    def __init__(self, rows: list[dict]):
        self._rows = rows

    def fetchall(self):
        return [SimpleNamespace(**r) for r in self._rows]


@pytest.mark.asyncio
async def test_checkpoint_hit_backfills_missing_embeddings():
    """检查点命中且存在 NULL embedding 存量记录：只补向量，不重跑 LLM。"""
    module = CommunityModule()
    ctx = PostProcContext(document_id="d1", kb_id=_KB_UUID, doc_name="t.md")
    chunks = [{"chunk_uuid": _CHUNK_UUID, "content": "test"}]

    null_rows = [
        {"id": "c1", "title": "旧社区A", "summary": "旧摘要A", "findings": []},
        {"id": "c2", "title": "旧社区B", "summary": "旧摘要B",
         "findings": [{"summary": "f", "explanation": "e"}]},
    ]
    session = _NullEmbeddingSession(null_rows)

    embed_provider = AsyncMock()
    embed_provider.embed_documents = AsyncMock(return_value=[[0.4] * 1024, [0.5] * 1024])

    with (
        patch("aion_knowledge.infrastructure.graph.load_kb_graph",
              AsyncMock(return_value=(
                  [{"entity_name": "A", "entity_type": "ORG"}], []))),
        patch("aion_knowledge.pipeline.postproc.community.processor.get_llm_client_for_module",
              new_callable=AsyncMock) as mock_llm,
        patch("aion_knowledge.pipeline.postproc.community.checkpoint.CommunityCheckpointManager") as mock_cpm,
        patch("aion_knowledge.pipeline.postproc.community.processor.create_embedder",
              return_value=embed_provider),
        patch("aion_knowledge.pipeline.postproc.community.processor.get_session",
              return_value=session),
    ):
        mock_cpm_instance = AsyncMock()
        mock_cpm_instance.should_skip = AsyncMock(return_value=True)
        mock_cpm.return_value = mock_cpm_instance
        count = await module.process(ctx, chunks)

    assert count == 2  # 补了 2 条
    # LLM 未被调用（不重跑报告）：get_llm_client_for_module 从未被调用
    mock_llm.assert_not_called()
    # 嵌入文本与存量记录同源拼接
    texts = embed_provider.embed_documents.await_args.args[0]
    assert texts == ["旧社区A\n旧摘要A", "旧社区B\n旧摘要B\nf: e"]
    assert session._updated == ["c1", "c2"]
    assert session.commits == 1


@pytest.mark.asyncio
async def test_checkpoint_hit_no_null_skips():
    """检查点命中且无 NULL 记录：直接返回 0，无 embedder 调用。"""
    module = CommunityModule()
    ctx = PostProcContext(document_id="d1", kb_id=_KB_UUID, doc_name="t.md")
    chunks = [{"chunk_uuid": _CHUNK_UUID, "content": "test"}]

    session = _NullEmbeddingSession([])
    embed_provider = AsyncMock()

    with (
        patch("aion_knowledge.infrastructure.graph.load_kb_graph",
              AsyncMock(return_value=(
                  [{"entity_name": "A", "entity_type": "ORG"}], []))),
        patch("aion_knowledge.pipeline.postproc.community.processor.get_llm_client_for_module"),
        patch("aion_knowledge.pipeline.postproc.community.checkpoint.CommunityCheckpointManager") as mock_cpm,
        patch("aion_knowledge.pipeline.postproc.community.processor.create_embedder",
              return_value=embed_provider),
        patch("aion_knowledge.pipeline.postproc.community.processor.get_session",
              return_value=session),
    ):
        mock_cpm_instance = AsyncMock()
        mock_cpm_instance.should_skip = AsyncMock(return_value=True)
        mock_cpm.return_value = mock_cpm_instance
        count = await module.process(ctx, chunks)

    assert count == 0
    embed_provider.embed_documents.assert_not_awaited()


@pytest.mark.asyncio
async def test_backfill_embed_failure_returns_zero():
    """补向量时 embed 抛异常：返回 0，不抛异常，无 UPDATE。"""
    module = CommunityModule()
    null_rows = [
        {"id": "c1", "title": "旧社区A", "summary": "旧摘要A", "findings": []},
    ]
    session = _NullEmbeddingSession(null_rows)
    embed_provider = AsyncMock()
    embed_provider.embed_documents = AsyncMock(side_effect=RuntimeError("ollama down"))

    with (
        patch("aion_knowledge.pipeline.postproc.community.processor.create_embedder",
              return_value=embed_provider),
        patch("aion_knowledge.pipeline.postproc.community.processor.get_session",
              return_value=session),
    ):
        count = await module._backfill_missing_embeddings(_KB_UUID)

    assert count == 0
    assert session._updated == []
    assert session.commits == 0


@pytest.mark.asyncio
async def test_backfill_embed_count_mismatch_truncates():
    """embed 返回数量少于存量行：zip 截断，只更新返回数量的行。"""
    module = CommunityModule()
    null_rows = [
        {"id": "c1", "title": "旧社区A", "summary": "旧摘要A", "findings": []},
        {"id": "c2", "title": "旧社区B", "summary": "旧摘要B", "findings": []},
    ]
    session = _NullEmbeddingSession(null_rows)
    embed_provider = AsyncMock()
    embed_provider.embed_documents = AsyncMock(return_value=[[0.4] * 1024])

    with (
        patch("aion_knowledge.pipeline.postproc.community.processor.create_embedder",
              return_value=embed_provider),
        patch("aion_knowledge.pipeline.postproc.community.processor.get_session",
              return_value=session),
    ):
        count = await module._backfill_missing_embeddings(_KB_UUID)

    assert count == 1
    assert session._updated == ["c1"]  # zip 截断，只更新前 1 条
    assert session.commits == 1


@pytest.mark.asyncio
async def test_backfill_partial_none_embeddings():
    """embed 返回部分 None：None 对应的行跳过 UPDATE，其余正常。"""
    module = CommunityModule()
    null_rows = [
        {"id": "c1", "title": "旧社区A", "summary": "旧摘要A", "findings": []},
        {"id": "c2", "title": "旧社区B", "summary": "旧摘要B", "findings": []},
    ]
    session = _NullEmbeddingSession(null_rows)
    embed_provider = AsyncMock()
    embed_provider.embed_documents = AsyncMock(return_value=[[0.4] * 1024, None])

    with (
        patch("aion_knowledge.pipeline.postproc.community.processor.create_embedder",
              return_value=embed_provider),
        patch("aion_knowledge.pipeline.postproc.community.processor.get_session",
              return_value=session),
    ):
        count = await module._backfill_missing_embeddings(_KB_UUID)

    assert count == 1
    assert session._updated == ["c1"]
    assert session.commits == 1
