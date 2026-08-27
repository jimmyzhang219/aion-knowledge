"""SQLAlchemy 异步引擎和会话工厂。

用法：

    async with get_session() as session:
        result = await session.execute(...)
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from aion_knowledge.common.config import settings

_engine = create_async_engine(
    settings.db_url,
    echo=False,
    pool_size=5,
    max_overflow=10,
    # 检出前验证连接，自动替换远端 PG 的陈旧连接（长跑服务偶发
    # "connection was closed in the middle of operation"）
    pool_pre_ping=True,
    # 定期回收空闲连接，防御跨公网远程 PG 的中间设备空闲断开
    # （pre_ping 只能救「取连接时已死」，pool_recycle 主动避免空闲连接被回收）
    pool_recycle=1800,
)

_async_session_factory = async_sessionmaker(
    bind=_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    """生成异步会话；退出时提交，异常时回滚。"""
    session = _async_session_factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI 依赖：生成会话，异常时自动回滚。"""
    async with get_session() as session:
        yield session


def _is_additive(op: Any) -> bool:
    """判断 schema 同步操作是否为「新增/修改」类（Hibernate ``ddl-auto=update`` 语义）。

    只应用建表/加列/加索引/加约束/改类型/改注释等操作；
    删除类操作（删表/删列/删索引/删约束/改名等）一律跳过。
    """
    from alembic.operations.ops import (
        AddColumnOp,
        AddConstraintOp,
        AlterColumnOp,
        CreateIndexOp,
        CreateTableCommentOp,
        CreateTableOp,
    )

    return isinstance(
        op, (CreateTableOp, CreateIndexOp, AddColumnOp, AddConstraintOp,
             CreateTableCommentOp, AlterColumnOp)
    )


def _collect_additive(ops_list: list[Any], logger: Any = None) -> list[Any]:
    """展开 diff 指令列表，返回可直接 invoke 的「新增/修改」类叶子操作。

    ModifyTableOps 等容器递归展开（Operations.invoke 只接受叶子操作）；
    删除/改名等非白名单操作跳过并记日志。
    """
    from alembic.operations.ops import OpContainer

    result = []
    for op in ops_list:
        if isinstance(op, OpContainer):
            result.extend(_collect_additive(op.ops, logger))
        elif _is_additive(op):
            result.append(op)
        elif logger is not None:
            logger.info("skip destructive schema op: %s", type(op).__name__)
    return result


async def _sync_orm_schema() -> None:
    """启动时把 DB schema 对齐到 ORM 模型（进程内 diff 同步，不落盘迁移文件）。

    流程：
      1. 用 ``alembic.autogenerate.produce_migrations`` 对比 ORM 元数据与当前 DB，
         得到差异操作指令列表（与 ``alembic revision --autogenerate`` 同一套引擎）。
      2. 只应用「新增/修改」类操作（见 :func:`_is_additive`），跳过删除类操作。
      3. 全部操作在同一事务内执行，任一失败整体回滚并抛出异常，阻断启动。
    日常改 ORM 模型后只需重启服务，无需手动执行任何 Alembic 命令，
    也不会在 ``migrations/versions/`` 下累积迁移版本文件。
    """
    import logging

    from sqlalchemy import text

    from aion_knowledge.models.orm import Base

    logger = logging.getLogger(__name__)

    # 进程内 import 注册全部 postproc 表到 Base.metadata（同原 migrations/env.py 的方式）
    import aion_knowledge.pipeline.postproc.community.orm  # noqa: F401  ChunkCommunity
    import aion_knowledge.pipeline.postproc.disambiguation.orm  # noqa: F401  ChunkDisambiguation
    import aion_knowledge.pipeline.postproc.graph_extract.metadata_orm  # noqa: F401  GraphMetadata
    import aion_knowledge.pipeline.postproc.raptor.orm  # noqa: F401  ChunkRaptor
    import aion_knowledge.pipeline.postproc.text.orm  # noqa: F401  ChunkText
    import aion_knowledge.pipeline.postproc.vector.orm  # noqa: F401  ChunkVector
    import aion_knowledge.pipeline.postproc.wiki.orm  # noqa: F401  ChunkWiki

    def _apply_diff(sync_conn: Any) -> int:
        """在 run_sync 内执行 diff 并应用，避免在 async 上下文中创建
        同步引擎导致的事件循环阻塞。返回实际应用的操作数。
        """
        from alembic.autogenerate import produce_migrations
        from alembic.migration import MigrationContext
        from alembic.operations import Operations

        mc = MigrationContext.configure(
            sync_conn,
            opts={
                "compare_type": True,
                "compare_server_default": True,
                "compare_comments": True,
            },
        )
        script = produce_migrations(mc, Base.metadata)
        # upgrade_ops 类型为 UpgradeOps | None（升级关闭时为 None），默认升级路径
        # 恒非空，此处仅防御性兜底
        pending = _collect_additive(script.upgrade_ops.ops, logger) if script.upgrade_ops else []
        if not pending:
            return 0
        ops = Operations(mc)
        for op in pending:
            ops.invoke(op)
        return len(pending)

    async with _engine.begin() as conn:
        # 丢弃 Alembic 自有的版本书签表（纯 ORM 同步模式下无迁移版本概念）
        await conn.execute(text("DROP TABLE IF EXISTS alembic_version"))

        # 清理无列引用的孤立 ENUM 类型（早期 create_all 路径遗留的半成品库状态，
        # 不清理会导致 diff 建表时 CREATE TYPE 冲突）。不能用 CASCADE（会连带删除
        # 已有列）；仅清理真正孤立的类型，被引用时跳过不阻断。
        orphan_enums = (await conn.execute(text(
            "SELECT t.typname FROM pg_type t JOIN pg_namespace n ON n.oid = t.typnamespace "
            "WHERE n.nspname = 'public' AND t.typtype = 'e' "
            "AND NOT EXISTS (SELECT 1 FROM pg_attribute a WHERE a.atttypid = t.oid)"
        ))).scalars().all()
        for enum_name in orphan_enums:
            await conn.execute(text("SAVEPOINT drop_enum_sp"))
            try:
                await conn.execute(text(f"DROP TYPE IF EXISTS {enum_name} RESTRICT"))
                await conn.execute(text("RELEASE SAVEPOINT drop_enum_sp"))
            except Exception:
                await conn.execute(text("ROLLBACK TO SAVEPOINT drop_enum_sp"))
                await conn.execute(text("RELEASE SAVEPOINT drop_enum_sp"))

        applied = await conn.run_sync(_apply_diff)

    if applied:
        logger.info("schema sync applied %d change(s) to align DB with ORM", applied)
    else:
        logger.info("ORM and DB schema are in sync.")


async def init_db() -> None:
    """创建所有表并安装扩展。仅用于测试或本地开发。

    行为由配置 ``AION_DB_AUTO_MIGRATE`` 控制：

    - **True** （推荐）：安装 PG 扩展后，启动时对比 ORM 与 DB 结构差异并在
      进程内直接应用（新增/修改，不删除，不生成迁移文件）。ORM 是唯一 schema 来源。
    - **False** （默认，兼容旧行为）：沿用老的 ``create_all`` + 手写 ALTER TABLE 路径。
      适合已有数据库但尚未生成迁移脚本的环境。
    """
    from sqlalchemy import text

    from aion_knowledge.models.orm import Base
    from aion_knowledge.pipeline.postproc.community.orm import ChunkCommunity  # noqa: F401
    from aion_knowledge.pipeline.postproc.disambiguation.orm import (
        ChunkDisambiguation,  # noqa: F401
    )
    from aion_knowledge.pipeline.postproc.graph_extract.metadata_orm import (
        GraphMetadata,  # noqa: F401
    )
    from aion_knowledge.pipeline.postproc.raptor.orm import ChunkRaptor  # noqa: F401
    from aion_knowledge.pipeline.postproc.text.orm import ChunkText  # noqa: F401 注册表
    from aion_knowledge.pipeline.postproc.wiki.orm import ChunkWiki  # noqa: F401

    dim = settings.embedding_dimensions
    async with _engine.begin() as conn:
        # 安装扩展（容错：SAVEPOINT 隔离单个扩展失败，避免事务中止）
        for ext in ("vector", "pg_textsearch", "vectorscale", "zhparser"):
            await conn.execute(text("SAVEPOINT ext_sp"))
            try:
                await conn.execute(text(f"CREATE EXTENSION IF NOT EXISTS {ext}"))
                await conn.execute(text("RELEASE SAVEPOINT ext_sp"))
            except Exception:
                await conn.execute(text("ROLLBACK TO SAVEPOINT ext_sp"))
                await conn.execute(text("RELEASE SAVEPOINT ext_sp"))
                import logging
                logging.getLogger(__name__).warning("Extension '%s' not available, skipping", ext)

    # ── 创建中文 FTS 配置（zhparser parser → zh_cfg） ──
    import logging
    async with _engine.begin() as conn:
        result = await conn.execute(
            text("SELECT EXISTS(SELECT 1 FROM pg_catalog.pg_ts_config WHERE cfgname = 'zh_cfg')")
        )
        exists = result.scalar()
        if not exists:
            await conn.execute(text("CREATE TEXT SEARCH CONFIGURATION zh_cfg (PARSER = zhparser)"))

            # a=形容词, b=区别词, c=连词, d=副词, e=感叹词, f=方位词, g=词根, h=前连接成分, i=成语, j=简称, k=后连接成分, l=习用语, m=数词, n=名词, o=拟声词, p=介词, q=量词, r=代词, s=处所词, t=时语素, u=助词, v=动词, w=标点符号, x=未知词, y=语气词, z=状态词
            await conn.execute(text("""ALTER TEXT SEARCH CONFIGURATION zh_cfg
                                       ADD MAPPING FOR n,v,a,i,e,l,t WITH simple"""))
            logging.getLogger(__name__).info("FTS config 'zh_cfg' created")
        else:
            logging.getLogger(__name__).info("FTS config 'zh_cfg' already exists, skipping")

    # ── Schema 迁移 ──
    if settings.db_auto_migrate:
        # ORM 为唯一 schema 来源：启动时进程内 diff 同步
        await _sync_orm_schema()
    else:
        # ── [旧路径] create_all + raw ALTER TABLE，保留向前兼容 ──
        async with _engine.begin() as conn:
            # ── 清理残余 ENUM 类型（仅清理无列依赖的孤立类型）。
            #    不能用 CASCADE，否则会连带删除已有列。
            #    每个 DROP 用独立 SAVEPOINT 隔离，避免失败后事务不可用。
            for enum_name in ("chunkstrategy", "documentstatus", "ingestiontaskstatus"):
                await conn.execute(text("SAVEPOINT drop_enum_sp"))
                try:
                    await conn.execute(text(f"DROP TYPE IF EXISTS {enum_name} RESTRICT"))
                    await conn.execute(text("RELEASE SAVEPOINT drop_enum_sp"))
                except Exception:
                    await conn.execute(text("ROLLBACK TO SAVEPOINT drop_enum_sp"))
                    await conn.execute(text("RELEASE SAVEPOINT drop_enum_sp"))

            # ── create_all 包裹 SAVEPOINT，单个表失败不影响其他表 ──
            await conn.execute(text("SAVEPOINT orm_sp"))
            try:
                await conn.run_sync(Base.metadata.create_all)
                await conn.execute(text("RELEASE SAVEPOINT orm_sp"))
            except Exception as exc:
                await conn.execute(text("ROLLBACK TO SAVEPOINT orm_sp"))
                await conn.execute(text("RELEASE SAVEPOINT orm_sp"))
                import logging
                logging.getLogger(__name__).warning(
                    "Some ORM tables could not be created, continuing: %s", exc
                )

    # ── 给所有表和字段加注释（幂等操作，Navicat 可见，所有路径共用） ──
    async with _engine.begin() as conn:
        table_comments = {
            "kb_knowledge_bases":      ("知识库信息表", {
                "id": "知识库ID", "name": "知识库名称", "tags": "标签列表",
                "description": "知识库描述", "created_at": "创建时间", "updated_at": "更新时间",
            }),
            "doc_knowledge_documents": ("文档元数据表", {
                "id": "文档ID", "kb_id": "所属知识库ID", "doc_name": "文档名称（含扩展名）",
                "suffix": "文件后缀（pdf/docx/md 等）", "hash": "文件 SHA256 哈希（去重用）",
                "size": "文件大小（字节）", "status": "处理状态", "tags": "标签列表",
                "source_label": "来源标签", "creator": "创建者", "file_path": "对象存储路径",
                "chunk_strategy": "切片策略", "created_at": "创建时间", "updated_at": "更新时间",
            }),
            "task_ingestion_tasks":    ("文档处理任务表", {
                "id": "任务ID", "document_id": "关联文档ID", "pipeline_id": "管道流水线标识",
                "status": "任务状态", "retry_count": "重试次数", "checkpoint": "断点续传状态",
                "error_info": "错误信息", "created_at": "创建时间", "updated_at": "更新时间",
            }),
            "chunk_text":              ("文档切片存储表（text/table/image/parent 等类型）", {
                "id": "切片ID", "document_id": "所属文档ID", "kb_id": "所属知识库ID",
                "content": "切片内容（Markdown / 图片上下文 / VLM 描述）",
                "context_header": "上下文标题路径", "keywords": "关键词列表",
                "seq_num": "切片序号（文档内唯一）",
                "chunk_type": "切片类型：text/table/image/parent 等（见 ChunkType 枚举）",
                "parent_chunk_id": "父切片ID（RAPTOR 层级检索用）", "token_count": "Token 估算数",
                "metadata": "元数据 JSON（table_caption/heading_path/context_above 等）",
                "image_refs": "关联图片 S3 路径列表", "summary_text": "摘要文本",
                "content_tokens": "内容 zhparser 分词结果",
                "summary_tokens": "摘要 zhparser 分词结果",
                "created_at": "创建时间",
            }),
            "chunk_vector":            ("向量嵌入元数据表", {
                "id": "切片向量ID", "chunk_id": "关联切片ID", "kb_id": "所属知识库ID",
                "embedding": f"向量嵌入（{dim}维，pgvector）",
                "embedding_questions": "问题生成向量", "questions": "原始问题文本（调试用）",
                "embedding_summary": "摘要向量",
                "payload": "负载元数据（chunk_type/seq_num 等）", "created_at": "创建时间",
            }),
            "chunk_disambiguation":    ("实体消歧记录表", {
                "id": "消歧记录ID", "chunk_id": "关联切片ID（KB 级消歧决策为 NULL）", "kb_id": "所属知识库ID",
                "entity_name": "实体名称", "resolved_id": "消歧后标准ID",
                "confidence": "消歧置信度", "merged_into": "合并到的目标实体名",
                "merge_confidence": "合并置信度", "payload": "扩展元数据", "created_at": "创建时间",
            }),
            "chunk_community":         ("社区发现结果表", {
                "id": "社区记录ID", "chunk_id": "关联切片ID", "kb_id": "所属知识库ID",
                "community_id": "社区ID", "community_level": "社区层级", "summary": "社区摘要",
                "findings": "社区发现的关键发现",
                "embedding": "社区摘要向量（title+summary+findings 拼接，pgvector）",
                "payload": "扩展元数据", "created_at": "创建时间",
            }),
            "chunk_wiki":              ("百科数据表（KB 级页面池）", {
                "id": "页面ID", "kb_id": "所属知识库ID",
                "page_slug": "页面 slug（KB 内唯一）", "page_title": "页面标题",
                "content": "页面内容（Markdown，含 [[slug]] wikilink）",
                "chunk_refs": "引用 chunk UUID 列表", "source_refs": "贡献文档 UUID 列表",
                "out_links": "本页链接出去的 slug 列表", "in_links": "反向链接（被哪些 slug 链向）",
                "taxonomy_path": "分类路径", "status": "状态（draft/published）",
                "payload": "扩展元数据", "created_at": "创建时间", "updated_at": "更新时间",
            }),
            "chunk_raptor":            ("RAPTOR 递归摘要树节点表", {
                "id": "节点ID", "kb_id": "所属知识库ID",
                "doc_id": "文档ID（NULL 表示 dataset 级别）", "title": "摘要标题",
                "summary": "摘要内容", "embedding": "摘要向量（直接存本表，不依赖 chunk_vector）",
                "layer": "树层级（0=叶子，越大越靠近根）", "cluster_id": "所属聚类ID",
                "parent_id": "父节点ID（NULL=根节点）",
                "source_chunk_ids": "原始 chunk_text ID 列表（溯源链）",
                "tree_json": "整棵树序列化 dict（output_mode=tree 时使用）",
                "tree_builder": "树构建算法（raptor）",
                "clustering_method": "聚类算法（gmm/ahc）",
                "output_mode": "输出模式（flat/tree）", "payload": "扩展元数据",
                "created_at": "创建时间",
            }),
            "graph_metadata":          ("知识图谱元数据统计表", {
                "id": "主键ID", "kb_id": "知识库ID（图谱与知识库 1:1，一个 KB 至多一个图）",
                "status": "图谱状态", "doc_count": "关联文档数",
                "entity_count": "实体总数", "relation_count": "关系总数",
                "community_count": "社区总数", "version": "版本号",
                "checkpoints": "处理断点状态", "updated_at": "更新时间",
            }),
        }
        for table_name, (table_comment, cols) in table_comments.items():
            safe_tc = table_comment.replace("'", "''")
            try:
                await conn.execute(text(f'COMMENT ON TABLE "{table_name}" IS \'{safe_tc}\''))
            except Exception:
                pass  # 表可能不存在，忽略
            for col_name, col_comment in cols.items():
                safe_cc = col_comment.replace("'", "''")
                try:
                    sql = f'COMMENT ON COLUMN "{table_name}"."{col_name}" IS \'{safe_cc}\''
                    await conn.execute(text(sql))
                except Exception:
                    pass  # 列可能不存在，忽略


async def dispose_engine() -> None:
    """释放引擎（例如在应用关闭时）。"""
    await _engine.dispose()
