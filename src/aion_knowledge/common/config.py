"""通过 pydantic-settings 实现的应用配置（12factor、环境变量驱动）。"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class EmbeddingProvider(str, Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"
    ALI_CLOUD = "alicloud"
    JINA = "jina"
    AZURE = "azure"
    GEMINI = "gemini"
    ZHIPU = "zhipu"


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    OLLAMA = "ollama"
    GEMINI = "gemini"
    ZHIPU = "zhipu"
    ALI_CLOUD = "alicloud"
    DEEPSEEK = "deepseek"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AION_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    # 数据库
    db_url: str = "postgresql+asyncpg://localhost:5432/aion_knowledge"

    # 对象存储（兼容 S3 协议，支持 MinIO / 阿里云 OSS 等）
    # 目录结构：{s3_prefix}/{doc_id}/{filename}
    #            {s3_prefix}/{doc_id}/images/{filename}
    s3_endpoint: str = "http://localhost:9000"
    s3_access_key: str = ""
    s3_secret_key: str = ""
    s3_region: str = "us-east-1"
    s3_bucket: str = Field(default="aion-knowledge-dev", alias="AION_S3_DOCUMENTS_BUCKET")
    s3_prefix: str = "docs"
    s3_addressing_style: str = "virtual"  # virtual 或 path；阿里云OSS 要求 virtual，MinIO 用 path

    # 嵌入向量
    embedding_provider: EmbeddingProvider = EmbeddingProvider.OPENAI
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    embedding_base_url: str = ""  # Ollama 或其他兼容 API 的地址

    # ── 重排序（独立 reranker 服务，如 Infinity / bge-reranker）──
    reranker_enabled: bool = True                           # 是否开启神经网络重排序
    reranker_endpoint: str = "http://localhost:12111/rerank"
    reranker_max_tokens: int = 512                          # 字符 payload 防护基数（×1.5）；真 token 截断由 TEI 服务端
    reranker_max_batch: int = 8                             # 单批大小（候选分批发送，不再截断丢弃）
    reranker_max_concurrency: int = 4                      # 批次最大并发数（gather + Semaphore 限流，避免压垮 reranker 服务）

    # ── 上传限制 ──
    upload_max_size_mb: int = Field(
        default=50, ge=1, description="单文件上传上限（MB），超限返回 413",
    )

    # ── VLM 多模态模型（用于图片描述，独立于文字 LLM）──
    vlm_provider: str | None = None           # 默认 None = 使用 llm_provider
    vlm_model: str = "qwen-vl-plus"           # 默认 qwen-vl-plus
    vlm_api_key: str = ""
    vlm_base_url: str = ""
    vlm_request_timeout: int = 60

    # 大语言模型
    llm_provider: LLMProvider = LLMProvider.OPENAI
    llm_model: str = "gpt-4o"
    llm_api_key: str = ""
    llm_base_url: str = ""  # OpenAI 兼容 API 地址
    llm_request_timeout: int = 60
    llm_enable_thinking: bool = False
    llm_thinking_budget: int = 0
    llm_max_completion_tokens: int = 4096

    # ── 文档解析 ──
    pdf_render_dpi: int = 200
    pdf_render_max_workers: int = 1
    pdf_render_parallelism: int = 0       # 0 = auto (min(4, cpu_count))
    pdf_render_max_edge: int = 2000
    pdf_jpeg_quality: int = 85
    pdf_force_scanned: bool = False
    pdf_extract_embedded_images: bool = True
    pdf_layout_ordering: bool = True
    pdf_detect_headings: bool = True
    pdf_filter_hidden_text: bool = True
    pdf_external_url: str = ""
    pdf_external_api_key: str = ""
    pdf_external_merge_tables: bool = True
    pdf_external_timeout: int = 300

    markitdown_max_workers: int = 1
    odl_max_workers: int = 1
    odl_hybrid: str = "off"
    odl_hybrid_url: str = "http://127.0.0.1:5002"
    odl_hybrid_mode: str = "auto"
    odl_hybrid_fallback: bool = False
    odl_markdown_with_html: bool = False

    docx_max_pages: int = 0
    external_https_proxy: str = ""

    # 兼容旧字段（通过 model_config alias 复用）
    openai_api_key: str = ""


    # 后处理模块
    postproc_keyword_extract:      bool = True
    postproc_question_gen:         bool = True
    postproc_summarizer:           bool = True
    postproc_raptor:               bool = True
    postproc_graph_extract:        bool = True
    postproc_community:            bool = True
    postproc_disambiguation:       bool = True
    postproc_wiki:                 bool = True

    # ── 多路召回 ──
    # bm25 / vector / faq 无后处理依赖，保留独立开关
    # 其余 8 路（keyword / question_gen / summary / raptor /
    #           kg / disambiguation / community / wiki）与后处理共用
    #           postproc_* 标记（见上方）
    retrieval_path_bm25:            bool = True
    retrieval_path_vector:          bool = True
    retrieval_path_faq:             bool = True

    # RRF 参数
    rrf_k: int = Field(default=60, gt=0, description="平滑常数")
    rrf_top_k: int = Field(default=10, gt=0, description="最终保留结果数")
    rrf_path_top_k: int = Field(default=20, gt=0, description="每路预取结果数")

    # RRF 权重
    rrf_weight_vector:             float = 0.15
    rrf_weight_bm25:               float = 0.10
    rrf_weight_faq:                float = 0.05
    rrf_weight_keyword:            float = 0.08
    rrf_weight_question_gen:       float = 0.10
    rrf_weight_summary:            float = 0.10
    rrf_weight_raptor:             float = 0.05
    rrf_weight_graph:              float = 0.10
    rrf_weight_community:          float = 0.08
    rrf_weight_wiki:               float = 0.07

    # RAPTOR 树遍历检索参数（环境变量 AION_RAPTOR_TRAVERSE_*）
    raptor_traverse_top_trees:     int = 2       # 检索时进入的树（文档）数量
    raptor_traverse_beam:          int = 3       # 逐层剪枝保留的子节点数（beam 宽度）
    raptor_traverse_leaf_topk:     int = 5       # 每条路径末端叶子按相似度取前 N 个
    raptor_traverse_max_leaf_chars: int = 500    # 单叶截断字符数，防超长叶子撑爆上下文

    # 检索超时（秒）
    retrieval_path_timeout:        float = 30.0

    # ── 查询改写 ──
    query_rewrite_enabled: bool = True

    # ── FAQ ──
    faq_score_boost:               float = 1.2               # 检索时 FAQ 分数提升系数
    faq_direct_answer_threshold:   float = 0.85              # 直接答案通道阈值
    faq_embed_answers:             bool = False              # 向量计算是否包含答案
    faq_default_answer_strategy:   str = "all"               # all | random

    # per-module LLM 覆盖（key=模块名, value=覆盖的 LLM 参数字典）
    postproc_module_llm:           dict[str, dict[str, Any]] = Field(default_factory=dict)

    # Neo4j 图数据库（默认禁用，需在 .env 中配置）
    neo4j_uri: str = ""
    neo4j_user: str = ""
    neo4j_password: str = ""
    neo4j_enabled: bool = False

    # ── 数据库 schema 同步 ──
    db_auto_migrate: bool = Field(
        default=False,
        description="启动时是否自动同步 DB schema 到 ORM（进程内 diff 应用，不生成迁移文件）",
    )

    # 日志
    log_level: LogLevel = LogLevel.INFO

    # 开发模式：仅执行到 chunker，跳过后处理和 11 路检索
    index_only: bool = False

    # ── 删除 ──
    deletion_logical: bool = True   # AION_DELETION_LOGICAL：启用逻辑删除（只置标记不删数据）


settings = Settings()
