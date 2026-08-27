"""GraphExtract 模块专属配置，从 AION_GRAPH_* 环境变量加载。

用法::

    cfg = GraphExtractConfig()
    print(cfg.max_concurrent)

与全局 ``Settings`` 的关系：``postproc_graph_extract`` 开关仍在全局 config 中，
所有 ``graph_*`` 参数搬到此处。推荐在 ``.env`` 中以 ``AION_GRAPH_*`` 形式配置。
"""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GraphExtractConfig(BaseSettings):
    """图谱提取模块配置，从 AION_GRAPH_* 环境变量加载。"""

    model_config = SettingsConfigDict(env_prefix="AION_GRAPH_", extra="ignore")

    max_concurrent: int = 5
    """LLM 并发提取的 chunk 数。高并发提升速度但受 API rate limit 约束。"""

    max_gleanings: int = 2
    """每 chunk gleaning 轮数。0 = 不 gleaning（单轮），1-2 = 推荐。"""

    entity_types: list[str] = Field(default_factory=list)
    """限定提取的实体类型。空列表 = 提取全部类型。"""


graph_config = GraphExtractConfig()
"""模块级单例，processor 直接引用。"""
