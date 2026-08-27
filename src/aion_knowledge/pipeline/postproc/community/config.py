"""Community 模块专属配置，从 AION_COMMUNITY_* 环境变量加载。"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class CommunityConfig(BaseSettings):
    """社区发现模块配置，从 AION_COMMUNITY_* 环境变量加载。"""

    model_config = SettingsConfigDict(env_prefix="AION_COMMUNITY_", extra="ignore")

    max_cluster_size: int = 100
    """每个 community 的最大节点数，超过则递归聚类。"""

    enable_checkpoint: bool = True
    """启用图 hash 检查点，图未变则跳过社区检测。"""


community_config = CommunityConfig()
"""模块级单例，processor 直接引用。"""
