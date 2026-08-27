"""Disambiguation 模块专属配置，从 AION_DISAMBIGUATION_* 环境变量加载。"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class DisambiguationConfig(BaseSettings):
    """实体消歧模块配置，从 AION_DISAMBIGUATION_* 环境变量加载。"""

    model_config = SettingsConfigDict(
        env_prefix="AION_DISAMBIGUATION_", extra="ignore"
    )

    edit_distance_threshold: int = 3
    """英文实体编辑距离阈值，≤ 此值判定为候选对。"""

    jaccard_threshold: float = 0.7
    """CJK 实体字符级 Jaccard 相似度阈值。"""

    batch_size: int = 100
    """每批发送给 LLM 的候选对数。"""


disambiguation_config = DisambiguationConfig()
"""模块级单例，processor 直接引用。"""
