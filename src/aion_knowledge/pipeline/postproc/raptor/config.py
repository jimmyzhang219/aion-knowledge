"""RAPTOR 模块专属配置，从 AION_RAPTOR_* 环境变量加载。

用法::

    cfg = RaptorConfig()
    print(cfg.prompt)

与全局 ``Settings`` 的关系：``postproc_raptor`` 开关与树遍历检索参数
（``raptor_traverse_*``，RAPTORRetriever 使用，位于 ``common/config.py``）
在全局 config 中；其余 ``raptor_*`` 字段在此处。
推荐在 ``.env`` 中以 ``AION_RAPTOR_*`` 形式配置。
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class RaptorConfig(BaseSettings):
    """RAPTOR 模块配置，从 AION_RAPTOR_* 环境变量加载。"""

    model_config = SettingsConfigDict(env_prefix="AION_RAPTOR_", extra="ignore")

    prompt: str = (
        "请概括以下段落的核心内容。注意数字和关键信息，不要编造。"
        "段落如下：\n{cluster_content}"
    )
    max_token: int = 256
    threshold: float = 0.1
    max_cluster: int = 64
    random_seed: int = 0
    scope: str = "file"
    clustering_method: str = "gmm"
    output_mode: str = "flat"
    small_layer_collapse: int = 8
    max_errors: int = 3


raptor_config = RaptorConfig()
"""模块级单例，processor 和 core 直接引用。"""
