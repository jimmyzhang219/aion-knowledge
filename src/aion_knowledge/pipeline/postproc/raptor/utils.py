"""RAPTOR 常量、prompt 模板、工具函数。"""

from typing import Any

import xxhash

# 聚类方法常量
GMM_CLUSTERING_METHOD = "gmm"
AHC_CLUSTERING_METHOD = "ahc"
SUPPORTED_CLUSTERING_METHODS = {GMM_CLUSTERING_METHOD, AHC_CLUSTERING_METHOD}

# 输出模式
OUTPUT_MODE_FLAT = "flat"
OUTPUT_MODE_TREE = "tree"
SUPPORTED_OUTPUT_MODES = {OUTPUT_MODE_FLAT, OUTPUT_MODE_TREE}

# 结构化数据文件扩展名（自动禁用 RAPTOR）
STRUCTURED_EXTENSIONS = {".xls", ".xlsx", ".xlsm", ".xlsb", ".csv", ".tsv"}

# 默认摘要 prompt
DEFAULT_RAPTOR_PROMPT = (
    "请概括以下段落的核心内容。注意数字和关键信息，不要编造。"
    "段落如下：\n{cluster_content}"
)


def make_raptor_chunk_id(content: str, doc_id: str) -> str:
    """为 RAPTOR summary 生成稳定的 xxhash64 ID。"""
    return xxhash.xxh64((content + str(doc_id)).encode("utf-8")).hexdigest()


def should_skip_raptor(suffix: str | None = None, parser_id: str = "",
                       parser_config: dict[str, Any] | None = None) -> bool:
    """判断是否应跳过 RAPTOR（结构化数据、表格 PDF 等）。"""
    if not suffix:
        return False

    ft = suffix.lower()
    if not ft.startswith("."):
        ft = f".{ft}"

    if ft in STRUCTURED_EXTENSIONS:
        return True

    if ft in (".pdf", "pdf") and (parser_id == "table" or
                                   (parser_config or {}).get("html4excel", False)):
        return True

    return False


def get_skip_reason(suffix: str | None = None, parser_id: str = "",
                    parser_config: dict[str, Any] | None = None) -> str:
    """获取跳过 RAPTOR 的人类可读原因。"""
    if not suffix:
        return ""

    ft = suffix.lower()
    if not ft.startswith("."):
        ft = f".{ft}"

    if ft in STRUCTURED_EXTENSIONS:
        return f"结构化数据文件 ({ft}) — RAPTOR 自动禁用"

    if ft in (".pdf", "pdf") and (parser_id == "table" or
                                   (parser_config or {}).get("html4excel", False)):
        return f"表格 PDF (parser={parser_id}) — RAPTOR 自动禁用"

    return ""
