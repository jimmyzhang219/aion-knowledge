"""graph_extract/extractor.py — 共享实体/关系 LLM 提取。

供 graph_extract / disambiguation / community 模块共享。
"""

import logging
from typing import Any

from aion_knowledge.common.config import settings
from aion_knowledge.common.model_registry import get_model_max_input_tokens, truncate_by_tokens

logger = logging.getLogger(__name__)

GRAPH_EXTRACT_PROMPT = """从以下文本中提取实体和关系。
实体类型: [{entity_types}]
以 JSON 格式输出：
{{
  "entities": [{{"name": "实体名", "type": "类型", "description": "描述"}}],
  "relations": [{{"source": "源实体", "target": "目标实体",
                  "type": "关系类型", "description": "关系描述",
                  "weight": 强度分1-10}}]
}}

文本：{content}"""

GLEAN_PROMPT = """以下文本中可能还有遗漏的实体和关系。补充提取缺失的实体和关系。
已有实体：{existing_entities_str}

文本：{content}

以相同 JSON 格式输出补充的实体和关系。"""

ENTITY_EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["name"],
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "type": {"type": "string"},
                    "description": {"type": "string"},
                    "weight": {"type": "number"},
                },
                "required": ["source", "target", "type"],
            },
        },
    },
    "required": ["entities", "relations"],
}


async def extract_entities_with_gleaning(
    llm: Any,
    content: str,
    entity_types: list[str],
    max_gleanings: int = 2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """共享 LLM 实体/关系提取（含 gleaning 多轮补充）。

    Args:
        llm: LLM 客户端
        content: 文本内容
        entity_types: 实体类型列表
        max_gleanings: 最多 gleaning 轮数

    Returns:
        (entities_list, relations_list) 元组
    """
    entity_types_str = ", ".join(entity_types)
    _max_tokens = get_model_max_input_tokens(settings.llm_model, ratio=0.03)
    prompt = GRAPH_EXTRACT_PROMPT.format(
        entity_types=entity_types_str,
        content=truncate_by_tokens(content, _max_tokens) if content else "",
    )
    try:
        result = await llm.generate_structured(prompt, output_schema=ENTITY_EXTRACTION_SCHEMA)
    except Exception:
        logger.warning("【实体提取】首次提取失败")
        return [], []
    if not isinstance(result, dict):
        # json_mode 不强制 schema：LLM 可能返回顶层 JSON 数组，跳过本轮
        logger.warning("【实体提取】LLM 输出非对象（%s），跳过", type(result).__name__)
        return [], []

    all_entities: dict[str, dict[str, Any]] = {}
    all_relations: dict[tuple[str, str, str], dict[str, Any]] = {}
    # LLM 输出 shape 不可信（json_mode 不强制 schema）：跳过非 dict 条目，
    # 与下方 gleaning 轮实体提取的 isinstance 防御保持一致
    for e in result.get("entities", []):
        if not isinstance(e, dict):
            continue
        name = e.get("name", "")
        if name:
            all_entities[name] = e
    for r in result.get("relations", []):
        if not isinstance(r, dict):
            continue
        key = (r.get("source", ""), r.get("target", ""), r.get("type", ""))
        if all(key):
            all_relations[key] = r

    # Gleaning 轮次
    for i in range(max_gleanings):
        existing = list(all_entities.keys())
        glean_prompt = GLEAN_PROMPT.format(
            existing_entities_str=", ".join(existing),
            content=truncate_by_tokens(content, _max_tokens) if content else "",
        )
        try:
            glean_result = await llm.generate_structured(
                glean_prompt, output_schema=ENTITY_EXTRACTION_SCHEMA
            )
        except Exception:
            break
        if not isinstance(glean_result, dict):
            break  # LLM 输出非对象（如顶层数组），放弃后续 gleaning
        new_entities = 0
        for e in glean_result.get("entities", []):
            if isinstance(e, str):
                ename = e
            elif isinstance(e, dict):
                ename = e.get("name", "")
            else:
                continue
            if not ename:
                continue
            if ename not in all_entities:
                all_entities[ename] = {"name": ename, "type": ""} if isinstance(e, str) else e
                new_entities += 1
        for r in glean_result.get("relations", []):
            if not isinstance(r, dict):
                continue  # LLM 可能返回字符串而非对象，跳过
            key = (r.get("source", ""), r.get("target", ""), r.get("type", ""))
            if not all(key):
                continue
            if key not in all_relations:
                all_relations[key] = r
        if new_entities == 0:
            break

    return list(all_entities.values()), list(all_relations.values())
