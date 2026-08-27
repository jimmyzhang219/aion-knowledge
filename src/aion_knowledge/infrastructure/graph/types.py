"""Neo4j 数据模型。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Entity:
    entity_name: str
    entity_type: str = ""
    description: str = ""
    weight: float = 1.0
    source_docs: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class Relation:
    source_entity: str
    target_entity: str
    relation_type: str
    description: str = ""
    weight: float = 1.0
    source_docs: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)
