"""Graph — Neo4j 图谱操作门面。"""

from aion_knowledge.infrastructure.graph.client import Neo4jConnection
from aion_knowledge.infrastructure.graph.reader import (
    expand_neighbors,
    load_kb_graph,
    search_entities,
)
from aion_knowledge.infrastructure.graph.writer import (
    add_graph,
    delete_document_graph,
    delete_graph,
    get_stats,
    merge_aliases,
    merge_entities,
)

__all__ = [
    "Neo4jConnection",
    "add_graph",
    "merge_entities",
    "merge_aliases",
    "delete_graph",
    "delete_document_graph",
    "get_stats",
    "search_entities",
    "expand_neighbors",
    "load_kb_graph",
]
