"""验证调度器能自动发现所有模块。"""

from aion_knowledge.pipeline.postproc.dispatcher import PostProcDispatcher


def test_all_modules_discovered():
    """PostProcDispatcher 应自动发现所有模块（TagModule 已合并至 KeywordExtractModule）。"""
    d = PostProcDispatcher({})
    names = list(d._modules.keys())
    print(f"Discovered modules: {names}")
    assert len(names) == 11, f"Expected 11 modules, got {len(names)}: {names}"
    assert "text" in names
    assert "vector" in names
    assert "keyword_extract" in names
    assert "question_gen" in names
    assert "summarizer" in names
    assert "graph_extract" in names
    assert "raptor" in names
    assert "disambiguation" in names
    assert "community" in names
    assert "wiki" in names
