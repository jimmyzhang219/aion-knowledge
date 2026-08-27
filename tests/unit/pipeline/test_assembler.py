"""Assembler 单元测试。"""

from aion_knowledge.pipeline.assembler import assemble


class TestAssemble:
    """验证 assemble 排序合并逻辑。"""

    def test_sorts_by_seq_num(self):
        text = [{"seq_num": 3, "content": "c"}, {"seq_num": 1, "content": "a"}]
        tables = [{"seq_num": 2, "content": "b"}]
        images = []

        result = assemble(text, tables, images)
        assert [c["content"] for c in result] == ["a", "b", "c"]

    def test_handles_empty_lists(self):
        assert assemble([], [], []) == []

    def test_includes_all_types(self):
        text = [{"seq_num": 0, "type": "text"}]
        tables = [{"seq_num": 1, "type": "table"}]
        images = [{"seq_num": 2, "type": "image"}]

        result = assemble(text, tables, images)
        assert len(result) == 3
