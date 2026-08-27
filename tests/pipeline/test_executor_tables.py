"""测试：assemble 交错合并。"""
from aion_knowledge.pipeline.assembler import assemble


class TestAssembleChunks:
    """_assemble() 测试。"""

    def make_text_chunk(self, seq: int, text: str = "") -> dict:
        return {"content": text or f"text_{seq}", "chunk_type": "text", "seq_num": seq}

    def make_table_chunk(self, seq: int, content: str = "") -> dict:
        return {
            "type": "table",
            "content": content or f"table_{seq}",
            "chunk_type": "table",
            "heading_path": [],
            "table_caption": "",
            "seq_num": seq,
        }

    def make_image_chunk(self, seq: int) -> dict:
        return {
            "content": "",
            "chunk_type": "image",
            "image_url": f"https://example.com/img{seq}.png",
            "context_above": "",
            "context_below": "",
            "token_count": 0,
            "seq_num": seq,
        }

    def test_assemble_chunks_interleaved(self):
        """验证 text/table/image 交错排列，按 seq_num 有序。"""
        text_chunks = [
            self.make_text_chunk(0),
            self.make_text_chunk(3),
            self.make_text_chunk(6),
        ]
        table_chunks = [
            self.make_table_chunk(1),
            self.make_table_chunk(4),
        ]
        image_chunks = [
            self.make_image_chunk(2),
            self.make_image_chunk(5),
        ]

        result = assemble(text_chunks, table_chunks, image_chunks)

        assert len(result) == 7
        seqs = [c["seq_num"] for c in result]
        assert seqs == sorted(seqs), f"Expected sorted seq_nums, got {seqs}"

        # 验证类型交错
        assert result[0]["chunk_type"] == "text"
        assert result[1]["chunk_type"] == "table"
        assert result[2]["chunk_type"] == "image"
        assert result[3]["chunk_type"] == "text"
        assert result[4]["chunk_type"] == "table"
        assert result[5]["chunk_type"] == "image"
        assert result[6]["chunk_type"] == "text"

    def test_assemble_chunks_empty(self):
        """无 table/image 时正常返回 text chunks。"""
        text_chunks = [
            self.make_text_chunk(0),
            self.make_text_chunk(1),
        ]

        result = assemble(text_chunks, [], [])

        assert len(result) == 2
        assert all(c["chunk_type"] == "text" for c in result)
        assert [c["seq_num"] for c in result] == [0, 1]

    def test_assemble_chunks_no_text(self):
        """仅有 table 和 image chunks。"""
        table_chunks = [self.make_table_chunk(0)]
        image_chunks = [self.make_image_chunk(1)]

        result = assemble([], table_chunks, image_chunks)

        assert len(result) == 2
        assert result[0]["chunk_type"] == "table"
        assert result[1]["chunk_type"] == "image"
