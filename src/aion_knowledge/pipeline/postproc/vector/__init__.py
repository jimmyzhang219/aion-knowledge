"""向量嵌入模块。

计算 chunks 的 embedding 并写入 chunk_vector 表。
依赖 text 模块先执行拿到 chunk_uuid 和 content。
支持 Ollama / OpenAI 兼容 API。
"""
