"""Tests for create_embedder factory."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from aion_knowledge.infrastructure.embedder import Embedder, create_embedder


class TestCreateEmbedder:
    def test_returns_embedder_compatible_instance(self):
        """create_embedder() 返回的对象满足 Embedder Protocol。"""
        with patch("aion_knowledge.infrastructure.embedder.factory.settings") as ms:
            ms.embedding_base_url = "http://localhost:11434"
            ms.embedding_model = "bge-m3"
            ms.embedding_dimensions = 1024
            embedder = create_embedder()
            assert isinstance(embedder, Embedder)

    def test_raises_when_base_url_empty(self):
        """base_url 为空时抛出 ValueError。"""
        with patch("aion_knowledge.infrastructure.embedder.factory.settings") as ms:
            ms.embedding_base_url = ""
            with pytest.raises(ValueError, match="embedding_base_url"):
                create_embedder()
