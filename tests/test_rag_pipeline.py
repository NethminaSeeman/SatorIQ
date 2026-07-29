"""Basic tests for the SatorIQ RAG pipeline."""

import os
import tempfile
import shutil

import pytest
from langchain_core.documents import Document

from app.rag.chunker import TextChunker
from app.rag.embeddings import EmbeddingsManager


class TestTextChunker:
    """Tests for document chunking."""

    def test_empty_documents_returns_empty_list(self):
        chunker = TextChunker()
        assert chunker.chunk_documents([]) == []

    def test_single_document_is_split_into_chunks(self):
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        long_text = "Explainable AI in healthcare. " * 50
        docs = [Document(page_content=long_text, metadata={"source": "test.pdf"})]

        chunks = chunker.chunk_documents(docs)

        assert len(chunks) > 1
        assert all(isinstance(c, Document) for c in chunks)

    def test_small_document_produces_at_least_one_chunk(self):
        chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
        docs = [Document(page_content="Short research abstract.", metadata={"source": "test.pdf"})]

        chunks = chunker.chunk_documents(docs)

        assert len(chunks) >= 1
        assert "Short research abstract." in chunks[0].page_content


class TestEmbeddingsManager:
    """Tests for Sentence Transformers embeddings initialization."""

    def test_embeddings_lazy_load(self):
        manager = EmbeddingsManager()
        assert manager._embeddings is None

        embeddings = manager.get_embeddings()

        assert embeddings is not None
        assert manager._embeddings is not None

    def test_embeddings_produce_vector(self):
        manager = EmbeddingsManager()
        embeddings = manager.get_embeddings()

        vector = embeddings.embed_query("Explainable AI in healthcare")

        assert isinstance(vector, list)
        assert len(vector) > 0
        assert all(isinstance(v, float) for v in vector)


class TestVectorStore:
    """Tests for ChromaDB vector store add and search."""

    @pytest.fixture
    def temp_db_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_add_and_search_documents(self, temp_db_dir):
        from app.rag.vector_store import VectorStoreManager

        store = VectorStoreManager(persist_directory=temp_db_dir)
        docs = [
            Document(page_content="Explainable AI improves trust in clinical decisions.", metadata={"source": "paper1.pdf"}),
            Document(page_content="Deep learning models lack interpretability in medical imaging.", metadata={"source": "paper2.pdf"}),
        ]
        store.add_documents(docs)

        retriever = store.get_retriever(search_kwargs={"k": 1})
        results = retriever.invoke("Explainable AI in healthcare")

        assert len(results) >= 1
        assert "Explainable AI" in results[0].page_content
