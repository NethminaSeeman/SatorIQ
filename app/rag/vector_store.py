import os
from typing import List
from langchain_chroma import Chroma
from langchain_core.documents import Document
from app.rag.embeddings import EmbeddingsManager

class VectorStoreManager:
    """
    Manages the ChromaDB vector database for storing and querying document embeddings.
    """
    def __init__(self, persist_directory: str = "data/vector_db"):
        self.persist_directory = persist_directory
        self.embedding_manager = EmbeddingsManager()
        self.embeddings = self.embedding_manager.get_embeddings()
        
        # Initialize or load existing Chroma DB
        self.db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )

    def add_documents(self, documents: List[Document]) -> None:
        """
        Adds chunked documents to the vector store.
        
        Args:
            documents (List[Document]): The chunked documents to add.
        """
        if not documents:
            print("No documents to add to the vector store.")
            return
            
        print(f"Adding {len(documents)} chunks to ChromaDB at {self.persist_directory}...")
        self.db.add_documents(documents)
        print("Documents successfully added to the vector store.")

    def get_retriever(self, search_kwargs: dict = None):
        """
        Returns a LangChain retriever interface for the vector store.
        
        Args:
            search_kwargs (dict): Search arguments like 'k' (top_k). Defaults to {'k': 4}.
            
        Returns:
            VectorStoreRetriever: The configured LangChain retriever.
        """
        if search_kwargs is None:
            search_kwargs = {"k": 4}
        return self.db.as_retriever(search_kwargs=search_kwargs)
