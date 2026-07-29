from typing import List, Dict, Any
from app.rag.loader import PDFLoader
from app.rag.chunker import TextChunker
from app.rag.vector_store import VectorStoreManager

class RAGRetriever:
    """
    A facade class that wraps the individual RAG components (Loader, Chunker, Vector Store)
    and exposes a simple interface for the Retriever Agent.
    """
    def __init__(self):
        self.vector_store = VectorStoreManager()

    def rebuild_index(self) -> None:
        """
        Helper method to reload all PDFs from disk, chunk them, and add to the vector database.
        Typically run once or whenever the knowledge base needs updating.
        """
        loader = PDFLoader()
        chunker = TextChunker()
        
        print("Loading PDFs...")
        documents = loader.load_all_pdfs()
        
        if not documents:
            print("No documents found to process.")
            return
            
        print(f"Loaded {len(documents)} raw documents. Chunking...")
        chunks = chunker.chunk_documents(documents)
        
        self.vector_store.add_documents(chunks)
        print("Index rebuild complete.")

    def retrieve(self, query: str, top_k: int = 4) -> Dict[str, Any]:
        """
        Queries the vector store for the most relevant document chunks.
        Designed to return a structured dictionary as required by the multi-agent system.
        
        Args:
            query (str): The search query.
            top_k (int): Number of chunks to retrieve.
            
        Returns:
            dict: Structured dictionary containing the retrieved chunks and their sources.
        """
        retriever = self.vector_store.get_retriever(search_kwargs={"k": top_k})
        docs = retriever.invoke(query)
        
        # Structure the results for the Retriever Agent
        result = {
            "chunks": [doc.page_content for doc in docs],
            "sources": [doc.metadata.get("source", "Unknown Source") for doc in docs]
        }
        return result
