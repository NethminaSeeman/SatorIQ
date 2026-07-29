from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class TextChunker:
    """
    Responsible for splitting large documents into smaller chunks for embeddings.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits a list of documents into smaller chunks.
        
        Args:
            documents (List[Document]): The loaded documents.
            
        Returns:
            List[Document]: The chunked documents.
        """
        if not documents:
            return []
        
        return self.splitter.split_documents(documents)
