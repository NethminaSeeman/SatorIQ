import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

class PDFLoader:
    """
    Responsible for loading PDF documents from the raw papers directory.
    """
    def __init__(self, directory_path: str = "data/raw_papers"):
        self.directory_path = directory_path

    def load_all_pdfs(self) -> List[Document]:
        """
        Loads all PDF files from the configured directory.
        
        Returns:
            List[Document]: A list of LangChain Document objects containing the text and metadata.
        """
        documents = []
        if not os.path.exists(self.directory_path):
            os.makedirs(self.directory_path)
            return documents
            
        for filename in os.listdir(self.directory_path):
            if filename.endswith(".pdf"):
                file_path = os.path.join(self.directory_path, filename)
                try:
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    documents.extend(docs)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    
        return documents
