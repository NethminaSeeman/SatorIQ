from langchain_community.embeddings import HuggingFaceEmbeddings

class EmbeddingsManager:
    """
    Manages the Sentence Transformers embeddings used for the vector store.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._embeddings = None

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Returns the HuggingFace embeddings model (lazy loads it).
        
        Returns:
            HuggingFaceEmbeddings: The embedding model instance.
        """
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        return self._embeddings
