from app.agents.state import GraphState
from app.rag.retriever import RAGRetriever

class RetrieverAgent:
    """
    Retriever Agent: Interfaces with the RAG pipeline to search ChromaDB and fetch chunks.
    """
    def __init__(self, retriever_facade: RAGRetriever):
        self.retriever = retriever_facade

    def execute(self, state: GraphState) -> GraphState:
        """
        Retrieves relevant documents based on the user query.
        """
        print(f"RetrieverAgent: Searching Vector DB for '{state.get('user_query')}'...")
        # Utilize the Facade from Feature 2
        results = self.retriever.retrieve(state.get("user_query"))
        
        state["retrieved_chunks"] = results.get("chunks", [])
        state["retrieved_sources"] = results.get("sources", [])
        state["current_task"] = "summarize"
        return state
