from app.agents.state import GraphState
from app.rag.retriever import RAGRetriever


class RetrieverAgent:
    """
    Retriever Agent: Interfaces with the RAG pipeline to search ChromaDB and fetch chunks.
    """
    def __init__(self, retriever_facade: RAGRetriever):
        self.retriever = retriever_facade

    def execute(self, state: GraphState) -> dict:
        """Retrieves relevant documents based on the user query."""
        query = state.get("search_query") or state.get("user_query", "")
        print(f"RetrieverAgent: Searching Vector DB for '{query}'...")

        results = self.retriever.retrieve(query)
        chunks = results.get("chunks", [])
        sources = results.get("sources", [])
        docs = results.get("docs", [])

        updates: dict = {
            "retrieved_chunks": chunks,
            "retrieved_sources": sources,
            "retrieved_docs": docs,
        }

        if not chunks:
            message = (
                "I could not find any relevant papers in the knowledge base. "
                "Please add PDF research papers to `data/raw_papers/` and click "
                "**Build Vector Index** in the sidebar, then ask your question again."
            )
            updates.update({
                "skip_reflection": True,
                "summary_result": "No relevant documents were retrieved.",
                "analysis_result": message,
                "final_answer": message,
            })

        return updates
