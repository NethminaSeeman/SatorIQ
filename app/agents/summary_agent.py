from langchain_core.messages import SystemMessage, HumanMessage
from app.agents.state import GraphState
from app.utils.citation_helpers import format_labeled_chunks


class SummaryAgent:
    """
    Summary Agent: Summarizes the retrieved research papers and extracts key findings.
    Powered by Groq. Runs in parallel with Analysis — returns only summary_result.
    """
    def __init__(self, model_client):
        self.llm = model_client.get_llm()

    def execute(self, state: GraphState) -> dict:
        """Generates a summary from the retrieved chunks."""
        print("SummaryAgent: Generating summary from retrieved documents...")

        if state.get("skip_reflection"):
            existing = state.get("summary_result")
            return {"summary_result": existing or "No relevant documents were retrieved."}

        docs = state.get("retrieved_docs", [])
        chunks = state.get("retrieved_chunks", [])
        chunks_text = format_labeled_chunks(docs) if docs else "\n\n".join(chunks)
        if not chunks_text:
            return {"summary_result": "No relevant documents were retrieved."}

        system_prompt = """
        You are a highly efficient academic assistant.
        Summarize the following retrieved text chunks. Extract key findings, methodology, and conclusions.
        Note which source (Source 1, Source 2, etc.) each finding comes from.
        Be concise, accurate, and discard irrelevant information.
        """

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=chunks_text),
            ])
            return {"summary_result": response.content}
        except Exception as e:
            print(f"SummaryAgent Error: {e}")
            return {"summary_result": "Error generating summary."}
