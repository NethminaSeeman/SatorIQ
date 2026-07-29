from app.agents.state import GraphState

class SummaryAgent:
    """
    Summary Agent: Summarizes the retrieved research papers and extracts key findings.
    Powered by Groq.
    """
    def __init__(self, model_client):
        self.model_client = model_client

    def execute(self, state: GraphState) -> GraphState:
        """
        Generates a summary from the retrieved chunks.
        """
        print("SummaryAgent: Generating summary from retrieved documents...")
        # Placeholder for actual Groq call
        state["summary_result"] = "Initial summary of research findings based on chunks."
        state["current_task"] = "analyze"
        return state
