from app.agents.state import GraphState

class AnalysisAgent:
    """
    Analysis Agent: Compares research papers and generates deep insights.
    Powered by OpenRouter.
    """
    def __init__(self, model_client):
        self.model_client = model_client

    def execute(self, state: GraphState) -> GraphState:
        """
        Analyzes the summary and retrieved chunks to draw analytical conclusions.
        """
        print("AnalysisAgent: Performing deep cross-comparison of papers...")
        # Placeholder for actual OpenRouter call
        state["analysis_result"] = "Detailed analytical insights drawing connections between sources."
        state["current_task"] = "reflect"
        return state
