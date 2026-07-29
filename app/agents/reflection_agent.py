from app.agents.state import GraphState

class ReflectionAgent:
    """
    Reflection Agent: Reviews the final answer and detects any missing information.
    Powered by Groq.
    """
    def __init__(self, model_client):
        self.model_client = model_client

    def execute(self, state: GraphState) -> GraphState:
        """
        Reflects on the analysis and determines if it sufficiently answers the user query.
        """
        print("ReflectionAgent: Critiquing the generated answer...")
        # Placeholder for actual Groq call
        state["reflection_approved"] = True
        state["final_answer"] = f"Final Answer: {state.get('analysis_result')}"
        state["current_task"] = "done"
        return state
