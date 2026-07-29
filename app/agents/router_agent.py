from app.agents.state import GraphState

class RouterAgent:
    """
    Router Agent: Understands the user request and decides which agent workflow to execute next.
    Powered by Groq.
    """
    def __init__(self, model_client):
        self.model_client = model_client

    def execute(self, state: GraphState) -> GraphState:
        """
        Processes the current state and sets the routing_decision.
        """
        print("RouterAgent: Analyzing query to determine next steps...")
        # Placeholder for actual Groq call
        state["routing_decision"] = "retrieve"
        state["current_task"] = "retrieve"
        return state
