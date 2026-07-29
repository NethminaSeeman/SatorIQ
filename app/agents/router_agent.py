import json
from langchain_core.messages import SystemMessage, HumanMessage
from app.agents.state import GraphState

class RouterAgent:
    """
    Router Agent: Understands the user request and decides which agent workflow to execute next.
    Powered by Groq.
    """
    def __init__(self, model_client):
        self.llm = model_client.get_llm()

    def execute(self, state: GraphState) -> GraphState:
        """
        Processes the current state and sets the routing_decision.
        """
        print("RouterAgent: Analyzing query to determine next steps...")
        
        system_prompt = """
        You are the Router Agent for an academic research assistant.
        Analyze the user query and decide if it needs document retrieval.
        Respond ONLY with a raw JSON object (no markdown, no backticks).
        Format: {"task": "retrieve" or "direct", "query": "optimized search query if retrieve, else empty"}
        """
        
        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=state.get("user_query", ""))
            ])
            
            # Remove any possible markdown blocks if the LLM hallucinated them
            content = response.content.strip().replace("```json", "").replace("```", "")
            decision_data = json.loads(content)
            
            state["routing_decision"] = decision_data.get("task", "retrieve")
            state["current_task"] = decision_data.get("task", "retrieve")
            # We can optionally override the query if it optimized it for search
            if decision_data.get("query"):
                state["user_query"] = decision_data.get("query")
                
        except Exception as e:
            print(f"RouterAgent Error: {e}")
            # Fallback
            state["routing_decision"] = "retrieve"
            state["current_task"] = "retrieve"
            
        print(f"Router Decision: {state['routing_decision']}")
        return state
