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

    def execute(self, state: GraphState) -> dict:
        """Processes the current state and sets the routing_decision."""
        print("RouterAgent: Analyzing query to determine next steps...")

        updates: dict = {
            "routing_decision": "retrieve",
            "current_task": "retrieve",
        }

        system_prompt = """
        You are the Router Agent for an academic research assistant.
        Analyze the user query and decide if it needs document retrieval.
        Respond ONLY with a raw JSON object (no markdown, no backticks).
        Format: {"task": "retrieve" or "direct", "query": "optimized search query if retrieve, else empty"}
        """

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=state.get("user_query", "")),
            ])

            content = response.content.strip().replace("```json", "").replace("```", "")
            decision_data = json.loads(content)

            task = decision_data.get("task", "retrieve")
            updates["routing_decision"] = task
            updates["current_task"] = task

            # Keep original user_query; store optimized text separately for retrieval.
            if decision_data.get("query"):
                updates["search_query"] = decision_data.get("query")

        except Exception as e:
            print(f"RouterAgent Error: {e}")

        print(f"Router Decision: {updates['routing_decision']}")
        return updates
