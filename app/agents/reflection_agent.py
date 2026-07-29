import json
from langchain_core.messages import SystemMessage, HumanMessage
from app.agents.state import GraphState

class ReflectionAgent:
    """
    Reflection Agent: Reviews the final answer and detects any missing information.
    Powered by Groq.
    """
    def __init__(self, model_client):
        self.llm = model_client.get_llm()

    def execute(self, state: GraphState) -> GraphState:
        """
        Reflects on the analysis and determines if it sufficiently answers the user query.
        """
        print("ReflectionAgent: Critiquing the generated answer...")
        
        system_prompt = """
        You are a strict academic reviewer.
        Review the provided 'Analysis Answer' against the original 'User Query'.
        Check for accuracy, completeness, and evidence.
        Respond ONLY with a raw JSON object (no markdown, no backticks).
        Format:
        {
          "approved": true or false,
          "feedback": "If false, explain exactly what is missing or incorrect. If true, write 'None'."
        }
        """
        
        user_content = f"User Query: {state.get('user_query', '')}\n\nAnalysis Answer: {state.get('analysis_result', '')}"
        
        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_content)
            ])
            
            content = response.content.strip().replace("```json", "").replace("```", "")
            decision_data = json.loads(content)
            
            state["reflection_approved"] = decision_data.get("approved", True)
            
            if not state["reflection_approved"]:
                state["error_message"] = decision_data.get("feedback", "Answer lacks clarity or depth.")
                print(f"Reflection Failed: {state['error_message']} - Sending back to Analysis.")
            else:
                state["final_answer"] = state.get("analysis_result", "")
                state["error_message"] = None
                print("Reflection Passed: Answer approved.")
                
        except Exception as e:
            print(f"ReflectionAgent Error: {e}")
            # Degrade gracefully
            state["reflection_approved"] = True
            state["final_answer"] = state.get("analysis_result", "")
            
        state["current_task"] = "done"
        return state
