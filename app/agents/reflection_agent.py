import json
from langchain_core.messages import SystemMessage, HumanMessage
from app.agents.state import GraphState, MAX_REFLECTION_RETRIES
from app.utils.pipeline_helpers import should_skip_reflection, reflection_retries_exhausted


class ReflectionAgent:
    """
    Reflection Agent: Reviews the final answer and detects any missing information.
    Powered by Groq.
    """
    def __init__(self, model_client):
        self.llm = model_client.get_llm()

    def _approve(self, state: GraphState) -> dict:
        return {
            "reflection_approved": True,
            "final_answer": state.get("analysis_result", "") or state.get("final_answer", ""),
            "error_message": None,
            "current_task": "done",
        }

    def execute(self, state: GraphState) -> dict:
        """Reflects on the analysis and determines if it sufficiently answers the user query."""
        print("ReflectionAgent: Critiquing the generated answer...")

        retry_count = state.get("reflection_retry_count", 0)

        if should_skip_reflection(state) or reflection_retries_exhausted(state):
            if reflection_retries_exhausted(state) and not state.get("reflection_approved"):
                print(
                    f"ReflectionAgent: Max retries ({MAX_REFLECTION_RETRIES}) reached — "
                    "approving best available answer."
                )
            return self._approve(state)

        system_prompt = """
        You are a strict academic reviewer.
        Review the provided 'Analysis Answer' and 'Summary' against the original 'User Query'.
        Check for accuracy, completeness, and evidence grounded in the retrieved research.
        If the Analysis Answer is an error message or states that no papers were found, approve it.
        Respond ONLY with a raw JSON object (no markdown, no backticks).
        Format:
        {
          "approved": true or false,
          "feedback": "If false, explain exactly what is missing or incorrect. If true, write 'None'."
        }
        """

        user_content = (
            f"User Query: {state.get('user_query', '')}\n\n"
            f"Summary: {state.get('summary_result', '')}\n\n"
            f"Analysis Answer: {state.get('analysis_result', '')}"
        )

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_content),
            ])

            content = response.content.strip().replace("```json", "").replace("```", "")
            decision_data = json.loads(content)
            approved = decision_data.get("approved", True)

            if approved:
                return self._approve(state)

            retry_count += 1
            print(
                f"Reflection Failed (attempt {retry_count}/{MAX_REFLECTION_RETRIES}): "
                f"{decision_data.get('feedback', 'Answer lacks clarity or depth.')}"
            )

            if retry_count >= MAX_REFLECTION_RETRIES:
                print("ReflectionAgent: Retry limit reached — approving current answer.")
                return self._approve(state)

            return {
                "reflection_approved": False,
                "reflection_retry_count": retry_count,
                "error_message": decision_data.get("feedback", "Answer lacks clarity or depth."),
                "current_task": "revise_analysis",
            }

        except Exception as e:
            print(f"ReflectionAgent Error: {e}")
            return self._approve(state)
