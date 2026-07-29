from langchain_core.messages import SystemMessage, HumanMessage
from app.agents.state import GraphState

class AnalysisAgent:
    """
    Analysis Agent: Compares research papers and generates deep insights.
    Powered by OpenRouter.
    """
    def __init__(self, model_client):
        self.llm = model_client.get_llm()

    def execute(self, state: GraphState) -> GraphState:
        """
        Analyzes the summary and retrieved chunks to draw analytical conclusions.
        """
        print("AnalysisAgent: Performing deep cross-comparison of papers...")
        
        system_prompt = """
        You are a Senior Academic Researcher.
        Draft a comprehensive, well-structured answer to the user's query based ONLY on the provided research summaries.
        Compare and synthesize the findings. If critique/feedback is provided from a previous review, you must address it and improve your answer.
        Output your analysis directly in markdown format.
        """
        
        content_prompt = f"User Query: {state.get('user_query', '')}\n\nResearch Summary: {state.get('summary_result', '')}"
        
        if state.get("error_message"):
            content_prompt += f"\n\nPrevious Reviewer Feedback (MUST ADDRESS): {state.get('error_message')}"
            
        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=content_prompt)
            ])
            state["analysis_result"] = response.content
        except Exception as e:
            print(f"AnalysisAgent Error: {e}")
            state["analysis_result"] = "Error performing analysis. Please check your OpenRouter API Key."
            
        state["current_task"] = "reflect"
        return state
