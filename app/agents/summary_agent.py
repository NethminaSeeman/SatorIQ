from langchain_core.messages import SystemMessage, HumanMessage
from app.agents.state import GraphState

class SummaryAgent:
    """
    Summary Agent: Summarizes the retrieved research papers and extracts key findings.
    Powered by Groq.
    """
    def __init__(self, model_client):
        self.llm = model_client.get_llm()

    def execute(self, state: GraphState) -> GraphState:
        """
        Generates a summary from the retrieved chunks.
        """
        print("SummaryAgent: Generating summary from retrieved documents...")
        
        system_prompt = """
        You are a highly efficient academic assistant.
        Summarize the following retrieved text chunks. Extract key findings, methodology, and conclusions.
        Be concise, accurate, and discard irrelevant information.
        """
        
        chunks_text = "\n\n".join(state.get("retrieved_chunks", []))
        
        if not chunks_text:
            state["summary_result"] = "No relevant documents were retrieved."
            state["current_task"] = "analyze"
            return state
            
        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=chunks_text)
            ])
            state["summary_result"] = response.content
        except Exception as e:
            print(f"SummaryAgent Error: {e}")
            state["summary_result"] = "Error generating summary."
            
        state["current_task"] = "analyze"
        return state
