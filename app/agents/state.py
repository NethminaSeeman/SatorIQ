from typing import TypedDict, List, Dict, Any, Optional

class GraphState(TypedDict):
    """
    Represents the state of the agentic workflow in LangGraph.
    All agents will read from and write to this state dictionary.
    """
    user_query: str
    current_task: str
    routing_decision: str
    retrieved_chunks: List[str]
    retrieved_sources: List[str]
    summary_result: str
    analysis_result: str
    reflection_approved: bool
    final_answer: str
    error_message: Optional[str]
