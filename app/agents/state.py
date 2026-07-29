from typing import Annotated, TypedDict, List, Optional, Any

# Maximum times Reflection may send work back to Analysis before forcing completion.
MAX_REFLECTION_RETRIES = 2


def last_wins(_left, right):
    """Reducer: last parallel update wins (LangGraph concurrent-safe)."""
    return right


class GraphState(TypedDict, total=False):
    """
    LangGraph shared state. Annotated fields support parallel Summary/Analysis nodes.
    """
    user_query: Annotated[str, last_wins]
    search_query: Annotated[str, last_wins]
    current_task: Annotated[str, last_wins]
    routing_decision: Annotated[str, last_wins]
    retrieved_chunks: Annotated[List[str], last_wins]
    retrieved_sources: Annotated[List[str], last_wins]
    retrieved_docs: Annotated[List[Any], last_wins]
    summary_result: Annotated[str, last_wins]
    analysis_result: Annotated[str, last_wins]
    reflection_approved: Annotated[bool, last_wins]
    reflection_retry_count: Annotated[int, last_wins]
    skip_reflection: Annotated[bool, last_wins]
    final_answer: Annotated[str, last_wins]
    error_message: Annotated[Optional[str], last_wins]
