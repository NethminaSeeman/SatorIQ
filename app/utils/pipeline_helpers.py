"""Shared helpers for workflow routing and terminal-state detection."""

from app.agents.state import GraphState, MAX_REFLECTION_RETRIES


def is_non_retriable_answer(analysis: str) -> bool:
    """
    Detect analysis outputs that should not trigger a Reflection retry loop.
    Used when the corpus is empty, the index is missing, or an API error occurred.
    """
    if not analysis:
        return False

    markers = (
        "could not find any relevant papers",
        "error performing analysis",
        "please make sure you have added pdf",
        "build vector index",
        "no relevant documents were retrieved",
        "no documents found",
        "vector index",
    )
    lower = analysis.lower()
    return any(marker in lower for marker in markers)


def should_skip_reflection(state: GraphState) -> bool:
    """Return True when Reflection should approve immediately without retrying."""
    if state.get("skip_reflection"):
        return True
    if not state.get("retrieved_chunks"):
        return True
    return is_non_retriable_answer(state.get("analysis_result", ""))


def reflection_retries_exhausted(state: GraphState) -> bool:
    """Return True when the Reflection→Analysis loop must stop."""
    return state.get("reflection_retry_count", 0) >= MAX_REFLECTION_RETRIES
