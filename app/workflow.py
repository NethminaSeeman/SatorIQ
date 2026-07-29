from langgraph.graph import StateGraph, END
from app.agents.state import GraphState
from app.agents.router_agent import RouterAgent
from app.agents.retriever_agent import RetrieverAgent
from app.agents.summary_agent import SummaryAgent
from app.agents.analysis_agent import AnalysisAgent
from app.agents.reflection_agent import ReflectionAgent
from app.models.groq_client import GroqClient
from app.models.openrouter_client import OpenRouterClient
from app.rag.retriever import RAGRetriever
from app.utils.pipeline_helpers import should_skip_reflection, reflection_retries_exhausted


def route_next(state: GraphState):
    """Conditional routing based on the Router's decision."""
    if state.get("routing_decision") == "retrieve":
        return "retriever"
    return END


def route_after_retrieval(state: GraphState):
    """
    Empty index → Reflection only.
    Otherwise fan out Summary + Analysis in parallel (assignment architecture).
    """
    if state.get("skip_reflection"):
        return "reflection"
    return ["summary", "analysis"]


def route_after_analysis(state: GraphState):
    """After revision loop, skip join and go straight back to Reflection."""
    if state.get("reflection_retry_count", 0) > 0:
        return "reflection"
    return "join"


def route_reflection(state: GraphState):
    """Conditional routing based on Reflection Agent's critique."""
    if state.get("reflection_approved"):
        return END
    if should_skip_reflection(state) or reflection_retries_exhausted(state):
        return END
    return "analysis"


def join_workers(_state: GraphState) -> dict:
    """
    Barrier node — runs once after BOTH Summary and Analysis finish.
    Prevents Reflection from firing twice on parallel edges.
    """
    return {"current_task": "reflect"}


def build_workflow():
    """
    LangGraph workflow matching assignment architecture:

        Router → Retriever → ChromaDB → Retrieved Chunks
                                      ├→ Summary  ─┐
                                      └→ Analysis ─┴→ Join → Reflection → Response
    """
    groq_client = GroqClient()
    openrouter_client = OpenRouterClient()
    rag_retriever = RAGRetriever()

    router = RouterAgent(groq_client)
    retriever = RetrieverAgent(rag_retriever)
    summary = SummaryAgent(groq_client)
    analysis = AnalysisAgent(openrouter_client)
    reflection = ReflectionAgent(groq_client)

    workflow = StateGraph(GraphState)

    workflow.add_node("router", router.execute)
    workflow.add_node("retriever", retriever.execute)
    workflow.add_node("summary", summary.execute)
    workflow.add_node("analysis", analysis.execute)
    workflow.add_node("join", join_workers)
    workflow.add_node("reflection", reflection.execute)

    workflow.set_entry_point("router")
    workflow.add_conditional_edges(
        "router",
        route_next,
        {"retriever": "retriever", END: END},
    )
    workflow.add_conditional_edges(
        "retriever",
        route_after_retrieval,
        ["summary", "analysis", "reflection"],
    )
    workflow.add_edge("summary", "join")
    workflow.add_conditional_edges(
        "analysis",
        route_after_analysis,
        {"join": "join", "reflection": "reflection"},
    )
    workflow.add_edge("join", "reflection")
    workflow.add_conditional_edges(
        "reflection",
        route_reflection,
        {"analysis": "analysis", END: END},
    )

    return workflow.compile()
