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

def route_next(state: GraphState):
    """Conditional routing based on the Router's decision."""
    if state.get("routing_decision") == "retrieve":
        return "retriever"
    else:
        # If no retrieval needed, we could route directly to END or a direct answering agent.
        # For this assignment, we mostly pursue the full RAG pipeline.
        return END

def route_reflection(state: GraphState):
    """Conditional routing based on Reflection Agent's critique."""
    if state.get("reflection_approved"):
        return END
    else:
        # Loop back to Analysis for improvement
        return "analysis"

def build_workflow():
    """
    Constructs the LangGraph state machine orchestrating all agents.
    """
    # Initialize Models & Tools
    groq_client = GroqClient()
    openrouter_client = OpenRouterClient()
    rag_retriever = RAGRetriever()
    
    # Initialize Agents
    router = RouterAgent(groq_client)
    retriever = RetrieverAgent(rag_retriever)
    summary = SummaryAgent(groq_client)
    analysis = AnalysisAgent(openrouter_client)
    reflection = ReflectionAgent(groq_client)
    
    # Define Graph
    workflow = StateGraph(GraphState)
    
    # Add Nodes
    workflow.add_node("router", router.execute)
    workflow.add_node("retriever", retriever.execute)
    workflow.add_node("summary", summary.execute)
    workflow.add_node("analysis", analysis.execute)
    workflow.add_node("reflection", reflection.execute)
    
    # Add Edges
    workflow.set_entry_point("router")
    workflow.add_conditional_edges(
        "router",
        route_next,
        {"retriever": "retriever", END: END}
    )
    workflow.add_edge("retriever", "summary")
    workflow.add_edge("summary", "analysis")
    workflow.add_edge("analysis", "reflection")
    workflow.add_conditional_edges(
        "reflection",
        route_reflection,
        {"analysis": "analysis", END: END}
    )
    
    # Compile
    return workflow.compile()
