from langchain_core.messages import SystemMessage, HumanMessage
from app.agents.state import GraphState
from app.utils.pipeline_helpers import is_non_retriable_answer


class AnalysisAgent:
    """
    Analysis Agent: Compares research papers and generates deep insights.
    Powered by OpenRouter. Runs in parallel with Summary — returns only analysis fields.
    """
    def __init__(self, model_client):
        self.llm = model_client.get_llm()

    def execute(self, state: GraphState) -> dict:
        """Analyzes retrieved chunks and draws analytical conclusions."""
        print("AnalysisAgent: Performing deep cross-comparison of papers...")

        if state.get("skip_reflection") and state.get("analysis_result"):
            return {"analysis_result": state["analysis_result"]}

        chunks = state.get("retrieved_chunks", [])
        chunks_text = "\n\n".join(chunks)
        sources = state.get("retrieved_sources", [])

        if not chunks_text:
            message = (
                "I could not find any relevant papers in the knowledge base. "
                "Please add PDF research papers to `data/raw_papers/` and click "
                "**Build Vector Index** in the sidebar, then ask your question again."
            )
            return {
                "analysis_result": message,
                "skip_reflection": True,
                "final_answer": message,
            }

        system_prompt = """
        You are a Senior Academic Researcher.
        Draft a comprehensive, well-structured answer to the user's query based ONLY on the provided research chunks.
        Compare and synthesize the findings across sources. Cite themes from the retrieved text only.
        If critique/feedback is provided from a previous review, address it and improve your answer.
        Output your analysis directly in markdown format.
        """

        source_list = "\n".join(f"- {source}" for source in sources[:8])
        content_prompt = (
            f"User Query: {state.get('user_query', '')}\n\n"
            f"Retrieved Sources:\n{source_list}\n\n"
            f"Retrieved Research Chunks:\n{chunks_text}"
        )

        if state.get("error_message"):
            content_prompt += (
                f"\n\nPrevious Reviewer Feedback (MUST ADDRESS): {state.get('error_message')}"
            )

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=content_prompt),
            ])
            analysis_result = response.content
        except Exception as e:
            print(f"AnalysisAgent Error: {e}")
            analysis_result = (
                f"Error performing analysis: {str(e)}\n\n"
                "Please ensure your OPENROUTER_API_KEY is valid and has credits."
            )
            return {
                "analysis_result": analysis_result,
                "skip_reflection": True,
                "final_answer": analysis_result,
            }

        updates: dict = {"analysis_result": analysis_result}
        if is_non_retriable_answer(analysis_result):
            updates["skip_reflection"] = True
            updates["final_answer"] = analysis_result

        return updates
