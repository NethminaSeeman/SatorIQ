from langchain_core.messages import SystemMessage, HumanMessage
from app.agents.state import GraphState
from app.utils.pipeline_helpers import is_non_retriable_answer
from app.utils.citation_helpers import format_labeled_chunks, format_source_index


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

        docs = state.get("retrieved_docs", [])
        chunks = state.get("retrieved_chunks", [])
        sources = state.get("retrieved_sources", [])
        chunks_text = format_labeled_chunks(docs) if docs else "\n\n".join(chunks)

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
        You are a Senior Academic Researcher helping with a fast literature review.
        Draft a comprehensive, well-structured answer to the user's query based ONLY on the provided research chunks.

        CITATION RULES (required for literature review):
        - Every factual claim, finding, or conclusion MUST end with an inline citation.
        - Use the exact label from each chunk header, e.g. [Source 1: paper.pdf, p.3] or [Source 2: other.pdf].
        - When comparing papers, state which paper supports each point explicitly.
        - Do not invent sources — cite only the Source labels provided in the chunks.

        After your main answer, add a section titled "## Paper-by-paper contributions" with one bullet per source:
        - **filename.pdf** — list the specific topics, findings, or claims this paper contributes to the answer.

        If critique/feedback is provided from a previous review, address it and improve your answer.
        Output your analysis directly in markdown format.
        """

        source_list = format_source_index(docs) if docs else "\n".join(
            f"- {source}" for source in sources[:8]
        )
        content_prompt = (
            f"User Query: {state.get('user_query', '')}\n\n"
            f"Available Sources:\n{source_list}\n\n"
            f"Retrieved Research Chunks (each labeled with its source):\n{chunks_text}"
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
