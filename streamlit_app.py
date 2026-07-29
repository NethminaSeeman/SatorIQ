import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from app.workflow import build_workflow
from app.agents.state import GraphState
from app.utils.citation_helpers import build_paper_contributions, cited_sources_in_answer

st.set_page_config(page_title="SatorIQ Assistant", page_icon="🧠", layout="wide")

SAMPLE_QUERIES = [
    "What are the challenges of Explainable AI in healthcare?",
    "How does XAI improve clinical decision-making?",
    "Compare accuracy vs interpretability in medical AI.",
]

if "pending_query" not in st.session_state:
    st.session_state["pending_query"] = None

# Sidebar
with st.sidebar:
    st.title("🧠 SatorIQ")
    st.markdown("Agentic AI Research Paper Screening & Literature Assistant")
    st.divider()
    st.markdown("**Agents Active:**")
    st.markdown("- 🚦 Router (Groq)")
    st.markdown("- 🔍 Retriever (RAG)")
    st.markdown("- 📝 Summary (Groq)")
    st.markdown("- 🧠 Analysis (OpenRouter)")
    st.markdown("- 🤔 Reflection (Groq)")
    st.divider()

    # Check for API Keys
    if not os.environ.get("GROQ_API_KEY"):
        st.error("Missing GROQ_API_KEY")
    if not os.environ.get("OPENROUTER_API_KEY"):
        st.error("Missing OPENROUTER_API_KEY")

    st.divider()
    st.markdown("**Knowledge Base**")

    @st.cache_resource(show_spinner=False)
    def get_index_chunk_count() -> int:
        from app.rag.vector_store import VectorStoreManager
        return VectorStoreManager().get_chunk_count()

    chunk_count = get_index_chunk_count()
    if chunk_count > 0:
        st.success(f"Vector index ready ({chunk_count:,} chunks indexed)")
    else:
        st.error("Vector index is empty — upload PDFs below and build the index.")

    uploaded_pdfs = st.file_uploader(
        "Upload research PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Optional: add or replace papers, then rebuild the index.",
    )
    if uploaded_pdfs:
        os.makedirs("data/raw_papers", exist_ok=True)
        saved = 0
        for uploaded in uploaded_pdfs:
            dest = os.path.join("data/raw_papers", uploaded.name)
            with open(dest, "wb") as f:
                f.write(uploaded.getbuffer())
            saved += 1
        st.info(f"Saved {saved} PDF(s) to data/raw_papers/.")

    st.divider()
    if st.button("🏗️ Build Vector Index (Run Once)"):
        with st.spinner("Reading PDFs, chunking, and embedding... this may take a minute."):
            from app.rag.retriever import RAGRetriever
            try:
                rag = RAGRetriever()
                rag.rebuild_index()
                get_index_chunk_count.clear()
                st.success("Vector DB index built successfully! You can now query papers.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to build index: {e}")

    st.divider()
    st.markdown("**Try a sample question**")
    for sample in SAMPLE_QUERIES:
        if st.button(sample, key=f"sample_{sample[:24]}"):
            st.session_state["pending_query"] = sample
            st.rerun()

st.title("Research Assistant")
st.write("Ask questions about the uploaded research papers, and watch the agents collaborate to find the answer.")

query = st.chat_input("Enter your research question here...")
if not query and st.session_state.get("pending_query"):
    query = st.session_state.pop("pending_query")

if query:
    # Display user's question
    st.chat_message("user").write(query)
    
    # Build LangGraph Application
    try:
        app = build_workflow()
    except Exception as e:
        st.error(f"Failed to initialize workflow: {e}")
        st.stop()
        
    initial_state = {
        "user_query": query,
        "reflection_retry_count": 0,
        "skip_reflection": False,
        "reflection_approved": False,
    }
    
    # Execute workflow and show progress
    with st.chat_message("assistant"):
        status_box = st.status("Agents are thinking...", expanded=True)
        final_answer = ""
        retrieved_sources: list[str] = []
        retrieved_docs: list[dict] = []
        
        try:
            # We use stream() to catch events as they happen from each node
            for output in app.stream(initial_state):
                # output is a dict like {'router': {'routing_decision': 'retrieve', ...}}
                for key, value in output.items():
                    agent_name = key.capitalize()
                    
                    if key == "router":
                        decision = value.get("routing_decision")
                        status_box.write(f"🚦 **Router:** Interpreted query. Routing to `{decision}` pipeline.")
                        
                    elif key == "retriever":
                        chunks_len = len(value.get("retrieved_chunks", []))
                        if value.get("retrieved_sources"):
                            retrieved_sources = value.get("retrieved_sources", [])
                        if value.get("retrieved_docs"):
                            retrieved_docs = value.get("retrieved_docs", [])
                        if chunks_len == 0:
                            status_box.write(
                                "🔍 **Retriever:** No chunks found — vector index may be empty. "
                                "Build the index from the sidebar."
                            )
                        else:
                            status_box.write(
                                f"🔍 **Retriever:** Pulled {chunks_len} relevant chunks from ChromaDB."
                            )
                        
                    elif key == "summary":
                        status_box.write("📝 **Summary:** Condensed findings from retrieved chunks (parallel worker).")
                        
                    elif key == "analysis":
                        status_box.write("🧠 **Analysis:** Cross-compared sources to draft the response (parallel worker).")
                        
                    elif key == "join":
                        status_box.write("🔗 **Join:** Summary and Analysis complete — sending to Reflection.")

                    elif key == "reflection":
                        if value.get("reflection_approved"):
                            status_box.write("✅ **Reflection:** Final answer approved.")
                        else:
                            fb = value.get("error_message")
                            retry = value.get("reflection_retry_count", 0)
                            status_box.write(
                                f"⚠️ **Reflection:** Revision requested ({retry}/2) — {fb}"
                            )
                            
                    # Capture final answer whenever available (specifically after reflection approves)
                    if value.get("final_answer"):
                        final_answer = value.get("final_answer")
                        
            status_box.update(label="Complete", state="complete", expanded=False)
            
            # Display Final Answer outside the status box
            if final_answer:
                st.markdown(final_answer)

                if retrieved_docs:
                    contributions = build_paper_contributions(final_answer, retrieved_docs)
                    paper_contribs = [
                        item for item in contributions
                        if item["source"] != "_general_synthesis_"
                    ]

                    if paper_contribs:
                        with st.expander("📖 Which paper says what? (Literature review map)", expanded=True):
                            st.caption(
                                "Each paper below lists the specific claims from the answer "
                                "that cite that source — use this to jump straight to relevant papers."
                            )
                            for item in paper_contribs:
                                source = item["source"]
                                pages = item["pages"]
                                page_label = (
                                    f" (pages {', '.join(str(p) for p in pages)})"
                                    if pages else ""
                                )
                                st.markdown(f"**`{source}`**{page_label}")
                                for claim in item["claims"]:
                                    st.markdown(f"- {claim}")
                                st.divider()

                    cited = cited_sources_in_answer(final_answer, retrieved_docs)
                    uncited_retrieved = sorted(set(retrieved_sources) - set(cited))
                    if uncited_retrieved:
                        with st.expander("📄 Retrieved but not cited in answer"):
                            for source in uncited_retrieved:
                                st.markdown(f"- `{source}`")

                elif retrieved_sources:
                    unique_sources = sorted(set(retrieved_sources))
                    with st.expander("📚 Sources cited"):
                        for source in unique_sources:
                            st.markdown(f"- `{source}`")
            else:
                st.warning("No definitive answer was reached.")
                
        except Exception as e:
            status_box.update(label="Pipeline failed.", state="error")
            st.error(f"Error during execution: {e}")
