# SatorIQ Viva Guide (10–15 Minutes)

Use this script to demonstrate and explain SatorIQ during your viva examination.

## 1. Introduction (1 minute)

> "SatorIQ is an Agentic AI research assistant built with Python, Streamlit, LangGraph, and RAG. Instead of a single chatbot, it uses five specialized AI agents that collaborate to search, summarize, analyze, and review academic research papers stored in a ChromaDB vector database."

## 2. Live Demo (4 minutes)

1. Open the Streamlit app: `streamlit run streamlit_app.py`
2. Show the sidebar listing all five active agents and their model providers
3. If not yet built, click **Build Vector Index** and explain:
   - PDFs are loaded from `data/raw_papers/`
   - Text is chunked and embedded using Sentence Transformers
   - Vectors are stored in ChromaDB
4. Ask a sample question: *"What are the main challenges of Explainable AI in healthcare?"*
5. Expand the agent status panel and walk through each step:
   - Router interprets the query
   - Retriever pulls chunks from ChromaDB
   - Summary condenses findings
   - Analysis compares and synthesizes
   - Reflection approves the final answer

## 3. Architecture Explanation (3 minutes)

Open [docs/architecture.md](architecture.md) or the diagram and explain:

- **Why agents?** Each agent has one job (Single Responsibility Principle)
- **How they communicate:** Shared `GraphState` dictionary passed through LangGraph
- **Why two LLM providers?** Groq for speed (routing, summary, reflection), OpenRouter for deep analysis
- **RAG purpose:** Grounds answers in actual research papers, reducing hallucination

## 4. Design Patterns (3 minutes)

| Pattern | Where | What to Say |
|---------|-------|-------------|
| Planning | Router Agent | "The router analyzes the query and decides the workflow before any retrieval happens." |
| Tool Use | Retriever Agent | "The retriever calls the RAG pipeline as a tool — it searches ChromaDB and returns structured chunks." |
| Sequential Workflow | Summary → Analysis → Reflection | "Each agent adds to the shared state in a fixed pipeline order." |
| Reflection | Reflection Agent | "The reflection agent critiques the answer. If quality is insufficient, it sends feedback back to the analysis agent for revision." |

## 5. Code Walkthrough (2 minutes)

Show these key files:

1. `app/workflow.py` — LangGraph graph definition with conditional edges
2. `app/agents/state.py` — Shared state TypedDict
3. `app/rag/retriever.py` — RAG facade used by the Retriever Agent
4. `streamlit_app.py` — UI that streams agent events in real time

## 6. Testing & Deployment (1 minute)

- Run tests: `python -m pytest tests/ -v`
- Mention Streamlit Cloud deployment with secrets for API keys
- PDFs stay local; vector index is built at runtime

## 7. Closing (1 minute)

> "SatorIQ demonstrates how agentic AI patterns — planning, tool use, sequential workflows, and reflection — can be combined with RAG to create a modular, production-quality research assistant. The codebase follows SOLID principles with type hints, docstrings, logging, and environment-based configuration."

## Common Viva Questions

**Q: Why LangGraph instead of a simple function chain?**
A: LangGraph provides conditional routing (reflection loop), state management, and streaming — making the agent workflow explicit and extensible.

**Q: Why ChromaDB?**
A: Lightweight, persistent, local vector store ideal for a student project without cloud database costs.

**Q: What if no papers match the query?**
A: The Analysis Agent detects empty retrieval and returns a helpful message instead of hallucinating.

**Q: How do you prevent API key exposure?**
A: Keys are loaded from `.env` locally and Streamlit secrets in deployment — never hardcoded.

**Q: Can you add more agents?**
A: Yes — add a new node in `workflow.py`, define its logic in `app/agents/`, and extend `GraphState`.
