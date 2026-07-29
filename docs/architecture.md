# SatorIQ System Architecture

## Overview

SatorIQ implements an agentic AI architecture where five specialized agents collaborate through a LangGraph state machine to answer research questions grounded in a local corpus of academic papers.

## High-Level Flow

```
User Query
    │
    ▼
Streamlit Interface (streamlit_app.py)
    │
    ▼
LangGraph Workflow (app/workflow.py)
    │
    ├── Router Agent (Groq)          ← Planning Pattern
    │       │
    │       ▼
    ├── Retriever Agent (RAG Tool)   ← Tool Use Pattern
    │       │
    │       ▼
    ├── Summary Agent (Groq)
    │       │
    │       ▼
    ├── Analysis Agent (OpenRouter)
    │       │
    │       ▼
    └── Reflection Agent (Groq)      ← Reflection Pattern
            │
            ├── approved → END
            └── rejected → loop back to Analysis
```

## RAG Pipeline

The Retriever Agent uses a modular RAG pipeline:

```
PDF Files (data/raw_papers/)
    │
    ▼
PDFLoader (app/rag/loader.py)
    │
    ▼
TextChunker (app/rag/chunker.py)        — 1000 chars, 200 overlap
    │
    ▼
EmbeddingsManager (app/rag/embeddings.py) — Sentence Transformers
    │
    ▼
VectorStoreManager (app/rag/vector_store.py) — ChromaDB
    │
    ▼
RAGRetriever (app/rag/retriever.py)     — Facade for agents
```

## LangGraph State

All agents read from and write to a shared `GraphState` TypedDict defined in `app/agents/state.py`:

| Field | Type | Set By |
|-------|------|--------|
| `user_query` | str | User / Router |
| `routing_decision` | str | Router |
| `retrieved_chunks` | List[str] | Retriever |
| `retrieved_sources` | List[str] | Retriever |
| `summary_result` | str | Summary |
| `analysis_result` | str | Analysis |
| `reflection_approved` | bool | Reflection |
| `final_answer` | str | Reflection |
| `error_message` | Optional[str] | Reflection (feedback loop) |

## Model Routing Strategy

Different models are used for different cognitive tasks:

| Task | Provider | Model | Rationale |
|------|----------|-------|-----------|
| Routing | Groq | llama-3.3-70b-versatile | Fast JSON decision-making |
| Summarization | Groq | llama-3.3-70b-versatile | Efficient text condensation |
| Analysis | OpenRouter | meta-llama/llama-3.1-8b-instruct | Deeper reasoning for comparison |
| Reflection | Groq | llama-3.3-70b-versatile | Fast quality critique |

This multi-model approach avoids using a single LLM for every task, optimizing for both speed and quality.

## Agentic Design Patterns

### 1. Planning Pattern (Router Agent)

The Router Agent analyzes the user query and produces a structured JSON plan deciding whether document retrieval is needed. This is the entry point of the LangGraph workflow.

### 2. Tool Use Pattern (Retriever Agent)

The Retriever Agent acts as a tool-calling agent that invokes the RAG pipeline (ChromaDB search) and returns structured results to the shared state.

### 3. Sequential Workflow

After retrieval, agents execute in fixed order: Summary → Analysis → Reflection. Each agent enriches the state before passing control to the next.

### 4. Reflection Pattern (Reflection Agent)

The Reflection Agent critiques the Analysis output. If the answer is incomplete, it sets `reflection_approved = false` and routes back to the Analysis Agent with feedback — creating a self-correction loop.

## Module Dependencies

```
streamlit_app.py
    └── app/workflow.py
            ├── app/agents/router_agent.py    → app/models/groq_client.py
            ├── app/agents/retriever_agent.py → app/rag/retriever.py
            ├── app/agents/summary_agent.py   → app/models/groq_client.py
            ├── app/agents/analysis_agent.py  → app/models/openrouter_client.py
            └── app/agents/reflection_agent.py → app/models/groq_client.py
```

## Error Handling

- Router falls back to `"retrieve"` if JSON parsing fails
- Analysis returns a user-friendly message if no documents are retrieved
- Reflection degrades gracefully by approving the answer if critique fails
- Streamlit displays agent-level errors in the status panel
