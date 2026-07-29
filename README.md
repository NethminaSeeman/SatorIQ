# SatorIQ – Agentic AI Research Paper Screening & Literature Assistant

SatorIQ is an Agentic AI-powered research assistant that helps students and researchers search, summarize, compare, and analyze academic research papers using multiple collaborating AI agents and Retrieval-Augmented Generation (RAG).

Instead of a traditional chatbot, the system uses five specialized agents orchestrated by LangGraph, each with a distinct responsibility.

## Features

- **Multi-Agent Architecture** — Router, Retriever, Summary, Analysis, and Reflection agents
- **RAG Pipeline** — PDF loading, chunking, Sentence Transformers embeddings, ChromaDB vector search
- **Multiple LLM Providers** — Groq (fast routing/summary/reflection) and OpenRouter (deep analysis)
- **Agentic Patterns** — Planning, Reflection, Tool Use, Sequential Workflow
- **Streamlit UI** — Live agent status streaming during query processing

## Architecture

```
User → Streamlit UI → Router Agent → Retriever Agent → ChromaDB
                              ↓
                    Summary Agent + Analysis Agent
                              ↓
                      Reflection Agent → Final Answer
```

See [docs/architecture.md](docs/architecture.md) and [diagrams/system_architecture.mmd](diagrams/system_architecture.mmd) for full details.

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| UI | Streamlit |
| Agent Orchestration | LangGraph |
| Vector Database | ChromaDB |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Fast LLM | Groq (`llama-3.3-70b-versatile`) |
| Analysis LLM | OpenRouter (`meta-llama/llama-3.1-8b-instruct`) |

## Project Structure

```
app/
  agents/          # Five specialized AI agents
  rag/             # RAG pipeline (loader, chunker, embeddings, vector store)
  models/          # Groq and OpenRouter client wrappers
  workflow.py      # LangGraph state machine
data/
  raw_papers/      # Research PDFs (local only, not in git)
  vector_db/       # ChromaDB persistence (generated at runtime)
docs/              # Architecture, deployment, and viva guides
diagrams/          # System architecture diagrams
tests/             # Unit tests for RAG pipeline
streamlit_app.py   # Streamlit entry point
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/NethminaSeeman/SatorIQ.git
cd SatorIQ
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
copy .env.example .env       # Windows
# cp .env.example .env       # macOS/Linux
```

Edit `.env` and add your API keys:

```
GROQ_API_KEY=your_groq_key
OPENROUTER_API_KEY=your_openrouter_key
```

### 5. Add research papers

Place 20–30 PDF research papers in `data/raw_papers/`. These files are gitignored and stay on your local machine.

### 6. Run the application

```bash
streamlit run streamlit_app.py
```

On first run, click **Build Vector Index** in the sidebar to ingest PDFs into ChromaDB.

## Agent Responsibilities

| Agent | Model | Role |
|-------|-------|------|
| Router | Groq | Understands query, plans workflow, routes to retrieval |
| Retriever | RAG/ChromaDB | Searches vector DB, returns relevant document chunks |
| Summary | Groq | Summarizes retrieved research, extracts key findings |
| Analysis | OpenRouter | Compares papers, generates insights, answers questions |
| Reflection | Groq | Reviews answer quality, triggers revision if needed |

## Agent Communication

Agents communicate through a shared LangGraph `GraphState` dictionary:

```python
# Router → Retriever
{"task": "retrieve", "query": "Explainable AI in healthcare"}

# Retriever → Summary
{"chunks": [...], "sources": [...]}

# Summary → Analysis
{"summary": "..."}

# Analysis → Reflection
{"analysis": "..."}

# Reflection → Final
{"approved": true, "answer": "..."}
```

## Deployment

See [docs/deployment.md](docs/deployment.md) for Streamlit Community Cloud deployment instructions.

## Testing

```bash
python -m pytest tests/ -v
```

## Viva Preparation

See [docs/viva-guide.md](docs/viva-guide.md) for a 10–15 minute viva script covering demo flow, agent explanations, and design patterns.

## License

This project was developed as a university assignment.

## Author

**Nethmina Seeman** — [GitHub](https://github.com/NethminaSeeman)
