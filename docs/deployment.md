# SatorIQ Deployment Guide

This guide explains how to deploy SatorIQ on [Streamlit Community Cloud](https://streamlit.io/cloud).

## Prerequisites

- A public GitHub repository: `NethminaSeeman/SatorIQ`
- A [Groq](https://console.groq.com/) API key
- An [OpenRouter](https://openrouter.ai/) API key
- Research PDFs placed locally in `data/raw_papers/` (not committed to git)

## Streamlit Cloud Setup

1. Sign in to [share.streamlit.io](https://share.streamlit.io/) with your GitHub account.
2. Click **New app**.
3. Select repository: `NethminaSeeman/SatorIQ`
4. Branch: `main` (or `develop` for staging)
5. Main file path: `streamlit_app.py`
6. Click **Advanced settings** and set Python version to **3.10** or **3.11**.

## Configure Secrets

In the Streamlit Cloud app settings, open **Secrets** and add:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
OPENROUTER_API_KEY = "your_openrouter_api_key_here"
LOG_LEVEL = "INFO"
```

These map to the same variables documented in [`.env.example`](../.env.example).

## First Run on Cloud

Streamlit Cloud does not include your local PDF files by default.

For a live demo you have two options:

1. **Pre-build the vector index locally** and commit only the ChromaDB folder structure is **not recommended** (large binary artifacts). Instead, upload PDFs through your deployment workflow or rebuild at runtime.
2. **Upload PDFs before indexing**: Add papers to `data/raw_papers/` in the deployed environment, then click **Build Vector Index** in the sidebar.

Recommended demo flow:

1. Deploy the app with secrets configured.
2. Upload 20–30 PDF research papers into `data/raw_papers/`.
3. Open the app and click **Build Vector Index** once.
4. Ask a research question and observe the multi-agent pipeline.

## Local Deployment (Development)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
copy .env.example .env       # then add your API keys
streamlit run streamlit_app.py
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Missing API key error | Verify secrets in Streamlit Cloud or `.env` locally |
| No papers found | Add PDFs to `data/raw_papers/` and rebuild the index |
| OpenRouter 402/403 | Check credits and model slug in `app/models/openrouter_client.py` |
| Slow first query | Sentence Transformers downloads the embedding model on first run |

## Architecture Note

Deployment uses the same LangGraph workflow as local development. Only environment configuration changes between environments; agent logic remains unchanged.
