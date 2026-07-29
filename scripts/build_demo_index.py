"""Build a small ChromaDB index (~5 MB) for Streamlit Cloud deployment."""

import shutil
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

from app.rag.chunker import TextChunker
from app.rag.vector_store import VectorStoreManager

DEMO_PDFS = [
    "Explainable_AI_in_Healthcare.pdf",
    "IDEAL2021_paper58.pdf",
    "ssrn-3367686.pdf",
    "EmanPublisher_13_6372ai-2110685.pdf",
    "41666_2022_Article_114.pdf",
    "ssrn-4637897.pdf",
    "s12911-020-01332-6.pdf",
]


def main() -> None:
    base = Path("data/raw_papers")
    db_path = Path("data/vector_db")
    if db_path.exists():
        shutil.rmtree(db_path)

    documents = []
    for name in DEMO_PDFS:
        documents.extend(PyPDFLoader(str(base / name)).load())

    chunks = TextChunker().chunk_documents(documents)
    store = VectorStoreManager()
    store.add_documents(chunks)

    total_bytes = sum(f.stat().st_size for f in db_path.rglob("*") if f.is_file())
    print(f"Indexed {len(DEMO_PDFS)} PDFs -> {len(chunks)} chunks ({total_bytes / (1024 * 1024):.2f} MB)")


if __name__ == "__main__":
    main()
