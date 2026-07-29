"""Helpers for source-labeled chunks and literature-review citation mapping."""

from __future__ import annotations

import re
from typing import Any, TypedDict


class RetrievedDoc(TypedDict):
    source: str
    page: int | None
    content: str
    label: str


SOURCE_CITATION = re.compile(
    r"\[Source\s+(?P<num>\d+)(?::\s*(?P<file>[^\],]+))?(?:,\s*p\.?\s*(?P<page>\d+))?\]",
    re.IGNORECASE,
)
FILE_CITATION = re.compile(
    r"\[(?!Source\s)(?P<file>[^\],]+\.pdf)(?:,\s*p\.?\s*(?P<page>\d+))?\]",
    re.IGNORECASE,
)


def _extract_cited_sources(text: str, label_to_source: dict[str, str]) -> set[str]:
    cited: set[str] = set()
    for match in SOURCE_CITATION.finditer(text):
        source = label_to_source.get(f"Source {match.group('num')}")
        if source:
            cited.add(source)
    for match in FILE_CITATION.finditer(text):
        cited.add(match.group("file").strip())
    return cited


def _strip_citations(text: str) -> str:
    cleaned = SOURCE_CITATION.sub("", text)
    cleaned = FILE_CITATION.sub("", cleaned)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def doc_from_chunk(source: str, page: int | None, content: str, index: int) -> RetrievedDoc:
    """Build a labeled retrieved-doc record from a single chunk."""
    return {
        "source": source,
        "page": page,
        "content": content,
        "label": f"Source {index}",
    }


def format_labeled_chunks(docs: list[RetrievedDoc]) -> str:
    """Format chunks with explicit source labels for LLM prompts."""
    blocks: list[str] = []
    for doc in docs:
        page_part = f", p.{doc['page']}" if doc.get("page") is not None else ""
        blocks.append(
            f"[{doc['label']}: {doc['source']}{page_part}]\n{doc['content']}"
        )
    return "\n\n".join(blocks)


def format_source_index(docs: list[RetrievedDoc]) -> str:
    """Compact bibliography list for prompts."""
    seen: set[str] = set()
    lines: list[str] = []
    for doc in docs:
        key = doc["label"]
        if key in seen:
            continue
        seen.add(key)
        page_part = f", p.{doc['page']}" if doc.get("page") is not None else ""
        lines.append(f"- {doc['label']}: {doc['source']}{page_part}")
    return "\n".join(lines)


def _split_into_claims(text: str) -> list[str]:
    """Split answer text into claim-sized units (paragraphs or bullets)."""
    claims: list[str] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        block = block.strip()
        if not block:
            continue
        if block.startswith(("- ", "* ", "1.", "2.", "3.", "4.", "5.")):
            for line in block.splitlines():
                line = line.strip()
                if line:
                    claims.append(line)
        else:
            claims.append(block)
    return claims


def build_paper_contributions(
    answer: str,
    docs: list[RetrievedDoc],
) -> list[dict[str, Any]]:
    """
    Map each retrieved paper to the claims in the answer that cite it.
    Returns a list sorted by paper name for literature-review triage.
    """
    label_to_source = {doc["label"]: doc["source"] for doc in docs}
    source_pages: dict[str, set[int]] = {}
    for doc in docs:
        source_pages.setdefault(doc["source"], set())
        if doc.get("page") is not None:
            source_pages[doc["source"]].add(doc["page"])

    contributions: dict[str, list[str]] = {doc["source"]: [] for doc in docs}
    uncited: list[str] = []

    for claim in _split_into_claims(answer):
        if claim.startswith("#"):
            continue

        cited_sources = _extract_cited_sources(claim, label_to_source)
        if not cited_sources:
            if len(claim) > 40 and not claim.startswith("##"):
                uncited.append(claim)
            continue

        clean_claim = _strip_citations(claim)
        if not clean_claim:
            continue

        for source in cited_sources:
            if clean_claim not in contributions.setdefault(source, []):
                contributions[source].append(clean_claim)

    result = []
    for source in sorted(contributions.keys()):
        claims = contributions[source]
        if not claims:
            continue
        pages = sorted(source_pages.get(source, set()))
        result.append({
            "source": source,
            "pages": pages,
            "claims": claims,
        })

    if uncited:
        result.append({
            "source": "_general_synthesis_",
            "pages": [],
            "claims": uncited,
        })

    return result


def cited_sources_in_answer(answer: str, docs: list[RetrievedDoc]) -> list[str]:
    """Return unique paper filenames explicitly cited in the answer."""
    label_to_source = {doc["label"]: doc["source"] for doc in docs}
    return sorted(_extract_cited_sources(answer, label_to_source))
