"""Tests for citation helper utilities."""

from app.utils.citation_helpers import (
    build_paper_contributions,
    cited_sources_in_answer,
    format_labeled_chunks,
    format_source_index,
)


SAMPLE_DOCS = [
    {
        "source": "IDEAL2021_paper58.pdf",
        "page": 4,
        "content": "XAI can improve clinical decision support.",
        "label": "Source 1",
    },
    {
        "source": "ssrn-4637897.pdf",
        "page": 12,
        "content": "Interpretability gaps remain in medical AI.",
        "label": "Source 2",
    },
]


class TestFormatLabeledChunks:
    def test_includes_source_label_and_page(self):
        text = format_labeled_chunks(SAMPLE_DOCS)
        assert "[Source 1: IDEAL2021_paper58.pdf, p.4]" in text
        assert "[Source 2: ssrn-4637897.pdf, p.12]" in text


class TestFormatSourceIndex:
    def test_lists_unique_sources(self):
        index = format_source_index(SAMPLE_DOCS)
        assert "Source 1: IDEAL2021_paper58.pdf, p.4" in index
        assert "Source 2: ssrn-4637897.pdf, p.12" in index


class TestBuildPaperContributions:
    def test_maps_claims_to_papers(self):
        answer = (
            "XAI improves clinical decisions [Source 1: IDEAL2021_paper58.pdf, p.4].\n\n"
            "Medical AI still lacks interpretability [Source 2: ssrn-4637897.pdf, p.12]."
        )
        contributions = build_paper_contributions(answer, SAMPLE_DOCS)

        by_source = {item["source"]: item["claims"] for item in contributions}
        assert "IDEAL2021_paper58.pdf" in by_source
        assert "ssrn-4637897.pdf" in by_source
        assert any("clinical decisions" in claim for claim in by_source["IDEAL2021_paper58.pdf"])


class TestCitedSourcesInAnswer:
    def test_extracts_cited_filenames(self):
        answer = "Finding A [Source 1: IDEAL2021_paper58.pdf, p.4]."
        cited = cited_sources_in_answer(answer, SAMPLE_DOCS)
        assert cited == ["IDEAL2021_paper58.pdf"]
