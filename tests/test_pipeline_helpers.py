"""Tests for workflow pipeline helper utilities."""

from app.utils.pipeline_helpers import (
    is_non_retriable_answer,
    should_skip_reflection,
    reflection_retries_exhausted,
)


class TestIsNonRetriableAnswer:
    def test_empty_analysis_is_retriable(self):
        assert is_non_retriable_answer("") is False

    def test_empty_knowledge_base_message_is_non_retriable(self):
        message = (
            "I could not find any relevant papers in the knowledge base. "
            "Please add PDF research papers."
        )
        assert is_non_retriable_answer(message) is True

    def test_normal_analysis_is_retriable(self):
        assert is_non_retriable_answer("Explainable AI improves clinician trust.") is False


class TestShouldSkipReflection:
    def test_skips_when_flag_set(self):
        state = {"skip_reflection": True, "retrieved_chunks": ["chunk"]}
        assert should_skip_reflection(state) is True

    def test_skips_when_no_chunks(self):
        state = {"skip_reflection": False, "retrieved_chunks": []}
        assert should_skip_reflection(state) is True

    def test_runs_when_chunks_present(self):
        state = {
            "skip_reflection": False,
            "retrieved_chunks": ["chunk"],
            "analysis_result": "Grounded synthesis from papers.",
        }
        assert should_skip_reflection(state) is False


class TestReflectionRetriesExhausted:
    def test_not_exhausted_below_limit(self):
        assert reflection_retries_exhausted({"reflection_retry_count": 1}) is False

    def test_exhausted_at_limit(self):
        assert reflection_retries_exhausted({"reflection_retry_count": 2}) is True
