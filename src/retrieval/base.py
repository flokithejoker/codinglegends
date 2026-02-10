from abc import ABC, abstractmethod
from typing import Any

from retrieval.models import GuidelineChunk, RetrievedCode, RetrievedTerm


class RetrievalBackend(ABC):
    """Backend-agnostic retrieval contract used by the pipeline."""

    @abstractmethod
    def retrieve_terms(self, queries: list[str], limit: int = 20) -> list[RetrievedTerm]:
        """Return term candidates for locate step."""

    @abstractmethod
    def retrieve_codes(
        self, terms: list[RetrievedTerm], limit: int = 30
    ) -> list[RetrievedCode]:
        """Return code candidates for assign step."""

    @abstractmethod
    def retrieve_guidelines(
        self, codes: list[str], limit: int = 20
    ) -> list[GuidelineChunk]:
        """Return guideline chunks for verify step."""

    @abstractmethod
    def retrieve_instructional_notes(self, codes: list[str]) -> list[dict[str, Any]]:
        """Return instructional note records for verify step."""
