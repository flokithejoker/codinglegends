import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from retrieval import HashEmbeddingProvider, InMemoryVectorRetriever


class _FakeNode:
    def __init__(self, description: str):
        self.description = description


class _FakeTrie:
    def __init__(self, lookup: dict[str, str]):
        self.lookup = lookup
        self.index: dict[str, object] = {}

    def get_term_codes(self, term_id: str, subterms: bool = False) -> list[str]:
        return []

    def get_guidelines(self, codes: list[str]) -> list[dict[str, str]]:
        return []

    def get_instructional_notes(self, codes: list[str]) -> list[dict[str, object]]:
        return []

    def __getitem__(self, code: str) -> _FakeNode:
        return _FakeNode(description=f"description for {code}")


def _make_retriever(
    lookup_codes: list[str],
    allowed_codes: set[str] | None = None,
) -> InMemoryVectorRetriever:
    lookup = {code: code for code in lookup_codes}
    trie = _FakeTrie(lookup=lookup)
    return InMemoryVectorRetriever(
        trie=trie,  # type: ignore[arg-type]
        embedder=HashEmbeddingProvider(dim=16),
        allowed_codes=allowed_codes,
        max_terms=100,
    )


class VectorRetrieverResolutionTests(unittest.TestCase):
    def test_resolve_dotted_from_undotted_input(self) -> None:
        retriever = _make_retriever(["K31.6"])
        self.assertEqual(retriever._resolve_code("K31.6"), "K31.6")
        self.assertEqual(retriever._resolve_code("K316"), "K31.6")

    def test_resolve_undotted_from_dotted_input(self) -> None:
        retriever = _make_retriever(["K316"])
        self.assertEqual(retriever._resolve_code("K316"), "K316")
        self.assertEqual(retriever._resolve_code("K31.6"), "K316")

    def test_resolve_code_with_truncation_fallback(self) -> None:
        retriever = _make_retriever(["I25.2"])
        self.assertEqual(retriever._resolve_code("I252999"), "I25.2")
        self.assertEqual(retriever._resolve_code("I25.2"), "I25.2")

    def test_matches_allowed_code_uses_compact_forms(self) -> None:
        retriever = _make_retriever(
            ["K31.6", "I25.2", "J44.9"],
            allowed_codes={"K31.6", "I25.2"},
        )
        self.assertTrue(retriever._matches_allowed_code("K31.60"))
        self.assertTrue(retriever._matches_allowed_code("K31"))
        self.assertTrue(retriever._matches_allowed_code("I25.2"))
        self.assertFalse(retriever._matches_allowed_code("J44.9"))


if __name__ == "__main__":
    unittest.main()
