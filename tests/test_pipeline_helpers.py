import io
import sys
from pathlib import Path
import unittest
from contextlib import redirect_stdout

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from experiments.test_agents import dedupe_by_key, merge_selected_with_retrieved
from retrieval.models import RetrievedCode, RetrievedTerm


def _term(term_id: str, code: str) -> RetrievedTerm:
    return RetrievedTerm(
        term_id=term_id,
        code=code,
        path=f"path::{term_id}",
        source="test",
        score=1.0,
    )


def _code(code: str) -> RetrievedCode:
    return RetrievedCode(
        code=code,
        description=f"description::{code}",
        path=f"path::{code}",
        source="test",
        score=1.0,
    )


class PipelineHelpersTests(unittest.TestCase):
    def test_dedupe_by_key_preserves_order_for_terms(self) -> None:
        terms = [_term("001", "K31.6"), _term("002", "I25.2"), _term("001", "K31.6")]
        deduped = dedupe_by_key(terms, key_fn=lambda term: term.term_id)
        self.assertEqual([term.term_id for term in deduped], ["001", "002"])

    def test_dedupe_by_key_preserves_order_for_codes(self) -> None:
        codes = [_code("K31.6"), _code("I25.2"), _code("K31.6"), _code("J44.9")]
        deduped = dedupe_by_key(codes, key_fn=lambda code: code.code)
        self.assertEqual([code.code for code in deduped], ["K31.6", "I25.2", "J44.9"])

    def test_merge_selected_with_retrieved_respects_pool_size_and_dedup(self) -> None:
        selected = [_term("s1", "K31.6")]
        retrieved = [
            _term("s1", "K31.6"),
            _term("r1", "I25.2"),
            _term("r2", "J44.9"),
            _term("r3", "R91.8"),
        ]

        with redirect_stdout(io.StringIO()):
            merged = merge_selected_with_retrieved(
                selected,
                retrieved,
                pool_size=3,
                key_fn=lambda term: term.term_id,
                code_fn=lambda term: term.code,
                stage_name="Assign",
                item_label="terms",
            )

        self.assertEqual([term.term_id for term in merged], ["s1", "r1", "r2"])
        self.assertEqual([term.code for term in merged], ["K31.6", "I25.2", "J44.9"])


if __name__ == "__main__":
    unittest.main()
