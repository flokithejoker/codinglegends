import hashlib
import re
import sys
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from openai import OpenAI

from retrieval.base import RetrievalBackend
from retrieval.models import GuidelineChunk, RetrievedCode, RetrievedTerm

try:
    from icd_codes.icd import ICD10Trie
except ModuleNotFoundError:
    # Allow importing retrieval when only `src/` is added to PYTHONPATH.
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from icd_codes.icd import ICD10Trie


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_EPSILON = 1e-12


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, _EPSILON, None)


class EmbeddingProvider(Protocol):
    """Lightweight protocol for swappable embedding implementations."""

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed text rows into L2-normalized dense vectors."""


class HashEmbeddingProvider:
    """Fast local baseline embedder using hashed bag-of-words vectors."""

    def __init__(self, dim: int = 768):
        if dim <= 0:
            raise ValueError("`dim` must be positive.")
        self.dim = dim

    def _token_index(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, byteorder="big", signed=False)
        index = value % self.dim
        sign = 1.0 if ((value >> 1) & 1) == 0 else -1.0
        return index, sign

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row_idx, text in enumerate(texts):
            for token in _TOKEN_PATTERN.findall(text.lower()):
                index, sign = self._token_index(token)
                vectors[row_idx, index] += sign
        return _normalize_rows(vectors)


class OpenAIEmbeddingProvider:
    """OpenAI-compatible embedding provider."""

    def __init__(
        self,
        model: str,
        host: str = "localhost",
        port: int = 6546,
        batch_size: int = 128,
    ):
        self.model = model
        self.batch_size = batch_size
        self.client = OpenAI(base_url=f"http://{host}:{port}/v1/", api_key="EMPTY")

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        all_vectors: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            if not batch:
                continue
            response = self.client.embeddings.create(model=self.model, input=batch)  # type: ignore[arg-type]
            all_vectors.extend(
                np.asarray(row.embedding, dtype=np.float32) for row in response.data
            )
        vectors = (
            np.stack(all_vectors) if all_vectors else np.zeros((0, 0), dtype=np.float32)
        )
        return _normalize_rows(vectors)


class InMemoryVectorRetriever(RetrievalBackend):
    """Simple retriever built directly on top of ICD10Trie."""

    def __init__(
        self,
        trie: ICD10Trie,
        *,
        embedder: EmbeddingProvider,
        allowed_codes: set[str] | None = None,
        max_terms: int | None = 20000,
    ):
        self.trie = trie
        self.embedder = embedder
        self.max_terms = max_terms
        self.allowed_codes = self._resolve_allowed_codes(allowed_codes or set())
        self._allowed_code_compacts = {
            allowed_code.replace(".", "") for allowed_code in self.allowed_codes
        }

        self.term_candidates = self._build_term_candidates()
        self._term_vectors = self._build_vectors(
            [candidate.path for candidate in self.term_candidates]
        )

    @classmethod
    def from_icd_dir(
        cls,
        icd_dir: Path,
        *,
        embedder: EmbeddingProvider | None = None,
        allowed_codes: set[str] | None = None,
        max_terms: int | None = 20000,
    ) -> "InMemoryVectorRetriever":
        trie = ICD10Trie.from_dir(icd_dir)
        trie.parse()
        return cls(
            trie,
            embedder=embedder or HashEmbeddingProvider(),
            allowed_codes=allowed_codes,
            max_terms=max_terms,
        )

    @classmethod
    def from_icd_xml(
        cls,
        icd_dir: Path,
        *,
        embedder: EmbeddingProvider | None = None,
        allowed_codes: set[str] | None = None,
        max_codes: int | None = 20000,
        use_cache: bool = True,
    ) -> "InMemoryVectorRetriever":
        # Legacy alias kept to avoid breaking callers while moving to ICD10Trie.
        _ = use_cache
        return cls.from_icd_dir(
            icd_dir=icd_dir,
            embedder=embedder,
            allowed_codes=allowed_codes,
            max_terms=max_codes,
        )

    def retrieve_terms(self, queries: list[str], limit: int = 20) -> list[RetrievedTerm]:
        ranked = self._rank_documents(
            queries=queries,
            document_vectors=self._term_vectors,
            limit=limit,
        )
        return [
            self.term_candidates[doc_index].model_copy(update={"score": score})
            for doc_index, score in ranked
        ]

    def retrieve_codes(
        self, terms: list[RetrievedTerm], limit: int = 30
    ) -> list[RetrievedCode]:
        if limit <= 0:
            return []

        code_scores = self._score_codes_from_terms(terms)

        retrieved_codes: list[RetrievedCode] = []
        for code, score in code_scores.items():
            node = self.trie[code]
            retrieved_codes.append(
                RetrievedCode(
                    code=code,
                    description=node.description,
                    path=f"{code} | {node.description}",
                    source="icd10",
                    score=score,
                )
            )

        retrieved_codes.sort(key=lambda row: row.score, reverse=True)
        return retrieved_codes[:limit]

    def retrieve_guidelines(
        self, codes: list[str], limit: int = 20
    ) -> list[GuidelineChunk]:
        valid_codes = self._valid_codes(codes)
        if not valid_codes:
            return []

        guidelines = self.trie.get_guidelines(valid_codes)
        chunks: list[GuidelineChunk] = []
        for row in guidelines[:limit]:
            title = str(row.get("title", "")).strip()
            number = str(row.get("number", "")).strip()
            content = str(row.get("content", "")).strip()
            prefix = " ".join(part for part in [number, title] if part).strip()
            text = "\n".join(part for part in [prefix, content] if part).strip()
            if text:
                chunks.append(GuidelineChunk(content=text))
        return chunks

    def retrieve_instructional_notes(self, codes: list[str]) -> list[dict[str, object]]:
        valid_codes = self._valid_codes(codes)
        if not valid_codes:
            return []
        return self.trie.get_instructional_notes(valid_codes)

    def _build_term_candidates(self) -> list[RetrievedTerm]:
        candidates: list[RetrievedTerm] = []
        for term in self.trie.index.values():
            if not term.assignable:
                continue
            resolved_code = self._resolve_code(term.code or "")
            if resolved_code is None:
                continue
            if self.allowed_codes and not self._matches_allowed_code(resolved_code):
                continue
            path = (term.path or term.title or "").strip()
            if not path:
                continue
            candidates.append(
                RetrievedTerm(
                    term_id=term.id,
                    code=resolved_code,
                    path=path,
                    source="icd10",
                )
            )

        candidates.sort(key=lambda row: row.term_id)
        if self.max_terms is not None:
            return candidates[: self.max_terms]
        return candidates

    def _build_vectors(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        vectors = self.embedder.embed(texts)
        if vectors.ndim != 2:
            raise ValueError("Embedding provider must return a 2D array.")
        return _normalize_rows(vectors.astype(np.float32))

    def _codes_from_term(self, term: RetrievedTerm) -> list[str]:
        """Resolve candidate codes for one retrieved term."""
        try:
            return self.trie.get_term_codes(term.term_id, subterms=False)
        except Exception:
            return [term.code] if term.code else []

    def _score_codes_from_terms(self, terms: list[RetrievedTerm]) -> dict[str, float]:
        """Aggregate best score per resolved code from retrieved terms."""
        code_scores: dict[str, float] = {}
        for term in terms:
            for code in self._codes_from_term(term):
                resolved_code = self._resolve_code(code)
                if resolved_code is None:
                    continue
                prev_score = code_scores.get(resolved_code)
                if prev_score is None or term.score > prev_score:
                    code_scores[resolved_code] = term.score
        return code_scores

    def _rank_documents(
        self, queries: list[str], document_vectors: np.ndarray, limit: int
    ) -> list[tuple[int, float]]:
        cleaned_queries = [query.strip() for query in queries if query and query.strip()]
        if not cleaned_queries or limit <= 0 or document_vectors.size == 0:
            return []

        query_vectors = self._build_vectors(cleaned_queries)
        doc_count = document_vectors.shape[0]
        k = min(limit, doc_count)
        score_matrix = query_vectors @ document_vectors.T

        best_scores: dict[int, float] = {}
        for row in score_matrix:
            candidate_indices = np.argpartition(row, -k)[-k:]
            for doc_index in candidate_indices.tolist():
                score = float(row[doc_index])
                existing_score = best_scores.get(doc_index)
                if existing_score is None or score > existing_score:
                    best_scores[doc_index] = score

        ranked = sorted(best_scores.items(), key=lambda pair: pair[1], reverse=True)
        return ranked[:limit]

    def _valid_codes(self, codes: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for code in codes:
            resolved = self._resolve_code(code)
            if resolved is None or resolved in seen:
                continue
            seen.add(resolved)
            normalized.append(resolved)
        return normalized

    def _resolve_allowed_codes(self, codes: set[str]) -> set[str]:
        if not codes:
            return set()
        resolved: set[str] = set()
        for code in codes:
            resolved_code = self._resolve_code(code)
            if resolved_code is not None:
                resolved.add(resolved_code)
        return resolved

    def _normalize_code(self, code: str) -> str:
        return code.strip().upper()

    def _candidate_code_forms(self, code: str) -> list[str]:
        """Generate fallback code variants in priority order."""
        normalized = self._normalize_code(code)
        if not normalized:
            return []

        candidates: list[str] = []
        seen: set[str] = set()

        def add(candidate: str) -> None:
            if not candidate or candidate in seen:
                return
            seen.add(candidate)
            candidates.append(candidate)

        add(normalized)
        add(normalized.replace(".", ""))
        if "." not in normalized and len(normalized) > 3:
            add(f"{normalized[:3]}.{normalized[3:]}")

        for candidate in [
            normalized[:7],
            normalized[:6],
            normalized[:5],
            normalized[:4],
            normalized[:3],
        ]:
            add(candidate)
            add(candidate.replace(".", ""))
            if "." not in candidate and len(candidate) > 3:
                add(f"{candidate[:3]}.{candidate[3:]}")

        return candidates

    def _resolve_from_lookup(self, candidates: list[str]) -> str | None:
        """Return the first candidate present in trie lookup."""
        for candidate in candidates:
            if candidate in self.trie.lookup:
                return candidate
        return None

    def _resolve_code(self, code: str) -> str | None:
        if not code:
            return None
        candidates = self._candidate_code_forms(code)
        return self._resolve_from_lookup(candidates)

    def _matches_allowed_code(self, code: str) -> bool:
        if not self.allowed_codes:
            return True
        if code in self.allowed_codes:
            return True

        compact = code.replace(".", "")
        for allowed_compact in self._allowed_code_compacts:
            if compact.startswith(allowed_compact) or allowed_compact.startswith(compact):
                return True
        return False
