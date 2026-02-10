"""Simple test to verify agent classes work end-to-end with retrieval."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Callable, Hashable, Iterable, TypeVar

import datasets

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from agents.analyse_agent import AnalyseAgent
from agents.assign_agent import AssignAgent
from agents.locate_agent import LocateAgent
from agents.verify_agent import VerifyAgent
from dataloader import DATASET_CONFIGS, load_dataset
from dataloader.base import DatasetConfig
from retrieval import HashEmbeddingProvider, InMemoryVectorRetriever, RetrievedCode, RetrievedTerm

DEFAULT_DATASET_IDENTIFIER = "mimic-iv"
DEFAULT_DATASET_INDEX = 0
DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL = "gpt-4o-mini"

T = TypeVar("T")
K = TypeVar("K", bound=Hashable)


@dataclass(frozen=True)
class PipelineConfig:
    dataset_identifier: str = DEFAULT_DATASET_IDENTIFIER
    sample_index: int = DEFAULT_DATASET_INDEX
    locate_top_k: int = 100
    assign_top_k: int = 40
    verify_guideline_top_k: int = 20
    assign_term_pool_size: int = 40
    verify_code_pool_size: int = 20
    preview_rows: int = 10
    provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL


def _section(title: str, symbol: str = "=") -> None:
    line = symbol * 60
    print(f"\n{line}\n{title}\n{line}")


def _decode_one_indexed(items: list[T], selected_ids: list[int]) -> list[T]:
    return [items[idx - 1] for idx in selected_ids if 0 < idx <= len(items)]


def dedupe_by_key(items: Iterable[T], key_fn: Callable[[T], K]) -> list[T]:
    unique: list[T] = []
    seen: set[K] = set()
    for item in items:
        key = key_fn(item)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def merge_selected_with_retrieved(
    selected: list[T],
    retrieved: list[T],
    *,
    pool_size: int,
    key_fn: Callable[[T], K],
    code_fn: Callable[[T], str],
    stage_name: str,
    item_label: str,
) -> list[T]:
    selected_codes = {code_fn(item) for item in selected}
    merged = dedupe_by_key([*selected, *retrieved[:pool_size]], key_fn=key_fn)
    merged_codes = {code_fn(item) for item in merged}
    print(
        f"{stage_name} input pool: "
        f"selected_{item_label}={len(selected)} "
        f"retrieved_{item_label}={min(len(retrieved), pool_size)} "
        f"merged_{item_label}={len(merged)} "
        f"selected_unique_codes={len(selected_codes)} "
        f"merged_unique_codes={len(merged_codes)}"
    )
    return merged


def _print_terms(terms: list[RetrievedTerm], true_codes: set[str], title: str, max_rows: int) -> None:
    print(f"{title} ({len(terms)} total):")
    for idx, term in enumerate(terms[:max_rows], start=1):
        marker = "T" if term.code in true_codes else " "
        print(f"  [{marker}] {idx:>2}. code={term.code:<8} score={term.score:>7.4f} id={term.term_id} path={term.path[:100]}")


def _print_codes(codes: list[RetrievedCode], true_codes: set[str], title: str, max_rows: int) -> None:
    print(f"{title} ({len(codes)} total):")
    for idx, code in enumerate(codes[:max_rows], start=1):
        marker = "T" if code.code in true_codes else " "
        print(f"  [{marker}] {idx:>2}. code={code.code:<8} score={code.score:>7.4f} description={code.description[:100]}")


def _print_code_overlap(stage: str, predicted_codes: Iterable[str], true_codes: set[str]) -> None:
    predicted_set = set(predicted_codes)
    tp = sorted(predicted_set & true_codes)
    fp = sorted(predicted_set - true_codes)
    fn = sorted(true_codes - predicted_set)
    print(f"{stage} overlap:")
    print(f"  predicted={len(predicted_set)} true={len(true_codes)} tp={len(tp)} fp={len(fp)} fn={len(fn)}")
    print(f"  TP codes: {tp}")
    if fp:
        print(f"  FP codes: {fp}")
    if fn:
        print(f"  FN codes: {fn}")


def _load_dataset_sample(dataset_identifier: str, sample_index: int) -> dict[str, Any]:
    if dataset_identifier not in DATASET_CONFIGS:
        raise KeyError(f"Unknown dataset identifier: {dataset_identifier}")
    dataset_config = DatasetConfig(**DATASET_CONFIGS[dataset_identifier])
    dataset = load_dataset(dataset_config)
    if isinstance(dataset, datasets.DatasetDict):
        dataset = dataset[dataset_config.split] if dataset_config.split and dataset_config.split in dataset else next(iter(dataset.values()))
    if sample_index >= len(dataset):
        raise IndexError(f"Sample index {sample_index} is out of range for dataset '{dataset_identifier}'")
    record = dataset[sample_index]
    record["note"] = record.get("note") or record.get("text")
    if not record["note"]:
        raise ValueError(f"Dataset record does not have a 'note' or 'text' field: {record.keys()}")
    return record


def build_context(config: PipelineConfig) -> dict[str, Any]:
    sample = _load_dataset_sample(config.dataset_identifier, config.sample_index)
    true_codes = set(sample.get("codes", []))
    retriever = InMemoryVectorRetriever.from_icd_xml(
        icd_dir=PROJECT_ROOT / "data" / "medical-coding-systems" / "icd",
        allowed_codes=true_codes,
        max_codes=5000,
        embedder=HashEmbeddingProvider(dim=1024),
    )
    return {
        "sample": sample,
        "note": sample["note"],
        "true_codes": true_codes,
        "retriever": retriever,
        "analyse": AnalyseAgent(config.provider, config.model),
        "locate": LocateAgent(config.provider, config.model),
        "assign": AssignAgent(config.provider, config.model),
        "verify": VerifyAgent(config.provider, config.model),
    }


def _select_deduped(
    items: list[T],
    selected_ids: list[int],
    key_fn: Callable[[T], K],
    duplicate_message: str,
) -> tuple[list[T], list[T]]:
    selected_raw = _decode_one_indexed(items, selected_ids)
    selected = dedupe_by_key(selected_raw, key_fn=key_fn)
    if len(selected_raw) != len(selected):
        print(duplicate_message.format(raw=len(selected_raw), dedup=len(selected)))
    return selected_raw, selected


def run_pipeline(config: PipelineConfig | None = None) -> None:
    cfg = config or PipelineConfig()
    ctx = build_context(cfg)
    note = ctx["note"]
    true_codes = ctx["true_codes"]
    retriever = ctx["retriever"]

    _section("Running Full Pipeline", symbol="#")
    print(f"Dataset='{cfg.dataset_identifier}' sample_index={cfg.sample_index} sample_keys={list(ctx['sample'].keys())}")
    print(f"True codes for this note ({len(true_codes)}): {sorted(true_codes)}")
    print(f"Clinical note length: {len(note)} chars")
    print(
        f"Retrieval config: locate_top_k={cfg.locate_top_k}, assign_top_k={cfg.assign_top_k}, "
        f"verify_guideline_top_k={cfg.verify_guideline_top_k}, assign_term_pool={cfg.assign_term_pool_size}, "
        f"verify_code_pool={cfg.verify_code_pool_size}, retriever_terms={len(retriever.term_candidates)}"
    )

    _section("Step 1/4 - AnalyseAgent")
    analyse_terms = ctx["analyse"].run_single(note=note).terms
    print(f"Extracted terms ({len(analyse_terms)}): {analyse_terms}")

    _section("Step 2/4 - LocateAgent")
    print(f"Analyse queries ({len(analyse_terms)}): {analyse_terms}")
    retrieved_terms = retriever.retrieve_terms(analyse_terms, limit=cfg.locate_top_k)
    _print_terms(retrieved_terms, true_codes, "Retrieved term candidates", cfg.preview_rows)
    locate_result = ctx["locate"].run_single(note=note, terms=[{"path": term.path} for term in retrieved_terms])
    selected_terms_raw, selected_terms = _select_deduped(
        retrieved_terms,
        locate_result.term_ids,
        key_fn=lambda term: term.term_id,
        duplicate_message="Duplicate selected term IDs collapsed for downstream use ({raw} -> {dedup}).",
    )
    print(f"Selected term IDs from model: {locate_result.term_ids}")
    print(f"Selected term->code frequency: {dict(Counter(term.code for term in selected_terms_raw))}")
    _print_terms(selected_terms, true_codes, "Selected terms (deduped)", cfg.preview_rows)
    _print_code_overlap("Locate (via selected term codes)", (term.code for term in selected_terms), true_codes)

    terms_for_assign = merge_selected_with_retrieved(
        selected_terms,
        retrieved_terms,
        pool_size=cfg.assign_term_pool_size,
        key_fn=lambda term: term.term_id,
        code_fn=lambda term: term.code,
        stage_name="Assign",
        item_label="terms",
    )
    _print_code_overlap("Assign input pool (term codes)", (term.code for term in terms_for_assign), true_codes)

    _section("Step 3/4 - AssignAgent")
    print(f"Terms passed to assign ({len(terms_for_assign)}): {[f'{term.term_id}:{term.code}' for term in terms_for_assign]}")
    retrieved_codes = retriever.retrieve_codes(terms_for_assign, limit=cfg.assign_top_k)
    _print_codes(retrieved_codes, true_codes, "Retrieved code candidates", cfg.preview_rows)
    assign_result = ctx["assign"].run_single(
        note=note,
        codes=[code.model_dump(include={"code", "description", "path"}) for code in retrieved_codes],
    )
    _, selected_codes = _select_deduped(
        retrieved_codes,
        assign_result.code_ids,
        key_fn=lambda code: code.code,
        duplicate_message="Duplicate assigned codes collapsed ({raw} -> {dedup}).",
    )
    print(f"Assigned code IDs from model: {assign_result.code_ids}")
    _print_codes(selected_codes, true_codes, "Assigned codes (deduped)", cfg.preview_rows)
    _print_code_overlap("Assign", (code.code for code in selected_codes), true_codes)

    codes_for_verify = merge_selected_with_retrieved(
        selected_codes,
        retrieved_codes,
        pool_size=cfg.verify_code_pool_size,
        key_fn=lambda code: code.code,
        code_fn=lambda code: code.code,
        stage_name="Verify",
        item_label="codes",
    )
    _print_code_overlap("Verify input pool", (code.code for code in codes_for_verify), true_codes)

    _section("Step 4/4 - VerifyAgent")
    verify_names = [code.code for code in codes_for_verify]
    print(f"Codes passed to verify ({len(verify_names)}): {verify_names}")
    guidelines = [
        chunk.model_dump()
        for chunk in retriever.retrieve_guidelines(verify_names, limit=cfg.verify_guideline_top_k)
    ]
    instructional_notes = retriever.retrieve_instructional_notes(verify_names)
    print(f"Verify context: guideline_chunks={len(guidelines)} instructional_notes={len(instructional_notes)}")
    if guidelines:
        print(f"First guideline preview: {guidelines[0]['content'][:180]}...")
    verify_result = ctx["verify"].run_single(
        note=note,
        guidelines=guidelines,
        instructional_notes=instructional_notes,
        codes=[{"name": code.code, "description": code.description} for code in codes_for_verify],
    )
    _, verified_codes = _select_deduped(
        codes_for_verify,
        verify_result.code_ids,
        key_fn=lambda code: code.code,
        duplicate_message="Duplicate verified codes collapsed ({raw} -> {dedup}).",
    )
    print(f"Verified code IDs from model: {verify_result.code_ids}")
    _print_codes(verified_codes, true_codes, "Verified codes (deduped)", cfg.preview_rows)
    _print_code_overlap("Verify", (code.code for code in verified_codes), true_codes)

    _section("Pipeline Summary", symbol="#")
    print(f"Analyse -> {len(analyse_terms)} terms extracted")
    print(f"Locate -> selected term IDs: {[term.term_id for term in selected_terms]}")
    print(f"Locate -> selected term codes: {[term.code for term in selected_terms]}")
    print(f"Assign -> assigned codes: {[code.code for code in selected_codes]}")
    print(f"Verify -> verified codes: {[code.code for code in verified_codes]}")
    _print_code_overlap("Final Verify", (code.code for code in verified_codes), true_codes)


__all__ = ["PipelineConfig", "build_context", "dedupe_by_key", "merge_selected_with_retrieved", "run_pipeline"]


if __name__ == "__main__":
    run_pipeline()
