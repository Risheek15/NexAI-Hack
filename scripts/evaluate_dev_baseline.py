"""
Reproducible DEV baseline evaluation script.
Measures: Hit@5, MRR, citation ID validity, quote-validation failure rate (raw output),
correct abstention, false refusal, latency p50/p95, and error rate on the DEV set ONLY.
Outputs:
  - evaluations/baseline/dev_benchmark.json
  - evaluations/baseline/dev_benchmark.md
"""

import sys
import time
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import LogbookRecord
from src.abbreviation.models import Glossary
from src.abbreviation.expander import expand_record
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.semantic_retriever import SemanticRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.agent.evidence_gate import assess_retrieval_evidence
from src.agent.validator import validate_citations_and_quotes


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def bootstrap_ci(values: List[float], n_bootstrap: int = 1000, ci: float = 0.95, seed: int = 42) -> Tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    arr = np.array(values)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_means.append(float(np.mean(sample)))
    lower = float(np.percentile(boot_means, (1 - ci) / 2 * 100))
    upper = float(np.percentile(boot_means, (1 + ci) / 2 * 100))
    return (round(lower, 4), round(upper, 4))


def run_dev_baseline_evaluation(
    records_jsonl: str = "data/processed/records.jsonl",
    records_lookup_json: str = "data/processed/records_lookup.json",
    glossary_json: str = "data/glossary.json",
    queries_json: str = "data/eval_queries_40.json",
    output_dir: str = "evaluations/baseline",
    seed: int = 42,
    top_k: int = 5
) -> Dict[str, Any]:
    np.random.seed(seed)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading canonical dataset and lookup index...")
    records: List[LogbookRecord] = []
    with open(records_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(LogbookRecord(**json.loads(line)))

    with open(records_lookup_json, "r", encoding="utf-8") as f:
        records_lookup: Dict[str, Dict[str, Any]] = json.load(f)

    glossary = Glossary.load_json(glossary_json)
    expanded_records = [expand_record(r, glossary) for r in records]
    valid_db_ids = set(records_lookup.keys())

    print(f"      Loaded {len(records):,} records, {len(valid_db_ids):,} lookup IDs, {len(glossary.get_validated_map())} validated glossary terms.")

    print("[2/5] Initializing retrieval modes...")
    bm25_base = BM25Retriever(records=records, index_path="data/processed/bm25_base.pkl")
    semantic_retriever = SemanticRetriever(records=records)
    hybrid_base = HybridRetriever(records=records, bm25_retriever=bm25_base, semantic_retriever=semantic_retriever)

    bm25_expanded = BM25Retriever(records=expanded_records, index_path="data/processed/bm25_expanded.pkl")
    hybrid_expanded = HybridRetriever(records=expanded_records, bm25_retriever=bm25_expanded, semantic_retriever=semantic_retriever)

    modes = [
        ("BM25 (Baseline)", bm25_base),
        ("Dense Semantic", semantic_retriever),
        ("Hybrid RRF (Baseline)", hybrid_base),
        ("Hybrid RRF (Abbr Expanded)", hybrid_expanded),
        ("BM25 (Abbr Expanded)", bm25_expanded)
    ]

    print("[3/5] Loading DEV queries (strictly split == 'dev')...")
    with open(queries_json, "r", encoding="utf-8") as f:
        all_queries = json.load(f)

    dev_queries = [q for q in all_queries if q.get("split") == "dev"]
    answerable_queries = [q for q in dev_queries if q.get("target_record_ids")]
    unanswerable_queries = [q for q in dev_queries if not q.get("target_record_ids")]

    n_total = len(dev_queries)
    n_ans = len(answerable_queries)
    n_unans = len(unanswerable_queries)

    print(f"      DEV queries total: {n_total} (Answerable: {n_ans}, Unanswerable: {n_unans})")

    print("[4/5] Evaluating all retrieval and grounding metrics on DEV set...")
    mode_evaluations: List[Dict[str, Any]] = []

    for mode_name, retriever in modes:
        latencies: List[float] = []
        hits_at_1: List[int] = []
        hits_at_3: List[int] = []
        hits_at_5: List[int] = []
        reciprocal_ranks: List[float] = []
        false_refusals = 0
        correct_abstentions = 0
        errors = 0

        total_cited_ids_count = 0
        valid_cited_ids_count = 0
        invalid_nonexistent_ids_count = 0
        unretrieved_ids_count = 0

        total_quotes_count = 0
        invalid_quotes_count = 0
        queries_with_quotes_count = 0
        queries_with_invalid_quotes_count = 0

        query_results = []

        # Evaluate Answerable Queries
        for q in answerable_queries:
            query_text = q["query"]
            targets = set(q["target_record_ids"])
            try:
                t0 = time.perf_counter()
                results = retriever.search(query_text, top_k=top_k)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                latencies.append(elapsed_ms)

                assessment, _, _ = assess_retrieval_evidence(query_text, results)
                is_false_refusal = (assessment.status == "insufficient")
                if is_false_refusal:
                    false_refusals += 1

                retrieved_ids = [r.record_id for r in results]
                session_retrieved_ids = set(retrieved_ids)

                hit_rank = None
                for idx, r_id in enumerate(retrieved_ids, start=1):
                    if r_id in targets:
                        hit_rank = idx
                        break

                if hit_rank is not None:
                    hits_at_1.append(1 if hit_rank == 1 else 0)
                    hits_at_3.append(1 if hit_rank <= 3 else 0)
                    hits_at_5.append(1 if hit_rank <= 5 else 0)
                    reciprocal_ranks.append(1.0 / hit_rank)
                else:
                    hits_at_1.append(0)
                    hits_at_3.append(0)
                    hits_at_5.append(0)
                    reciprocal_ranks.append(0.0)

                # Synthesize raw retrieval output before correction
                raw_lines = [
                    f"Evidence Assessment: {assessment.status.upper()} — {assessment.reason}",
                    "### Retrieved Historical Maintenance Records:"
                ]
                for r in results:
                    rec_lookup = records_lookup.get(r.record_id, {})
                    prob = rec_lookup.get("problem_raw", "")
                    act = rec_lookup.get("action_raw", "")
                    raw_lines.append(f"- Record [ID: {r.record_id}] reported: \"{prob}\" Action: \"{act}\"")

                raw_answer = "\n".join(raw_lines)

                # Validate citations & quotes on raw output
                val_res = validate_citations_and_quotes(
                    answer_text=raw_answer,
                    valid_db_ids=valid_db_ids,
                    session_retrieved_ids=session_retrieved_ids,
                    records_lookup=records_lookup
                )

                total_cited_ids_count += len(val_res.cited_ids)
                valid_cited_ids_count += sum(1 for cid in val_res.cited_ids if cid in valid_db_ids)
                invalid_nonexistent_ids_count += len(val_res.invalid_nonexistent_ids)
                unretrieved_ids_count += len(val_res.unretrieved_ids)

                from src.agent.validator import extract_direct_quotes
                direct_quotes = extract_direct_quotes(raw_answer)
                if direct_quotes:
                    queries_with_quotes_count += 1
                    total_quotes_count += len(direct_quotes)
                    if val_res.invalid_quotes:
                        queries_with_invalid_quotes_count += 1
                        invalid_quotes_count += len(val_res.invalid_quotes)

                query_results.append({
                    "query_id": q["query_id"],
                    "query": query_text,
                    "target_record_ids": q["target_record_ids"],
                    "retrieved_record_ids": retrieved_ids,
                    "hit_rank": hit_rank,
                    "latency_ms": round(elapsed_ms, 2),
                    "evidence_status": assessment.status
                })

            except Exception as e:
                errors += 1
                latencies.append(0.0)
                hits_at_1.append(0)
                hits_at_3.append(0)
                hits_at_5.append(0)
                reciprocal_ranks.append(0.0)

        # Evaluate Unanswerable Queries
        for q in unanswerable_queries:
            query_text = q["query"]
            try:
                t0 = time.perf_counter()
                results = retriever.search(query_text, top_k=top_k)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                latencies.append(elapsed_ms)

                assessment, _, _ = assess_retrieval_evidence(query_text, results)
                is_correct_abstention = (assessment.status == "insufficient")
                if is_correct_abstention:
                    correct_abstentions += 1

                retrieved_ids = [r.record_id for r in results]
                query_results.append({
                    "query_id": q["query_id"],
                    "query": query_text,
                    "target_record_ids": [],
                    "retrieved_record_ids": retrieved_ids,
                    "latency_ms": round(elapsed_ms, 2),
                    "evidence_status": assessment.status,
                    "abstained": is_correct_abstention
                })
            except Exception as e:
                errors += 1
                latencies.append(0.0)

        # Metric aggregations
        hit_5_count = sum(hits_at_5)
        hit_5_rate = hit_5_count / n_ans if n_ans else 0.0
        hit_5_ci = bootstrap_ci([float(x) for x in hits_at_5], seed=seed)

        mrr_val = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
        mrr_ci = bootstrap_ci(reciprocal_ranks, seed=seed)

        citation_id_validity = (valid_cited_ids_count / total_cited_ids_count) if total_cited_ids_count else 1.0

        quote_val_failure_rate = (
            (queries_with_invalid_quotes_count / queries_with_quotes_count)
            if queries_with_quotes_count else 0.0
        )

        corr_abstention_rate = correct_abstentions / n_unans if n_unans else 0.0
        false_refusal_rate = false_refusals / n_ans if n_ans else 0.0

        p50_latency = float(np.percentile(latencies, 50)) if latencies else 0.0
        p95_latency = float(np.percentile(latencies, 95)) if latencies else 0.0
        avg_latency = float(np.mean(latencies)) if latencies else 0.0

        error_rate = errors / n_total if n_total else 0.0

        mode_evaluations.append({
            "mode": mode_name,
            "sample_size_total": n_total,
            "sample_size_ans": n_ans,
            "sample_size_unans": n_unans,
            "hit_at_5_count": f"{hit_5_count}/{n_ans}",
            "hit_at_5": round(hit_5_rate, 4),
            "hit_at_5_bootstrap_ci_95": hit_5_ci,
            "mrr": round(mrr_val, 4),
            "mrr_bootstrap_ci_95": mrr_ci,
            "citation_id_validity_count": f"{valid_cited_ids_count}/{total_cited_ids_count}",
            "citation_id_validity": round(citation_id_validity, 4),
            "invalid_nonexistent_ids": invalid_nonexistent_ids_count,
            "unretrieved_cited_ids": unretrieved_ids_count,
            "quote_validation_failure_count": f"{queries_with_invalid_quotes_count}/{queries_with_quotes_count}",
            "quote_validation_failure_rate": round(quote_val_failure_rate, 4),
            "invalid_quotes_count": invalid_quotes_count,
            "correct_abstention_count": f"{correct_abstentions}/{n_unans}",
            "correct_abstention_rate": round(corr_abstention_rate, 4),
            "false_refusal_count": f"{false_refusals}/{n_ans}",
            "false_refusal_rate": round(false_refusal_rate, 4),
            "latency_p50_ms": round(p50_latency, 2),
            "latency_p95_ms": round(p95_latency, 2),
            "latency_avg_ms": round(avg_latency, 2),
            "error_count": f"{errors}/{n_total}",
            "error_rate": round(error_rate, 4),
            "detailed_queries": query_results
        })

    # Record hashes and metadata
    benchmark_meta = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "split_evaluated": "dev",
        "held_out_test_evaluated": False,
        "seed": seed,
        "config": {
            "embedding_model": "all-MiniLM-L6-v2",
            "gemini_model_configured": "gemini-2.5-flash",
            "max_tool_calls": 5,
            "rrf_k": 60,
            "top_k": top_k
        },
        "dependency_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__
        },
        "file_hashes_sha256": {
            "data/raw/logbooks.csv": compute_sha256(PROJECT_ROOT / "data/raw/logbooks.csv"),
            "data/processed/records.jsonl": compute_sha256(PROJECT_ROOT / "data/processed/records.jsonl"),
            "data/processed/records_lookup.json": compute_sha256(PROJECT_ROOT / "data/processed/records_lookup.json"),
            "data/processed/embedding_ids.json": compute_sha256(PROJECT_ROOT / "data/processed/embedding_ids.json"),
            "data/processed/embeddings.npy": compute_sha256(PROJECT_ROOT / "data/processed/embeddings.npy"),
            "data/glossary.json": compute_sha256(PROJECT_ROOT / "data/glossary.json"),
            "data/eval_queries_40.json": compute_sha256(PROJECT_ROOT / "data/eval_queries_40.json"),
            ".env": compute_sha256(PROJECT_ROOT / ".env")
        },
        "reliability_observations": [
            "Small sample size: DEV set has 20 queries (15 answerable, 5 unanswerable). Single answerable query flipping shifts metrics by 6.67%; single unanswerable flips abstention by 20.0%.",
            "Multiple targets per query: DEV_01 has 1 target while DEV_10, DEV_11, DEV_12 have 6-7 targets, conferring unequal base retrieval probabilities across query classes.",
            "Lexical query-target overlap: Queries in exact_technical_term directly mirror problem_raw/action_raw phrasing from source records.",
            "Test set strictly isolated: Held-out test split (20 queries) was NOT executed during this baseline freeze in strict adherence to Rule 2."
        ],
        "modes": mode_evaluations
    }

    # Save JSON artifact
    json_path = out_dir / "dev_benchmark.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_meta, f, indent=2)

    # Save Markdown Report artifact
    md_path = out_dir / "dev_benchmark.md"
    md_lines = [
        "# Frozen Baseline Benchmark Report — DEV Split Only",
        f"**Date:** {benchmark_meta['created_at']} | **Seed:** {seed} | **Split:** Development (N=20)",
        "",
        "## 1. DEV Set Empirical Results",
        "| Retrieval Mode | Hit@5 (Count) [95% CI] | MRR [95% CI] | Citation ID Validity | Quote Validation Fail Rate | Correct Abstention | False Refusal | Latency p50 / p95 | Error Rate |",
        "|---|---|---|---|---|---|---|---|---|"
    ]

    for m in mode_evaluations:
        hit5_str = f"{m['hit_at_5']*100:.1f}% ({m['hit_at_5_count']}) [{m['hit_at_5_bootstrap_ci_95'][0]*100:.1f}%, {m['hit_at_5_bootstrap_ci_95'][1]*100:.1f}%]"
        mrr_str = f"{m['mrr']:.4f} [{m['mrr_bootstrap_ci_95'][0]:.4f}, {m['mrr_bootstrap_ci_95'][1]:.4f}]"
        cite_str = f"{m['citation_id_validity']*100:.1f}% ({m['citation_id_validity_count']})"
        quote_str = f"{m['quote_validation_failure_rate']*100:.1f}% ({m['quote_validation_failure_count']})"
        abst_str = f"{m['correct_abstention_rate']*100:.1f}% ({m['correct_abstention_count']})"
        false_ref_str = f"{m['false_refusal_rate']*100:.1f}% ({m['false_refusal_count']})"
        lat_str = f"{m['latency_p50_ms']} ms / {m['latency_p95_ms']} ms"
        err_str = f"{m['error_rate']*100:.1f}% ({m['error_count']})"

        md_lines.append(
            f"| **{m['mode']}** | {hit5_str} | {mrr_str} | {cite_str} | {quote_str} | {abst_str} | {false_ref_str} | {lat_str} | {err_str} |"
        )

    md_lines.extend([
        "",
        "## 2. Configuration & Model Identifiers",
        f"- **Embedding Model:** `{benchmark_meta['config']['embedding_model']}` (Dense 384-d normalized unit vectors)",
        f"- **Gemini Model (Configured):** `{benchmark_meta['config']['gemini_model_configured']}`",
        f"- **Hybrid Fusion:** Reciprocal Rank Fusion ($k={benchmark_meta['config']['rrf_k']}$)",
        f"- **Evaluation Top-K:** `{benchmark_meta['config']['top_k']}`",
        f"- **Random Seed:** `{benchmark_meta['seed']}`",
        "",
        "## 3. Dataset & Artifact Integrity (SHA-256)",
        "| File | SHA-256 Hash |",
        "|---|---|"
    ])

    for f_path, h_val in benchmark_meta["file_hashes_sha256"].items():
        md_lines.append(f"| `{f_path}` | `{h_val}` |")

    md_lines.extend([
        "",
        "## 4. Reliability Analysis & Known Weaknesses",
        "1. **Small Sample Size ($N=20$, $N_{\\text{ans}}=15$, $N_{\\text{unans}}=5$):**",
        "   - Each answerable query accounts for $1/15 = 6.67\\%$ of Hit@5 and MRR.",
        "   - Each unanswerable query accounts for $1/5 = 20.0\\%$ of the abstention rate.",
        "2. **Target Imbalance:**",
        "   - Single-target queries (e.g. `DEV_01`, `DEV_02`, `DEV_06`, `DEV_08`) test strict precision.",
        "   - Multi-target queries (e.g. `DEV_10`, `DEV_11`, `DEV_12` with 6-7 targets) disproportionately elevate Hit@5.",
        "3. **Lexical Proximity:**",
        "   - Exact technical queries mirror the verbatim token sequences from `problem_raw` / `action_raw`.",
        "4. **Strict Isolation of Held-Out Test Set:**",
        "   - Test split was entirely excluded from this run to prevent any leakage or tuning."
    ])

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\n[5/5] Baseline evaluation complete!")
    print(f"      Saved: {json_path}")
    print(f"      Saved: {md_path}")
    return benchmark_meta


if __name__ == "__main__":
    run_dev_baseline_evaluation()
