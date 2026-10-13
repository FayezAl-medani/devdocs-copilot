"""Benchmark evaluation script for TF-IDF and LSA search across hand-crafted queries."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from devdocs.corpus import load_markdown
from devdocs.metrics import mean_reciprocal_rank, recall_at_k
from devdocs.search import SemanticIndex

QUERIES_PATH = Path("eval/queries.jsonl")
DATA_DIR = Path("data/nestjs")


def load_eval_queries() -> list[dict[str, object]]:
    """Load JSONL evaluation queries."""
    if not QUERIES_PATH.exists():
        raise FileNotFoundError(f"Missing evaluation file: {QUERIES_PATH}")
    queries: list[dict[str, object]] = []
    for line in QUERIES_PATH.read_text(encoding="utf-8").splitlines():
        line_clean = line.strip()
        if line_clean:
            queries.append(json.loads(line_clean))
    return queries


def evaluate_index(
    index: SemanticIndex, queries: list[dict[str, object]], top_k_max: int = 10
) -> dict[str, float]:
    """Evaluate an index on queries, returning Recall@1, 5, 10, MRR, and avg latency."""
    all_retrieved: list[list[str]] = []
    all_targets: list[list[str]] = []

    start_time = time.perf_counter()
    for item in queries:
        query_text = str(item["query"])
        targets = [str(t) for t in item["target_ids"]]  # type: ignore[union-attr]

        hits = index.search(query_text, top_k=top_k_max)
        retrieved_ids = [h.chunk.id for h in hits]

        all_retrieved.append(retrieved_ids)
        all_targets.append(targets)

    total_latency_ms = (time.perf_counter() - start_time) * 1000.0
    avg_latency_ms = total_latency_ms / len(queries) if queries else 0.0

    r1 = sum(recall_at_k(r, t, 1) for r, t in zip(all_retrieved, all_targets, strict=True)) / len(
        queries
    )
    r5 = sum(recall_at_k(r, t, 5) for r, t in zip(all_retrieved, all_targets, strict=True)) / len(
        queries
    )
    r10 = sum(recall_at_k(r, t, 10) for r, t in zip(all_retrieved, all_targets, strict=True)) / len(
        queries
    )
    mrr_val = mean_reciprocal_rank(all_retrieved, all_targets)

    return {
        "recall@1": r1,
        "recall@5": r5,
        "recall@10": r10,
        "mrr": mrr_val,
        "latency_ms": avg_latency_ms,
    }


def main() -> None:
    """Run full benchmark comparison and display results table."""
    print("Loading NestJS document chunks...")
    chunks = load_markdown(DATA_DIR)
    print(f"Loaded {len(chunks)} chunks.")

    queries = load_eval_queries()
    print(f"Loaded {len(queries)} evaluation queries.")

    configurations = [
        ("TF-IDF", "tfidf", 0),
        ("LSA (k=64)", "lsa", 64),
        ("LSA (k=128)", "lsa", 128),
        ("LSA (k=256)", "lsa", 256),
    ]

    results: list[dict[str, str]] = []

    print("\n" + "=" * 80)
    print(
        f"{'Method':<16} | {'Recall@1':<10} | {'Recall@5':<10} | "
        f"{'Recall@10':<10} | {'MRR':<8} | {'Latency':<10}"
    )
    print("=" * 80)

    # Cache indices to inspect failure cases
    built_indices: dict[str, SemanticIndex] = {}

    for name, method, k in configurations:
        index = SemanticIndex(method=method, k=k if k > 0 else 128, min_df=1, max_features=8000)
        index.build(chunks)
        built_indices[name] = index

        metrics = evaluate_index(index, queries)

        r1_str = f"{metrics['recall@1'] * 100:.1f}%"
        r5_str = f"{metrics['recall@5'] * 100:.1f}%"
        r10_str = f"{metrics['recall@10'] * 100:.1f}%"
        mrr_str = f"{metrics['mrr']:.4f}"
        lat_str = f"{metrics['latency_ms']:.2f} ms"

        print(
            f"{name:<16} | {r1_str:<10} | {r5_str:<10} | "
            f"{r10_str:<10} | {mrr_str:<8} | {lat_str:<10}"
        )

        results.append(
            {
                "method": name,
                "r1": r1_str,
                "r5": r5_str,
                "r10": r10_str,
                "mrr": mrr_str,
                "latency": lat_str,
            }
        )

    print("\nMarkdown Table for README:")
    print("| الطريقة / النموذج | Recall@1 | Recall@5 | Recall@10 | MRR | زمن الاستعلام |")
    print("|---|---|---|---|---|---|")
    for r in results:
        print(
            f"| {r['method']} | {r['r1']} | {r['r5']} | {r['r10']} | {r['mrr']} | {r['latency']} |"
        )

    # Detailed Category Breakdown
    print("\n" + "=" * 80)
    print("Category Breakdown (Recall@5 & MRR):")
    print("=" * 80)
    categories = ["keyword_overlap", "paraphrased", "ambiguous_hard"]
    for cat in categories:
        cat_queries = [q for q in queries if q.get("category") == cat]
        print(f"\n--- Category: {cat} ({len(cat_queries)} queries) ---")
        for name, index in built_indices.items():
            metrics = evaluate_index(index, cat_queries)
            cat_r5 = f"{metrics['recall@5'] * 100:.1f}%"
            print(f"  {name:<15}: Recall@5 = {cat_r5}, MRR = {metrics['mrr']:.4f}")

    # Inspect notable failure cases to document in README
    print("\n" + "=" * 80)
    print("Notable Inspection Cases (Where TF-IDF vs LSA differed):")
    print("=" * 80)
    tfidf_idx = built_indices["TF-IDF"]
    lsa_idx = built_indices["LSA (k=128)"]

    for q_item in queries:
        q_text = str(q_item["query"])
        targets = set(str(t) for t in q_item["target_ids"])  # type: ignore[union-attr]

        hits_tf = [h.chunk.id for h in tfidf_idx.search(q_text, top_k=5)]
        hits_lsa = [h.chunk.id for h in lsa_idx.search(q_text, top_k=5)]

        tf_found = bool(set(hits_tf) & targets)
        lsa_found = bool(set(hits_lsa) & targets)

        if tf_found != lsa_found:
            print(f"\nQuery: '{q_text}' (Category: {q_item.get('category')})")
            print(f"  Targets: {targets}")
            print(f"  TF-IDF found? {tf_found} (Top hit: {hits_tf[0] if hits_tf else 'None'})")
            print(f"  LSA found?    {lsa_found} (Top hit: {hits_lsa[0] if hits_lsa else 'None'})")


if __name__ == "__main__":
    main()
