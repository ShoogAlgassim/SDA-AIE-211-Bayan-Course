"""Lab 5: labelled-query retrieval evaluation."""

import json
import time
from pathlib import Path

import faiss

from bayan.search.index import build_index
from bayan.search.service import CaseSearch
from bayan.preprocessing.core import preprocess


INDEX_PREFIX = "artifacts/case_index_v1"
QUERY_PATH = Path("data/search/bayan_queries.jsonl")


def load_queries():
    with QUERY_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def recall_at_k(retrieved_ids, relevant_ids, k=10):
    relevant = set(relevant_ids)
    return float(any(cid in relevant for cid in retrieved_ids[:k]))


def reciprocal_rank(retrieved_ids, relevant_ids, k=10):
    relevant = set(relevant_ids)

    for rank, cid in enumerate(retrieved_ids[:k], start=1):
        if cid in relevant:
            return 1.0 / rank

    return 0.0


def mean(values):
    return sum(values) / len(values) if values else 0.0


def main():
    print("Building FAISS index...")

    build_index(
        prefix=INDEX_PREFIX,
        limit=None,
    )

    print("Loading search service...")
    searcher = CaseSearch(INDEX_PREFIX)

    queries = load_queries()

    answerable = [q for q in queries if not q["no_answer"]]
    no_answer = [q for q in queries if q["no_answer"]]

    bi_recalls = []
    bi_mrrs = []

    rerank_recalls = []
    rerank_mrrs = []

    lang_scores = {}

    bi_latencies = []
    rerank_latencies = []

    print("Evaluating retrieval...")

    for q in answerable:
        query_text = q["query"]
        relevant = q["relevant_case_ids"]

        # Bi-encoder retrieval only
        query_prefix = searcher.manifest.get("query_prefix", "")
        normalized = query_prefix + preprocess(query_text)

        start = time.perf_counter()

        vec = searcher.encoder.encode(
            [normalized],
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")

        faiss.normalize_L2(vec)

        _, indices = searcher.index.search(vec, 10)

        bi_latencies.append(time.perf_counter() - start)

        bi_ids = [
            searcher.metadata[idx]["case_id"]
            for idx in indices[0]
            if idx >= 0
        ]

        bi_recalls.append(
            recall_at_k(bi_ids, relevant)
        )

        bi_mrrs.append(
            reciprocal_rank(bi_ids, relevant)
        )

        # Two-stage retrieval + reranking
        start = time.perf_counter()

        reranked = searcher.search(
            query_text,
            k=10,
            candidates=50,
            min_score=-999.0,
        )

        rerank_latencies.append(time.perf_counter() - start)

        reranked_ids = [
            result["case_id"]
            for result in reranked
        ]

        recall = recall_at_k(
            reranked_ids,
            relevant,
        )

        mrr = reciprocal_rank(
            reranked_ids,
            relevant,
        )

        rerank_recalls.append(recall)
        rerank_mrrs.append(mrr)

        lang_scores.setdefault(q["lang"], []).append(recall)

    print("\n=== Retrieval Metrics ===")
    print(f"Recall@10 without reranking: {mean(bi_recalls):.4f}")
    print(f"MRR@10 without reranking: {mean(bi_mrrs):.4f}")
    print(f"Recall@10 with reranking: {mean(rerank_recalls):.4f}")
    print(f"MRR@10 with reranking: {mean(rerank_mrrs):.4f}")

    print("\n=== Language Slice ===")

    lang_means = {}

    for lang, values in lang_scores.items():
        lang_means[lang] = mean(values)
        print(f"{lang} Recall@10: {lang_means[lang]:.4f}")

    if "ar" in lang_means and "en" in lang_means:
        gap = abs(lang_means["ar"] - lang_means["en"])
        print(f"Cross-lingual slice gap: {gap:.4f}")

    print("\n=== No-answer Threshold ===")

    thresholds = [0.0, 0.1, 0.25, 0.5, 0.75]

    best_threshold = None
    best_correct = -1

    for threshold in thresholds:
        correct = 0

        for q in no_answer:
            results = searcher.search(
                q["query"],
                k=5,
                candidates=50,
                min_score=threshold,
            )

            if not results:
                correct += 1

        print(
            f"threshold={threshold:.2f} "
            f"correct={correct}/{len(no_answer)}"
        )

        if correct > best_correct:
            best_correct = correct
            best_threshold = threshold

    print(f"Best no-answer threshold: {best_threshold}")
    print(f"No-answer correctness: {best_correct}/{len(no_answer)}")

    print("\n=== Stage Latency ===")
    print(
        f"Bi-encoder average latency: "
        f"{mean(bi_latencies) * 1000:.2f} ms"
    )

    print(
        f"Two-stage average latency: "
        f"{mean(rerank_latencies) * 1000:.2f} ms"
    )


if __name__ == "__main__":
    main()
