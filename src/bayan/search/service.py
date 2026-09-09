"""Lab 5: two-stage bilingual case search."""

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

from bayan.preprocessing.core import preprocess


DEFAULT_RERANKER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class CaseSearch:
    def __init__(self, prefix: str):
        prefix = Path(prefix)

        self.index_path = Path(f"{prefix}.faiss")
        self.metadata_path = Path(f"{prefix}_metadata.json")
        self.manifest_path = Path(f"{prefix}_manifest.json")

        for path in [
            self.index_path,
            self.metadata_path,
            self.manifest_path,
        ]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing search artifact: {path}"
                )

        self.manifest = json.loads(
            self.manifest_path.read_text(
                encoding="utf-8"
            )
        )

        required = [
            "model",
            "preproc_version",
            "n_vectors",
            "dim",
        ]

        for key in required:
            if key not in self.manifest:
                raise ValueError(
                    f"Manifest missing required field: {key}"
                )

        self.index = faiss.read_index(
            str(self.index_path)
        )

        self.metadata = json.loads(
            self.metadata_path.read_text(
                encoding="utf-8"
            )
        )

        if self.index.ntotal != self.manifest["n_vectors"]:
            raise ValueError(
                "Manifest/index mismatch: n_vectors does not match FAISS index"
            )

        if self.index.d != self.manifest["dim"]:
            raise ValueError(
                "Manifest/index mismatch: dimension does not match FAISS index"
            )

        if len(self.metadata) != self.manifest["n_vectors"]:
            raise ValueError(
                "Manifest/metadata mismatch: metadata length does not match n_vectors"
            )

        self.encoder = SentenceTransformer(
            self.manifest["model"]
        )

        self.reranker = CrossEncoder(
            DEFAULT_RERANKER
        )

    def search(
        self,
        query: str,
        k: int = 5,
        candidates: int = 50,
        min_score: float = 0.25,
    ):
        query = preprocess(query)

        if not query.strip():
            return []

        query_prefix = self.manifest.get(
            "query_prefix",
            ""
        )

        encoded_query = (
            query_prefix + query
        )

        query_vector = self.encoder.encode(
            [encoded_query],
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        query_vector = np.asarray(
            query_vector,
            dtype="float32",
        )

        faiss.normalize_L2(query_vector)

        n_candidates = min(
            candidates,
            self.index.ntotal,
        )

        bi_scores, bi_indices = self.index.search(
            query_vector,
            n_candidates,
        )

        candidate_rows = []

        for score, idx in zip(
            bi_scores[0],
            bi_indices[0],
        ):
            if idx < 0:
                continue

            item = dict(
                self.metadata[idx]
            )

            candidate_rows.append(
                {
                    "metadata": item,
                    "bi_score": float(score),
                }
            )

        if not candidate_rows:
            return []

        pairs = [
            [
                query,
                row["metadata"]["case_text"],
            ]
            for row in candidate_rows
        ]

        ce_scores = self.reranker.predict(
            pairs,
            show_progress_bar=False,
        )

        for row, ce_score in zip(
            candidate_rows,
            ce_scores,
        ):
            row["score"] = float(
                ce_score
            )

        candidate_rows.sort(
            key=lambda x: x["score"],
            reverse=True,
        )

        filtered = [
            row
            for row in candidate_rows
            if row["score"] >= min_score
        ]

        if not filtered:
            return []

        results = []

        for row in filtered[:k]:
            result = dict(
                row["metadata"]
            )

            result["bi_score"] = (
                row["bi_score"]
            )

            result["score"] = (
                row["score"]
            )

            results.append(result)

        return results
