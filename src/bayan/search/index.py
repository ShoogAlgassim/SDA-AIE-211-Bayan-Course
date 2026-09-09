"""Lab 5: versioned FAISS index build."""

import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from bayan.preprocessing.core import PREPROC_VERSION, preprocess


DATA_PATH = Path("data/search/bayan_cases.csv")
DEFAULT_MODEL = "intfloat/multilingual-e5-base"


def build_index(
    prefix,
    limit=None,
    model_name=DEFAULT_MODEL,
):
    prefix = Path(prefix)

    df = pd.read_csv(DATA_PATH)

    if limit is not None:
        df = df.head(limit).copy()

    if len(df) == 0:
        raise ValueError("No cases available to index")

    texts = [
        "passage: " + preprocess(text)
        for text in df["case_text"].astype(str).tolist()
    ]

    model = SentenceTransformer(model_name)

    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=64,
    )

    vectors = np.asarray(vectors, dtype="float32")

    faiss.normalize_L2(vectors)

    dim = vectors.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    index_path = Path(f"{prefix}.faiss")
    index_path.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_path))

    metadata = df[
        [
            "case_id",
            "lang",
            "topic",
            "case_text",
            "resolution",
            "status",
        ]
    ].to_dict(orient="records")

    metadata_path = Path(f"{prefix}_metadata.json")

    metadata_path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest = {
        "model": model_name,
        "preproc_version": PREPROC_VERSION,
        "n_vectors": int(index.ntotal),
        "dim": int(dim),
        "document_prefix": "passage: ",
        "query_prefix": "query: ",
    }

    manifest_path = Path(f"{prefix}_manifest.json")

    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Index built successfully")
    print("Vectors:", index.ntotal)
    print("Dimension:", dim)
    print("Model:", model_name)
    print("Preprocessing version:", PREPROC_VERSION)

    return {
        "index_path": str(index_path),
        "metadata_path": str(metadata_path),
        "manifest_path": str(manifest_path),
    }
