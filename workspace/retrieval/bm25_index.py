import json
import re
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import joblib
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> List[str]:
    # simple + robust tokenization for filings
    text = text.lower()
    text = re.sub(r"[^a-z0-9%$ ]+", " ", text)
    return [t for t in text.split() if t]


def build_bm25_index(
    chunks_path: str,
    meta_out_path: str,
    index_out_path: str,
):
    """
    Builds a BM25 index over chunk texts.
    Saves:
      - meta_out_path: JSONL with {doc_id, chunk_id, text, meta}
      - index_out_path: joblib with {"bm25": BM25Okapi, "tokenized": List[List[str]]}
    """
    chunks_path = str(chunks_path)
    meta_out_path = str(meta_out_path)
    index_out_path = str(index_out_path)

    meta: List[Dict[str, Any]] = []
    tokenized: List[List[str]] = []

    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            text = rec["text"]
            meta.append({
                "doc_id": rec["doc_id"],
                "chunk_id": rec["chunk_id"],
                "text": text,
                "meta": rec.get("meta", {}),
            })
            tokenized.append(_tokenize(text))

    bm25 = BM25Okapi(tokenized)

    # write meta jsonl
    Path(meta_out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(meta_out_path, "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # save bm25 object + tokenized (bm25 stores corpus too, but keep explicit)
    Path(index_out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"bm25": bm25, "tokenized": tokenized}, index_out_path)

    return {"n_chunks": len(meta), "meta_out": meta_out_path, "index_out": index_out_path}
