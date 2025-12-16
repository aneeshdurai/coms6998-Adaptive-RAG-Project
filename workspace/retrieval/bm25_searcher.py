import json
import re
from typing import List, Dict, Any

import joblib
import numpy as np


def _tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9%$ ]+", " ", text)
    return [t for t in text.split() if t]


class BM25Searcher:
    def __init__(self, index_path: str, meta_path: str):
        payload = joblib.load(index_path)
        self.bm25 = payload["bm25"]

        self.meta = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                self.meta.append(json.loads(line))

        if len(self.meta) == 0:
            raise ValueError("BM25Searcher meta is empty. Did you build chunks/meta correctly?")

    def search(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        q_toks = _tokenize(query)
        scores = self.bm25.get_scores(q_toks)  # numpy array
        if not isinstance(scores, np.ndarray):
            scores = np.array(scores)

        top_idx = np.argsort(-scores)[:k]
        hits = []
        for ix in top_idx:
            m = self.meta[int(ix)]
            hits.append({
                "score": float(scores[int(ix)]),  # bm25 score (not cosine)
                "doc_id": m["doc_id"],
                "chunk_id": m["chunk_id"],
                "text": m.get("text"),
                "meta": m.get("meta", {}),
            })
        return hits
