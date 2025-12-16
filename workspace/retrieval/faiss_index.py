import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import faiss
from workspace.utils.logging_utils import get_logger
from workspace.retrieval.embeddings import Embedder

logger = get_logger("retrieval.faiss_index")

def load_chunks(chunks_path: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    texts: List[str] = []
    meta: List[Dict[str, Any]] = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            texts.append(rec["text"])
            meta.append({k: rec.get(k) for k in ["doc_id", "chunk_id", "meta"]})
    return texts, meta

def build_faiss_index(
    chunks_path: str,
    meta_out_path: str,
    index_out_path: str,
    #embeddings_out_path: str | None,
    model_name: str,
    device: str = "cuda",
    batch_size: int = 64,
    normalize: bool = True,
    embeddings_out_path: str | None = None,
) -> None:
    texts, meta = load_chunks(chunks_path)
    logger.info(f"Loaded chunks: {len(texts)}")

    if len(texts) == 0:
        raise ValueError(f"No chunks found at {chunks_path}. Build corpus first.")

    embedder = Embedder(model_name=model_name, device=device)
    logger.info(f"Embedding chunks with {model_name}")
    embs = embedder.embed(texts, batch_size=batch_size, normalize=normalize)
    dim = embs.shape[1]
    logger.info(f"Embeddings shape: {embs.shape}")

    index = faiss.IndexFlatIP(dim)  # cosine if normalized
    index.add(embs)

    Path(meta_out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(meta_out_path, "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    Path(index_out_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_out_path)

    if embeddings_out_path:
        Path(embeddings_out_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(embeddings_out_path, embs)

    logger.info(f"Saved meta -> {meta_out_path}")
    logger.info(f"Saved index -> {index_out_path}")
    if embeddings_out_path:
        logger.info(f"Saved embeddings -> {embeddings_out_path}")

class FaissSearcher:
    def __init__(self, index_path: str, meta_path: str, embed_model: str, device: str = "cuda"):
        self.index = faiss.read_index(index_path)
        self.meta = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                self.meta.append(json.loads(line))
        self.embedder = Embedder(embed_model, device=device)

    def search(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        q_emb = self.embedder.embed([query], batch_size=1, normalize=True)
        scores, idxs = self.index.search(q_emb, k)
        out = []
        for s, i in zip(scores[0].tolist(), idxs[0].tolist()):
            if i < 0:
                continue
            m = self.meta[i]
            out.append({"score": float(s), **m})
        return out
