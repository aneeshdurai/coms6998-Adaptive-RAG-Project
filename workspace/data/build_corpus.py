# import json
# from pathlib import Path
# from typing import Iterable, Dict, Any, List, Set
# import pandas as pd
# from workspace.utils.logging_utils import get_logger

# logger = get_logger("data.build_corpus")

# def _normalize_ws(s: str) -> str:
#     return " ".join((s or "").split())

# def build_corpus_from_parquet(
#     parquet_paths: List[str],
#     out_chunks_path: str,
#     chunk_size_chars: int = 1800,
#     chunk_overlap_chars: int = 200,
#     min_chars: int = 200,
# ) -> None:
#     '''
#     Build a retrieval corpus by:
#     - reading parquet files containing (question, answer, context, meta)
#     - de-duplicating contexts
#     - chunking contexts into overlapping character chunks

#     Output JSONL: {doc_id, chunk_id, text, meta}
#     '''
#     contexts: List[Dict[str, Any]] = []
#     seen: Set[str] = set()

#     for p in parquet_paths:
#         df = pd.read_parquet(p)
#         # Many variants exist; we try common column names
#         # Expect at least: question, answer, context; plus ticker/company/year/report when available
#         for i, row in df.iterrows():
#             ctx = _normalize_ws(str(row.get("context", "")))
#             if len(ctx) < min_chars:
#                 continue
#             if ctx in seen:
#                 continue
#             seen.add(ctx)
#             meta = {}
#             for k in ["ticker", "symbol", "company", "year", "filing", "report", "form", "source"]:
#                 if k in row:
#                     meta[k] = row.get(k)
#             contexts.append({"context": ctx, "meta": meta})

#     logger.info(f"Unique contexts collected: {len(contexts)}")

#     outp = Path(out_chunks_path)
#     outp.parent.mkdir(parents=True, exist_ok=True)
#     n_chunks = 0

#     with outp.open("w", encoding="utf-8") as f:
#         for di, rec in enumerate(contexts):
#             doc_id = f"ctx_{di}"
#             text = rec["context"]
#             meta = rec.get("meta", {})
#             # simple char chunking (fast, deterministic)
#             start = 0
#             cid = 0
#             while start < len(text):
#                 end = min(len(text), start + chunk_size_chars)
#                 chunk = text[start:end].strip()
#                 if len(chunk) >= min_chars:
#                     f.write(json.dumps({
#                         "doc_id": doc_id,
#                         "chunk_id": cid,
#                         "text": chunk,
#                         "meta": meta,
#                     }, ensure_ascii=False) + "\n")
#                     n_chunks += 1
#                     cid += 1
#                 if end == len(text):
#                     break
#                 start = max(0, end - chunk_overlap_chars)

#     logger.info(f"Wrote chunks: {n_chunks} -> {outp}")

import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Set
import pandas as pd
from workspace.utils.logging_utils import get_logger

logger = get_logger("data.build_corpus")

def _normalize_ws(s: str) -> str:
    return " ".join((s or "").split())

def _hash_context(ctx: str) -> str:
    # Stable doc_id for retrieval metrics
    h = hashlib.sha1(ctx.encode("utf-8")).hexdigest()
    return f"ctxsha1_{h}"

def build_corpus_from_parquet(
    parquet_paths: List[str],
    out_chunks_path: str,
    chunk_size_chars: int = 1800,
    chunk_overlap_chars: int = 200,
    min_chars: int = 200,
) -> None:
    """
    Build a retrieval corpus from dataset contexts.
    Output JSONL: {doc_id, chunk_id, text, meta}
    doc_id is sha1 of full normalized context for reproducible evidence matching.
    """
    contexts: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    for p in parquet_paths:
        df = pd.read_parquet(p)

        if "context" not in df.columns:
            raise ValueError(f"Parquet missing 'context' column: {p}")

        for _, row in df.iterrows():
            ctx = _normalize_ws(str(row.get("context", "")))
            if len(ctx) < min_chars:
                continue
            if ctx in seen:
                continue
            seen.add(ctx)

            meta = {}
            for k in ["ticker", "symbol", "company", "year", "filing", "report", "form", "source"]:
                if k in df.columns:
                    meta[k] = row.get(k)

            contexts.append({"context": ctx, "meta": meta})

    logger.info(f"Unique contexts collected: {len(contexts)}")

    outp = Path(out_chunks_path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    n_chunks = 0
    with outp.open("w", encoding="utf-8") as f:
        for rec in contexts:
            text = rec["context"]
            doc_id = _hash_context(text)
            meta = rec.get("meta", {})

            start = 0
            cid = 0
            while start < len(text):
                end = min(len(text), start + chunk_size_chars)
                chunk = text[start:end].strip()
                if len(chunk) >= min_chars:
                    f.write(json.dumps({
                        "doc_id": doc_id,
                        "chunk_id": cid,
                        "text": chunk,
                        "meta": meta,
                    }, ensure_ascii=False) + "\n")
                    n_chunks += 1
                    cid += 1
                if end == len(text):
                    break
                start = max(0, end - chunk_overlap_chars)

    logger.info(f"Wrote chunks: {n_chunks} -> {outp}")
