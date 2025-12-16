# import json
# from pathlib import Path
# import pandas as pd
# from workspace.utils.logging_utils import get_logger

# logger = get_logger("data.build_benchmark")

# def build_benchmark_from_parquet(
#     parquet_path: str,
#     out_jsonl_path: str,
#     max_examples: int | None = None,
# ) -> None:
#     df = pd.read_parquet(parquet_path)
#     # normalize expected columns
#     # prefer 'question','answer','context', plus ticker/report if present
#     outp = Path(out_jsonl_path)
#     outp.parent.mkdir(parents=True, exist_ok=True)

#     rows = []
#     for i, row in df.iterrows():
#         q = row.get("question")
#         a = row.get("answer")
#         ctx = row.get("context")
#         if not isinstance(q, str) or not isinstance(a, str) or not isinstance(ctx, str):
#             continue
#         meta = {}
#         for k in ["ticker", "symbol", "company", "year", "filing", "report", "form", "source"]:
#             if k in df.columns:
#                 meta[k] = row.get(k)
#         rows.append({
#             "id": str(row.get("id", f"ex_{i}")),
#             "question": q.strip(),
#             "answer": a.strip(),
#             "meta": meta,
#         })
#         if max_examples is not None and len(rows) >= max_examples:
#             break

#     with outp.open("w", encoding="utf-8") as f:
#         for r in rows:
#             f.write(json.dumps(r, ensure_ascii=False) + "\n")

#     logger.info(f"Saved benchmark {len(rows)} rows -> {outp}")

import json
import hashlib
from pathlib import Path
import pandas as pd
from workspace.utils.logging_utils import get_logger

logger = get_logger("data.build_benchmark")

def _normalize_ws(s: str) -> str:
    return " ".join((s or "").split())

def _hash_context(ctx: str) -> str:
    h = hashlib.sha1(ctx.encode("utf-8")).hexdigest()
    return f"ctxsha1_{h}"

def build_benchmark_from_parquet(
    parquet_path: str,
    out_jsonl_path: str,
    max_examples: int | None = None,
) -> None:
    df = pd.read_parquet(parquet_path)
    if "question" not in df.columns or "answer" not in df.columns or "context" not in df.columns:
        raise ValueError(f"Expected columns question/answer/context in {parquet_path}")

    outp = Path(out_jsonl_path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, row in df.iterrows():
        q = row.get("question")
        a = row.get("answer")
        ctx = row.get("context")

        if not isinstance(q, str) or not isinstance(a, str) or not isinstance(ctx, str):
            continue

        ctx_norm = _normalize_ws(ctx)
        gold_doc_id = _hash_context(ctx_norm)

        meta = {}
        for k in ["ticker", "symbol", "company", "year", "filing", "report", "form", "source"]:
            if k in df.columns:
                meta[k] = row.get(k)

        rows.append({
            "id": str(row.get("id", f"ex_{i}")),
            "question": q.strip(),
            "answer": a.strip(),
            "gold_doc_id": gold_doc_id,   # ✅ enables Recall/MRR
            "meta": meta,
        })

        if max_examples is not None and len(rows) >= max_examples:
            break

    with outp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"Saved benchmark {len(rows)} rows -> {outp}")
