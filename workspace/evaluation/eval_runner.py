# import json
# from pathlib import Path
# from typing import Callable, Dict, Any
# from workspace.utils.logging_utils import get_logger
# from workspace.evaluation.metrics import exact_match, token_f1

# logger = get_logger("evaluation.eval_runner")

# def run_eval(
#     benchmark_jsonl: str,
#     answer_fn: Callable[[str], Dict[str, Any]],
#     out_path: str,
#     max_examples: int | None = 200,
# ) -> Dict[str, Any]:
#     outp = Path(out_path)
#     outp.parent.mkdir(parents=True, exist_ok=True)

#     records = []
#     ems = []
#     f1s = []
#     n = 0

#     with open(benchmark_jsonl, "r", encoding="utf-8") as f:
#         for line in f:
#             ex = json.loads(line)
#             q = ex["question"]
#             gold = ex["answer"]
#             res = answer_fn(q)
#             pred = res["answer"]
#             em = exact_match(pred, gold)
#             f1 = token_f1(pred, gold)
#             records.append({
#                 "id": ex["id"],
#                 "question": q,
#                 "gold": gold,
#                 "pred": pred,
#                 "confidence": res.get("confidence"),
#                 "gen_conf": res.get("gen_conf"),
#                 "retr_conf": res.get("retr_conf"),
#                 "adaptive_triggered": res.get("adaptive_triggered"),
#                 "rewritten_query": res.get("rewritten_query"),
#             })
#             ems.append(em)
#             f1s.append(f1)
#             n += 1
#             if max_examples and n >= max_examples:
#                 break

#     summary = {
#         "n": n,
#         "EM": float(sum(ems) / max(1, n)),
#         "F1": float(sum(f1s) / max(1, n)),
#     }
#     logger.info(f"Eval: EM={summary['EM']:.3f} F1={summary['F1']:.3f} (n={n})")

#     with outp.open("w", encoding="utf-8") as w:
#         json.dump({"summary": summary, "records": records}, w, ensure_ascii=False, indent=2)

#     logger.info(f"Saved eval -> {outp}")
#     return summary

import json
import time
import statistics
from pathlib import Path
from typing import Callable, Dict, Any, Optional, List

from workspace.utils.logging_utils import get_logger
from workspace.evaluation.metrics import (
    exact_match_norm, token_f1, rouge_l,
    answer_in_evidence, recall_at_k, mrr_at_k
)

logger = get_logger("evaluation.eval_runner")

def _build_evidence_text(hits: List[Dict[str, Any]], store, context_k: int = 5) -> str:
    parts = []
    for h in hits[:context_k]:
        parts.append(store.get_text(h["doc_id"], h["chunk_id"]))
    return "\n\n".join(parts)

def run_eval(
    benchmark_jsonl: str,
    answer_fn: Callable[[str], Dict[str, Any]],
    out_path: str,
    store=None,                 # pass CorpusStore for grounding metrics
    max_examples: int | None = 200,
    context_k_for_grounding: int = 5,
    recall_ks=(5,10,20),
    mrr_k=20,
) -> Dict[str, Any]:
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    records = []
    ems, f1s, rouges, supports = [], [], [], []
    latencies = []
    recalls = {k: [] for k in recall_ks}
    mrrs = []

    n = 0
    triggered = 0
    rewrite_steps = []

    t0_all = time.perf_counter()

    with open(benchmark_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            q = ex["question"]
            gold = ex["answer"]
            gold_doc_id = ex.get("gold_doc_id")

            t0 = time.perf_counter()
            res = answer_fn(q)
            t1 = time.perf_counter()
            lat = t1 - t0
            latencies.append(lat)

            pred = res["answer"]
            hits = res.get("hits", [])

            # Answer metrics
            em = exact_match_norm(pred, gold)
            f1 = token_f1(pred, gold)
            rg = rouge_l(pred, gold)
            ems.append(em); f1s.append(f1); rouges.append(rg)

            # Retrieval metrics (offline; uses gold_doc_id)
            for k in recall_ks:
                recalls[k].append(recall_at_k(hits, gold_doc_id, k))
            mrrs.append(mrr_at_k(hits, gold_doc_id, mrr_k))

            # Grounding / support overlap
            if store is not None:
                evidence_text = _build_evidence_text(hits, store, context_k=context_k_for_grounding)
                supports.append(answer_in_evidence(pred, evidence_text))

            # Adaptive stats
            if res.get("adaptive_triggered"):
                triggered += 1
                # if looped rewrites exist
                hist = res.get("rewrite_history") or []
                rewrite_steps.append(len(hist) if hist else 1)

            records.append({
                "id": ex.get("id"),
                "question": q,
                "gold": gold,
                "pred": pred,
                "gold_doc_id": gold_doc_id,
                "confidence": res.get("confidence"),
                "gen_conf": res.get("gen_conf"),
                "retr_conf": res.get("retr_conf"),
                "adaptive_triggered": res.get("adaptive_triggered"),
                "rewritten_query": res.get("rewritten_query"),
                "rewrite_history": res.get("rewrite_history"),
                "latency_s": lat,
                "top1_score": hits[0]["score"] if hits else None,
            })

            n += 1
            if max_examples and n >= max_examples:
                break

    total_time = time.perf_counter() - t0_all
    throughput = n / total_time if total_time > 0 else 0.0

    summary = {
        "n": n,
        "EM_norm": float(sum(ems) / max(1, n)),
        "F1": float(sum(f1s) / max(1, n)),
        "ROUGE_L": float(sum(rouges) / max(1, n)),
        "SupportOverlap": float(sum(supports) / max(1, len(supports))) if supports else None,

        **{f"Recall@{k}": float(sum(recalls[k]) / max(1, n)) for k in recall_ks},
        f"MRR@{mrr_k}": float(sum(mrrs) / max(1, n)),

        "Latency_p50_s": float(statistics.median(latencies)) if latencies else None,
        "Latency_p95_s": float(statistics.quantiles(latencies, n=20)[-1]) if len(latencies) >= 20 else None,
        "Throughput_qps": float(throughput),

        "Adaptive_trigger_rate": float(triggered / max(1, n)),
        "Adaptive_avg_steps": float(sum(rewrite_steps)/len(rewrite_steps)) if rewrite_steps else 0.0,
    }

    logger.info(
        "Eval "
        + " ".join([f"{k}={v:.3f}" for k, v in summary.items() if isinstance(v, (int, float)) and v is not None and k != "n"])
        + f" (n={n})"
    )

    with outp.open("w", encoding="utf-8") as w:
        json.dump({"summary": summary, "records": records}, w, ensure_ascii=False, indent=2)

    logger.info(f"Saved eval -> {outp}")
    return summary
