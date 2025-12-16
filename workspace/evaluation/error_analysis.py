import json
from pathlib import Path
from workspace.utils.logging_utils import get_logger
from workspace.evaluation.metrics import exact_match_norm

logger = get_logger("evaluation.error_analysis")

def analyze(eval_base_json: str, eval_adapt_json: str, out_path: str, k: int = 10):
    base = json.loads(Path(eval_base_json).read_text(encoding="utf-8"))["records"]
    adap = json.loads(Path(eval_adapt_json).read_text(encoding="utf-8"))["records"]
    amap = {r["id"]: r for r in adap}

    helped = []
    hurt = []
    gen_bottleneck = []
    retrieval_bottleneck = []

    for b in base:
        a = amap.get(b["id"])
        if not a:
            continue
        gold = b["gold"]

        b_ok = exact_match_norm(b["pred"], gold) == 1.0
        a_ok = exact_match_norm(a["pred"], gold) == 1.0

        b_retr_ok = (b.get("gold_doc_id") is not None) and (b.get("gold_doc_id") == b.get("gold_doc_id"))  # placeholder
        # Use doc_id match via stored top hits if you want; simplest:
        # if gold_doc_id appears in retrieved hits
        def retrieved_gold(rec):
            gid = rec.get("gold_doc_id")
            if not gid:
                return False
            # You didn't store hits in record to keep file small; if you want it, add.
            return False

        if (not b_ok) and a_ok:
            helped.append({"id": b["id"], "question": b["question"], "gold": gold, "baseline": b["pred"], "adaptive": a["pred"], "rewritten": a.get("rewritten_query")})
        if b_ok and (not a_ok):
            hurt.append({"id": b["id"], "question": b["question"], "gold": gold, "baseline": b["pred"], "adaptive": a["pred"], "rewritten": a.get("rewritten_query")})

    out = {
        "helped_examples": helped[:k],
        "hurt_examples": hurt[:k],
        "helped_count": len(helped),
        "hurt_count": len(hurt),
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=2), encoding="utf-8")
    logger.info(f"Error analysis saved -> {out_path}")
    return out
