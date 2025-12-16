import json
from pathlib import Path
from typing import Dict, Any, List
from workspace.evaluation.metrics import exact_match_norm, token_f1, answer_in_evidence

def pick_examples(
    eval_json_path: str,
    out_path: str,
    min_top1_gain: float = 0.05,
    min_conf_gain: float = 0.10,
    k: int = 10,
) -> Dict[str, Any]:
    data = json.loads(Path(eval_json_path).read_text(encoding="utf-8"))
    recs = data["records"]

    good = []
    for r in recs:
        if not r.get("adaptive_triggered"):
            continue

        # baseline signals
        b_top1 = r.get("baseline_top1_score")
        b_conf = r.get("baseline_confidence")
        b_ans = r.get("baseline_answer")

        # final signals
        a_top1 = r.get("top1_score")
        a_conf = r.get("confidence")
        a_ans = r.get("pred")

        if b_top1 is None or a_top1 is None or b_conf is None or a_conf is None:
            continue

        top1_gain = a_top1 - b_top1
        conf_gain = a_conf - b_conf

        if top1_gain < min_top1_gain or conf_gain < min_conf_gain:
            continue

        gold = r.get("gold", "")

        # Improvement criteria: EM/F1 improved OR CANNOT_ANSWER resolved
        b_em = exact_match_norm(b_ans, gold)
        a_em = exact_match_norm(a_ans, gold)
        b_f1 = token_f1(b_ans, gold)
        a_f1 = token_f1(a_ans, gold)

        improved = (a_em > b_em) or (a_f1 > b_f1) or (b_ans == "CANNOT_ANSWER" and a_ans != "CANNOT_ANSWER")
        if not improved:
            continue

        good.append({
            "id": r.get("id"),
            "question": r.get("question"),
            "gold": gold,
            "baseline": {
                "answer": b_ans,
                "confidence": b_conf,
                "top1_score": b_top1,
                "evidence": r.get("baseline_evidence_snippet"),
            },
            "adaptive": {
                "answer": a_ans,
                "confidence": a_conf,
                "top1_score": a_top1,
                "evidence": r.get("evidence_snippet"),
                "rewritten_query": r.get("rewritten_query"),
                "rewrite_history": r.get("rewrite_history"),
            },
            "deltas": {
                "top1_gain": top1_gain,
                "conf_gain": conf_gain,
                "em_gain": a_em - b_em,
                "f1_gain": a_f1 - b_f1,
            }
        })

    # Sort by confidence gain then top1 gain
    good.sort(key=lambda x: (x["deltas"]["conf_gain"], x["deltas"]["top1_gain"]), reverse=True)
    out = {"picked": good[:k], "total_candidates": len(good)}

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
