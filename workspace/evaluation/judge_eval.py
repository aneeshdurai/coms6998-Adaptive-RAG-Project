import json
from pathlib import Path
from workspace.utils.logging_utils import get_logger

logger = get_logger("evaluation.judge_eval")

JUDGE_PROMPT = """You are judging two answers to a financial question using ONLY the provided evidence.

Question:
{question}

Gold answer (for reference; do not overfit formatting):
{gold}

Answer A:
{a}

Answer B:
{b}

Choose which answer is better based on:
- correctness,
- numerical accuracy,
- clarity,
- avoids hallucination.

Reply with exactly one token: A, B, or TIE.
"""

def judge_winrate(eval_json_a: str, eval_json_b: str, judge_client, out_path: str, n: int = 100):
    """
    eval_json_a/b are outputs from eval_runner (contain records).
    judge_client must have .complete(prompt, temperature=0.0, max_tokens=1).
    """
    a = json.loads(Path(eval_json_a).read_text(encoding="utf-8"))["records"]
    b = json.loads(Path(eval_json_b).read_text(encoding="utf-8"))["records"]

    # align by id
    b_map = {r["id"]: r for r in b}
    pairs = []
    for r in a:
        if r["id"] in b_map:
            pairs.append((r, b_map[r["id"]]))

    pairs = pairs[:n]
    wins_a = wins_b = ties = 0
    judged = 0
    out_rows = []

    for ra, rb in pairs:
        prompt = JUDGE_PROMPT.format(
            question=ra["question"],
            gold=ra["gold"],
            a=ra["pred"],
            b=rb["pred"],
        )
        dec = judge_client.complete(prompt, temperature=0.0, max_tokens=1).strip().upper()
        judged += 1
        if dec == "A":
            wins_a += 1
        elif dec == "B":
            wins_b += 1
        else:
            ties += 1
        out_rows.append({"id": ra["id"], "decision": dec})

    summary = {
        "judged": judged,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "winrate_a": wins_a / max(1, judged),
        "winrate_b": wins_b / max(1, judged),
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps({"summary": summary, "decisions": out_rows}, indent=2), encoding="utf-8")
    logger.info(f"Judge win-rate saved -> {out_path}")
    return summary
