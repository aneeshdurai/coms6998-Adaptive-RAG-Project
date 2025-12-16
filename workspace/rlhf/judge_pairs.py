import json
from pathlib import Path
from workspace.utils.logging_utils import get_logger

logger = get_logger("rlhf.judge_pairs")

JUDGE_PROMPT = """You are judging two answers to a financial question using ONLY the provided evidence.

Evidence:
{context}

Question:
{question}

Answer A:
{a}

Answer B:
{b}

Choose which answer is better based on:
- correctness with respect to the evidence,
- numerical accuracy when applicable,
- faithfulness (no hallucination),
- following format: exactly two lines: ANSWER: ... and CONFIDENCE: ...

Reply with exactly one token: A, B, or TIE.
"""

def judge_pairs(candidates_jsonl: str, judge_client, out_dpo_jsonl: str, max_pairs: int = 500) -> None:
    outp = Path(out_dpo_jsonl)
    outp.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    total = 0

    with open(candidates_jsonl, "r", encoding="utf-8") as f, outp.open("w", encoding="utf-8") as w:
        for line in f:
            rec = json.loads(line)
            prompt = JUDGE_PROMPT.format(
                context=rec["context"],
                question=rec["question"],
                a=rec["cand_a"],
                b=rec["cand_b"],
            )
            decision = judge_client.complete(prompt, temperature=0.0, max_tokens=1).strip().upper()
            total += 1
            if decision not in ("A", "B"):
                continue
            chosen = rec["cand_a"] if decision == "A" else rec["cand_b"]
            rejected = rec["cand_b"] if decision == "A" else rec["cand_a"]
            w.write(json.dumps({
                "prompt": f"Evidence:\n{rec['context']}\n\nQuestion:\n{rec['question']}\n",
                "chosen": chosen,
                "rejected": rejected,
            }, ensure_ascii=False) + "\n")
            kept += 1
            if kept >= max_pairs:
                break

    logger.info(f"Judged {total} pairs, kept {kept} -> {outp}")
