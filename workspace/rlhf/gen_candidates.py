import json
from pathlib import Path
from typing import Dict, Any, List
from workspace.utils.logging_utils import get_logger
from workspace.models.prompts import ANSWER_PROMPT
from workspace.models.parser import parse_answer_and_conf

logger = get_logger("rlhf.gen_candidates")

def gen_candidates(
    dataset_jsonl: str,
    searcher,
    store,
    chat_client,
    out_path: str,
    n: int = 500,
    top_k: int = 20,
    context_k: int = 5,
) -> None:
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    def build_context(hits):
        parts = []
        for h in hits[:context_k]:
            txt = store.get_text(h["doc_id"], h["chunk_id"])
            parts.append(txt)
        return "\n\n".join(parts)

    count = 0
    with open(dataset_jsonl, "r", encoding="utf-8") as f, outp.open("w", encoding="utf-8") as w:
        for line in f:
            ex = json.loads(line)
            q = ex["question"]
            hits = searcher.search(q, k=top_k)
            ctx = build_context(hits)
            prompt = ANSWER_PROMPT.format(question=q, context=ctx)

            # Two candidates: deterministic and slightly sampled
            a0 = chat_client.complete(prompt, temperature=0.0, max_tokens=128)
            a1 = chat_client.complete(prompt, temperature=0.7, max_tokens=128)

            w.write(json.dumps({
                "id": ex["id"],
                "question": q,
                "context": ctx,
                "cand_a": a0,
                "cand_b": a1,
            }, ensure_ascii=False) + "\n")
            count += 1
            if count >= n:
                break
    logger.info(f"Wrote {count} candidate pairs -> {outp}")
