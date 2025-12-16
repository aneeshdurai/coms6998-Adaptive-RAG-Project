# Adaptive RAG for Financial QA (SEC 10-K) + Optional DPO Alignment

This repo implements an **end-to-end, reproducible** pipeline:
- Download a real financial QA dataset (`virattt/financial-qa-10K`)
- Build a retrieval corpus and FAISS index
- Run **Baseline RAG** vs **Adaptive RAG** (bounded to one rewrite step)
- Optionally create synthetic preferences (LLM-as-judge) and train a **DPO LoRA** adapter
- Evaluate variants and save results

## Quickstart (RunPod / local)
```bash
pip install -r requirements.txt
```

Then run the notebook:
- `notebooks/run_all.ipynb`

Artifacts will be written to `artifacts/`.

## Variants you can evaluate
- baseline_base: baseline RAG with base generator
- adaptive_base: adaptive RAG with base generator
- baseline_dpo: baseline RAG with DPO-aligned local generator (LoRA)
- adaptive_dpo: adaptive RAG with DPO-aligned local generator (LoRA)

## Notes
- Retrieval corpus is built from real 10-K-derived contexts in `virattt/financial-qa-10K`.
- Adaptive RAG triggers rewriting using a combined confidence score:
  - generator self-confidence + top-1 retrieval similarity
- Adaptive RAG performs **at most 1** rewrite/re-retrieve per query.
