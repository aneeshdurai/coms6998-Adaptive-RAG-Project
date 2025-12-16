# Step-by-Step Tutorial: Adaptive RAG for Financial QA

This tutorial will guide you through setting up and running the complete Adaptive RAG pipeline from scratch.

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Download Dataset](#2-download-dataset)
3. [Build Retrieval Corpus](#3-build-retrieval-corpus)
4. [Build Retrieval Index](#4-build-retrieval-index)
5. [Run Baseline RAG](#5-run-baseline-rag)
6. [Run Adaptive RAG](#6-run-adaptive-rag)
7. [Train DPO (Optional)](#7-train-dpo-optional)
8. [Evaluate Systems](#8-evaluate-systems)
9. [Use the UI](#9-use-the-ui)

---

## 1. Environment Setup

### 1.1 System Requirements

- **OS**: Linux, macOS, or Windows
- **Python**: 3.11 or higher
- **GPU**: CUDA-capable GPU recommended (optional)
- **RAM**: 16GB minimum
- **Disk**: 10GB free space

### 1.2 Install Python Dependencies

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### 1.3 Verify Installation

```bash
# Test imports
python -c "
import torch
import transformers
import peft
import streamlit
print('✅ All dependencies installed successfully!')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

### 1.4 Set Environment Variables (Optional)

```bash
# For OpenAI API (optional)
export OPENAI_API_KEY="your-key-here"

# For HuggingFace cache (optional)
export HF_HOME="/path/to/cache"
```

---

## 2. Download Dataset

The system uses the `virattt/financial-qa-10K` dataset from HuggingFace.

### 2.1 Using Python API

```python
from workspace.data.download_dataset import download_financial_qa_10k
from pathlib import Path

# Create artifacts directory
data_dir = Path("artifacts/benchmarks")
data_dir.mkdir(parents=True, exist_ok=True)

# Download dataset
download_financial_qa_10k(
    hf_id="virattt/financial-qa-10K",
    out_dir=str(data_dir),
    splits=("train", "test")
)
```

### 2.2 Expected Output

```
artifacts/benchmarks/
├── virattt__financial-qa-10K_train.parquet
└── virattt__financial-qa-10K_test.parquet
```

### 2.3 Create Test/Train Benchmarks

```python
from workspace.data.build_benchmark import build_benchmark_from_parquet

# Build train benchmark
build_benchmark_from_parquet(
    parquet_path="artifacts/benchmarks/virattt__financial-qa-10K_train.parquet",
    out_jsonl_path="artifacts/benchmarks/train.jsonl"
)

# Build test benchmark (limit to 200 examples for faster evaluation)
build_benchmark_from_parquet(
    parquet_path="artifacts/benchmarks/virattt__financial-qa-10K_test.parquet",
    out_jsonl_path="artifacts/benchmarks/test.jsonl",
    max_examples=200
)
```

---

## 3. Build Retrieval Corpus

Convert the dataset contexts into chunked documents for retrieval.

### 3.1 Build Corpus from Parquet Files

```python
from workspace.data.build_corpus import build_corpus_from_parquet

chunks_path = "artifacts/corpus/chunks.jsonl"

build_corpus_from_parquet(
    parquet_paths=[
        "artifacts/benchmarks/virattt__financial-qa-10K_train.parquet",
        "artifacts/benchmarks/virattt__financial-qa-10K_test.parquet"
    ],
    out_chunks_path=chunks_path,
    chunk_size_chars=1800,
    chunk_overlap_chars=200,
    min_chars=200
)
```

### 3.2 Corpus Statistics

Expected output:
- **Unique contexts**: ~2,881
- **Total chunks**: ~2,888
- **Chunk format**: `{doc_id, chunk_id, text, meta}`

### 3.3 Inspect Corpus

```python
import json

# Read first chunk
with open("artifacts/corpus/chunks.jsonl", "r") as f:
    first_chunk = json.loads(f.readline())
    print("Sample chunk:")
    print(json.dumps(first_chunk, indent=2))
```

---

## 4. Build Retrieval Index

Create a BM25 index for fast keyword-based retrieval.

### 4.1 Build BM25 Index

```python
from workspace.retrieval.bm25_index import build_bm25_index

result = build_bm25_index(
    chunks_path="artifacts/corpus/chunks.jsonl",
    meta_out_path="artifacts/indexes_bm25/chunks_meta.jsonl",
    index_out_path="artifacts/indexes_bm25/bm25_index.joblib"
)

print(f"✅ Built BM25 index with {result['n_chunks']} chunks")
```

### 4.2 Test Retrieval

```python
from workspace.retrieval.bm25_searcher import BM25Searcher

# Load searcher
searcher = BM25Searcher(
    index_path="artifacts/indexes_bm25/bm25_index.joblib",
    meta_path="artifacts/indexes_bm25/chunks_meta.jsonl"
)

# Test search
hits = searcher.search("What was Apple's revenue?", k=5)
for i, hit in enumerate(hits, 1):
    print(f"{i}. Score: {hit['score']:.2f} | {hit['text'][:100]}...")
```

---

## 5. Run Baseline RAG

Standard RAG without adaptive query rewriting.

### 5.1 Initialize Components

```python
from workspace.retrieval.corpus_store import CorpusStore
from workspace.models.local_generator import LocalChat
from workspace.models.rag import RAGGenerator, QueryRewriter, AdaptiveRAG, RagConfig

# Load corpus store
store = CorpusStore("artifacts/corpus/chunks.jsonl")

# Initialize local LLM
chat = LocalChat(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    device="cuda"  # or "cpu" if no GPU
)

# Create generator and rewriter
generator = RAGGenerator(chat)
rewriter = QueryRewriter(chat)

# Configure RAG
cfg = RagConfig(
    top_k=20,
    context_k=5,
    confidence_threshold=0.5,
    max_adaptive_steps=1,
    alpha_gen=0.6
)

# Create adaptive RAG system
rag = AdaptiveRAG(searcher, store, generator, rewriter, cfg)
```

### 5.2 Ask a Question (Baseline)

```python
# Use baseline mode (no rewriting)
result = rag.answer_baseline("What was Microsoft's revenue in 2023?")

print(f"Question: {result['question']}")
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Generator Confidence: {result['gen_conf']:.2f}")
print(f"Retrieval Confidence: {result['retr_conf']:.2f}")
```

---

## 6. Run Adaptive RAG

RAG with confidence-based adaptive query rewriting.

### 6.1 Ask a Question (Adaptive)

```python
# Use adaptive mode (with rewriting if confidence < threshold)
result = rag.answer_adaptive("What was Microsoft's revenue in 2023?")

print(f"Question: {result['question']}")
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Adaptive Triggered: {result['adaptive_triggered']}")

if result['adaptive_triggered']:
    print(f"Rewritten Query: {result['rewritten_query']}")
    
    # Show rewrite history
    if 'rewrite_history' in result:
        for step in result['rewrite_history']:
            print(f"  Step {step['step']}: {step['query']} (conf={step['confidence']:.2f})")
```

### 6.2 Compare Baseline vs Adaptive

```python
questions = [
    "What was Apple's debt in 2023?",
    "How many employees did Microsoft have?",
    "What was Amazon's operating income?"
]

for q in questions:
    baseline = rag.answer_baseline(q)
    adaptive = rag.answer_adaptive(q)
    
    print(f"\n📊 Question: {q}")
    print(f"Baseline: {baseline['answer'][:100]}... (conf={baseline['confidence']:.2f})")
    print(f"Adaptive: {adaptive['answer'][:100]}... (conf={adaptive['confidence']:.2f})")
    print(f"Rewritten: {adaptive.get('adaptive_triggered', False)}")
```

---

## 7. Train DPO (Optional)

Fine-tune the generator using Direct Preference Optimization.

### 7.1 Generate Candidate Pairs

```python
from workspace.rlhf.gen_candidates import gen_candidates

# Generate 300 candidate pairs (temp=0.0 vs temp=0.7)
gen_candidates(
    dataset_jsonl="artifacts/benchmarks/train.jsonl",
    searcher=searcher,
    store=store,
    chat_client=chat,
    out_path="artifacts/prefs/candidates.jsonl",
    n=300,
    top_k=20,
    context_k=5
)
```

### 7.2 Judge Pairs with LLM

```python
from workspace.rlhf.judge_pairs import judge_pairs

# Use LLM-as-judge to create preferences
judge_pairs(
    candidates_jsonl="artifacts/prefs/candidates.jsonl",
    judge_client=chat,  # or OpenAIChat for better judging
    out_dpo_jsonl="artifacts/prefs/dpo_dataset.jsonl",
    max_pairs=200
)
```

### 7.3 Train DPO LoRA Adapter

```python
from workspace.rlhf.train_dpo_lora import train_dpo_lora

# Train LoRA adapter on preferences
dpo_output = train_dpo_lora(
    dpo_jsonl="artifacts/prefs/dpo_dataset.jsonl",
    base_model_name="Qwen/Qwen2.5-3B-Instruct",
    output_dir="artifacts/models/dpo_lora_v1",
    batch_size=2,
    grad_accum=8,
    lr=1e-5,
    epochs=1,
    beta=0.1,
    max_prompt_length=1024,
    max_length=1152
)

print(f"✅ DPO LoRA saved to: {dpo_output}")
```

### 7.4 Load DPO Model

```python
# Initialize chat with DPO adapter
chat_dpo = LocalChat(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    device="cuda",
    lora_path="artifacts/models/dpo_lora_v1"
)

# Create new RAG with DPO generator
generator_dpo = RAGGenerator(chat_dpo)
rag_dpo = AdaptiveRAG(searcher, store, generator_dpo, rewriter, cfg)

# Test DPO model
result_dpo = rag_dpo.answer_baseline("What was Microsoft's revenue in 2023?")
print(f"DPO Answer: {result_dpo['answer']}")
```

---

## 8. Evaluate Systems

Run systematic evaluation on test set.

### 8.1 Evaluate Baseline

```python
from workspace.evaluation.eval_runner import run_eval

# Evaluate baseline RAG
summary_baseline = run_eval(
    benchmark_jsonl="artifacts/benchmarks/test.jsonl",
    answer_fn=rag.answer_baseline,
    out_path="artifacts/eval_runs/test_baseline_base.json",
    store=store,
    max_examples=200,
    context_k_for_grounding=5,
    recall_ks=(5, 10, 20),
    mrr_k=20
)

print("\n📊 Baseline Results:")
for key, value in summary_baseline.items():
    if isinstance(value, (int, float)) and key != "n":
        print(f"  {key}: {value:.4f}")
```

### 8.2 Evaluate Adaptive

```python
# Evaluate adaptive RAG
summary_adaptive = run_eval(
    benchmark_jsonl="artifacts/benchmarks/test.jsonl",
    answer_fn=rag.answer_adaptive,
    out_path="artifacts/eval_runs/test_adaptive_base.json",
    store=store,
    max_examples=200
)

print("\n📊 Adaptive Results:")
for key, value in summary_adaptive.items():
    if isinstance(value, (int, float)) and key != "n":
        print(f"  {key}: {value:.4f}")
```

### 8.3 Compare Results

```python
import json

# Load results
with open("artifacts/eval_runs/test_baseline_base.json") as f:
    baseline_data = json.load(f)
with open("artifacts/eval_runs/test_adaptive_base.json") as f:
    adaptive_data = json.load(f)

# Compare metrics
metrics = ["EM_norm", "F1", "ROUGE_L", "Recall@5", "MRR@20", "Adaptive_trigger_rate"]

print("\n📊 Baseline vs Adaptive Comparison:")
print(f"{'Metric':<25} {'Baseline':<12} {'Adaptive':<12} {'Δ':<10}")
print("-" * 60)

for metric in metrics:
    b_val = baseline_data['summary'].get(metric, 0)
    a_val = adaptive_data['summary'].get(metric, 0)
    delta = a_val - b_val
    print(f"{metric:<25} {b_val:<12.4f} {a_val:<12.4f} {delta:+.4f}")
```

---

## 9. Use the UI

### 9.1 Launch Streamlit UI

```bash
python run_ui.py
```

The UI will open at `http://localhost:8501`

### 9.2 UI Features

1. **Model Selection**:
   - Choose between OpenAI API or local models
   - Load DPO LoRA adapters (checkbox)

2. **RAG Configuration**:
   - Confidence threshold slider
   - Max adaptive steps
   - Generator confidence weight (α)

3. **Query Interface**:
   - Ask questions in natural language
   - Compare Baseline vs Adaptive results
   - View retrieved documents
   - See confidence scores and rewrite history

4. **Artifact Management**:
   - Auto-build missing artifacts
   - Monitor system status

### 9.3 Example Queries

Try these queries in the UI:

```
1. "What was Apple's revenue in 2023?"
2. "How many employees did Microsoft have at year end?"
3. "What factors affected Amazon's operating income?"
4. "What was the company's effective tax rate?"
5. "How much debt did the company repay in 2023?"
```

---

## Next Steps

- **Experiment with different configurations**: Adjust `confidence_threshold`, `alpha_gen`, etc.
- **Try different models**: Switch to larger models for better performance
- **Add your own data**: Adapt the pipeline to your domain
- **Fine-tune further**: Increase DPO training data for better alignment

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues and solutions.

---

**Congratulations!** 🎉 You've successfully set up and run the complete Adaptive RAG pipeline.
