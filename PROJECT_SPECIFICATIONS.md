# Project Specifications and Configurations

## Dataset Configuration

### Source Dataset
- **Dataset**: `virattt/financial-qa-10K` (HuggingFace)
- **Domain**: Financial QA based on SEC 10-K filings
- **Train Split**: 5,600 examples (80% of available data)
- **Test Split**: 1,400 examples (20% of available data)
- **Total Unique Contexts**: 2,881
- **Total Chunks Generated**: 2,888

### Data Processing
- **Train/Test Split**: Deterministic 80/20 split with `random_state=42`
- **Context Normalization**: Whitespace normalization applied to all contexts
- **Gold Document ID**: SHA1 hash of normalized context (`ctxsha1_{hash}`)

---

## Corpus Building Configuration

### Chunking Parameters
- **Chunk Size**: 1,800 characters
- **Chunk Overlap**: 200 characters
- **Minimum Chunk Size**: 200 characters
- **Chunking Method**: Character-based sliding window with overlap

### Corpus Statistics
- **Unique Contexts**: 2,881
- **Total Chunks**: 2,888
- **Output Format**: JSONL with fields: `doc_id`, `chunk_id`, `text`, `meta`

---

## Retrieval System Configuration

### Embedding Model
- **Model**: `BAAI/bge-small-en-v1.5`
- **Embedding Dimension**: 384
- **Normalization**: L2 normalization enabled (for cosine similarity)
- **Device**: CUDA (when available)

### FAISS Index
- **Index Type**: `IndexFlatIP` (Inner Product, equivalent to cosine similarity with normalized vectors)
- **Embedding Batch Size**: 64
- **Normalization**: Enabled (embeddings normalized to unit vectors)

### BM25 Index (Alternative)
- **Library**: `rank_bm25` (BM25Okapi)
- **Tokenization**: Lowercase, alphanumeric + `%$` characters only
- **Score Conversion**: Sigmoid function for confidence mapping
  - Formula: `1.0 / (1.0 + exp(-0.15 * (score - 8.0)))`
  - Maps BM25 scores to [0, 1] confidence range

---

## RAG System Configuration

### Retrieval Parameters
- **Top-K Retrieval**: 20 documents retrieved per query
- **Context-K**: 5 documents used for answer generation
- **Retrieval Method**: FAISS dense vector search (cosine similarity)

### Confidence Calculation
- **Generator Confidence Weight (α)**: 0.6
- **Retrieval Confidence Weight (1-α)**: 0.4
- **Combined Formula**: `α * gen_conf + (1-α) * retr_conf`
- **Confidence Range**: [0.0, 1.0] (clamped)

### Adaptive RAG Parameters
- **Confidence Threshold**: 0.5
- **Max Adaptive Steps**: 3 (configured in notebook; typically only 1 step needed in practice)
- **Rewrite Trigger**: When combined confidence < threshold
- **Early Stopping**: If confidence ≥ threshold after rewrite, return immediately
- **Note**: README mentions "at most 1 rewrite step" but code supports up to 3 steps with early stopping

### Query Rewriting
- **Temperature**: 0.0 (deterministic)
- **Max Tokens**: 64
- **Snippet Context**: Top 3 retrieved chunks (first 300 chars each)
- **Rewrite Guard**: Enabled with strict validation

### Rewrite Guard Validation
- **Forbid New Years**: Enabled (prevents adding years/dates not in original)
- **Forbid New Numbers**: Enabled (prevents adding numbers/amounts not in original)
- **Minimum Token Overlap**: 0.35 (35% of non-trivial tokens must overlap)
- **Safe Words Excluded**: Common stopwords and company suffixes (what, which, the, inc, corp, etc.)

---

## Generation Configuration

### Answer Generation
- **Temperature**: 0.0 (deterministic)
- **Max Tokens**: 128
- **Output Format**: Structured with `ANSWER:` and `CONFIDENCE:` lines
- **Confidence Extraction**: Regex parsing from model output
- **Confidence Range**: [0.0, 1.0] (clamped)

### Model Options

#### Local Generator (Default)
- **Base Model**: `Qwen/Qwen2.5-3B-Instruct`
- **Precision**: bfloat16
- **Device**: CUDA with `device_map="auto"`
- **LoRA Support**: Optional LoRA adapter loading
- **Sampling**: `top_p=0.95` when temperature > 0

#### OpenAI Generator (Optional)
- **Model**: `gpt-4o-mini`
- **Temperature**: 0.0 (deterministic) or 0.7 (sampled)
- **Max Tokens**: 128 (answer) or 64 (rewrite)

---

## DPO Training Configuration

### Dataset Generation
- **Candidate Pairs**: 300 generated from training set
- **Generation Method**: 
  - Candidate A: `temperature=0.0` (deterministic)
  - Candidate B: `temperature=0.7` (sampled)
- **Max Tokens per Candidate**: 128
- **Context**: Top 5 retrieved chunks (same as generation)
- **Retrieval for Candidates**: Top-K=20, Context-K=5

### LLM-as-Judge
- **Judge Model**: Same as generator (OpenAI or local)
- **Temperature**: 0.0 (deterministic)
- **Max Tokens**: 1 (single token: A, B, or TIE)
- **Decision Criteria**:
  - Correctness with respect to evidence
  - Numerical accuracy
  - Faithfulness (no hallucination)
  - Format compliance (ANSWER: and CONFIDENCE: lines)
- **Filtering**: Only keeps pairs with clear preference (A or B), discards TIE
- **Max Pairs Kept**: 200 (from 300 candidates, typically ~49-200 kept after filtering)

### DPO Training Hyperparameters
- **Base Model**: `Qwen/Qwen2.5-3B-Instruct`
- **Training Examples**: 49 (actual from notebook run; max_rows=200 configured but limited by judge filtering)
- **Batch Size**: 2
- **Gradient Accumulation**: 8 (effective batch size: 16)
- **Learning Rate**: 1e-5
- **Epochs**: 1
- **Beta (DPO temperature)**: 0.1
- **Max Prompt Length**: 1,024 tokens
- **Max Total Length**: 1,152 tokens
- **Precision**: bfloat16 (when CUDA available)
- **Device**: CUDA with `device_map="auto"`

### LoRA Configuration
- **LoRA Rank (r)**: 16
- **LoRA Alpha**: 32
- **LoRA Dropout**: 0.05
- **Bias**: None (not trained)
- **Target Modules**: `["q_proj", "k_proj", "v_proj", "o_proj"]`
- **Task Type**: Causal LM
- **PEFT Version**: 0.18.0

### Training Settings
- **Logging Steps**: 10
- **Save Steps**: 200
- **Reference Model**: None (uses implicit reference from base model)
- **Optimizer**: AdamW (default from TRL)
- **Mixed Precision**: bfloat16 enabled, fp16 disabled

---

## Evaluation Configuration

### Evaluation Metrics

#### Answer Quality Metrics
- **Exact Match (EM)**: Normalized (number-aware, case-insensitive)
- **Token F1**: Token-level F1 score
- **ROUGE-L**: Longest Common Subsequence-based F1

#### Retrieval Quality Metrics
- **Recall@K**: K ∈ {5, 10, 20}
- **MRR@K**: Mean Reciprocal Rank at K=20

#### Grounding Metrics
- **Support Overlap**: Binary check if normalized answer appears in evidence text
- **Evidence Context**: Top 5 retrieved chunks

#### Performance Metrics
- **Latency**: p50 (median) and p95 percentiles
- **Throughput**: Queries per second (QPS)

#### Adaptive Metrics
- **Adaptive Trigger Rate**: Percentage of queries that triggered rewriting
- **Average Rewrite Steps**: Mean number of rewrite steps when adaptive triggered

### Evaluation Settings
- **Test Set Size**: 200 examples (subset of full test set)
- **Context K for Grounding**: 5 chunks
- **Recall K Values**: [5, 10, 20]
- **MRR K**: 20

---

## System Variants Evaluated

1. **baseline_base**: Baseline RAG with base generator (no DPO)
2. **adaptive_base**: Adaptive RAG with base generator (no DPO)
3. **baseline_dpo**: Baseline RAG with DPO-aligned generator (LoRA adapter)
4. **adaptive_dpo**: Adaptive RAG with DPO-aligned generator (LoRA adapter)

---

## Software Dependencies

### Core Libraries
- **Python**: 3.11+
- **PyTorch**: ≥2.3.0
- **Transformers**: 4.44.2
- **Datasets**: 2.21.0
- **Pandas**: 2.2.2
- **NumPy**: 1.26.4

### Retrieval
- **FAISS**: 1.8.0.post1 (CPU version)
- **Sentence Transformers**: 3.0.1
- **rank-bm25**: (for BM25 indexing)

### RLHF/DPO
- **TRL**: 0.11.4
- **PEFT**: 0.12.0
- **Accelerate**: 0.33.0

### Optional
- **OpenAI**: 1.45.0 (for API-based generation/judging)

---

## Hardware Requirements

### Recommended
- **GPU**: CUDA-capable GPU (for local models)
- **Memory**: Sufficient for 3B parameter model + embeddings
- **Storage**: ~10GB for models, datasets, and artifacts

### Device Configuration
- **Embedding Device**: CUDA (when available)
- **Model Device**: CUDA with automatic device mapping
- **Precision**: bfloat16 for training and inference

---

## File Structure

### Artifacts Directory
```
artifacts/
├── benchmarks/          # Dataset parquet files and JSONL benchmarks
├── corpus/              # Chunked corpus (chunks.jsonl)
├── indexes/             # FAISS index and metadata
├── indexes_bm25/        # BM25 index (optional)
├── eval_runs/           # Evaluation results (JSON)
├── prefs/               # DPO dataset (candidates.jsonl, dpo_dataset.jsonl)
└── models/               # Trained LoRA adapters
    └── dpo_lora_v1/
```

---

## Key Implementation Details

### Confidence Score Conversion
- **BM25 to Confidence**: Sigmoid function `1 / (1 + exp(-0.15 * (score - 8.0)))`
- **Generator Confidence**: Extracted from model output via regex
- **Combined Confidence**: Weighted average with α=0.6 for generator

### Query Rewrite Safety
- **Validation**: Prevents introducing new constraints (years, numbers)
- **Topic Drift Protection**: Requires 35% token overlap
- **Fallback**: Returns original question if rewrite fails validation

### Answer Parsing
- **Format**: Two-line structured output
  - Line 1: `ANSWER: <text>`
  - Line 2: `CONFIDENCE: <float>`
- **Regex Patterns**: 
  - Answer: `^ANSWER:\s*(.*)$`
  - Confidence: `^CONFIDENCE:\s*([0-9]*\.?[0-9]+)\s*$`
- **Fallback**: `CANNOT_ANSWER` with confidence 0.0 if parsing fails

---

## Experimental Notes

- **Adaptive Steps**: Notebook uses `max_adaptive_steps=3` but typically only 1 step is needed in practice
- **DPO Dataset Size**: Limited by judge filtering - 49 examples kept from 300 candidates (16% retention rate)
- **Evaluation Subset**: 200 examples used for evaluation (from 1,400 test examples)
- **Corpus Coverage**: Corpus built from both train and test splits for better retrieval coverage
- **Judge Retention**: LLM-as-judge filters out TIE decisions, resulting in lower dataset size than candidate generation

