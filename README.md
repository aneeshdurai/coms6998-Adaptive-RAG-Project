# Adaptive RAG for Financial QA

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A confidence-aware Retrieval-Augmented Generation (RAG) system for financial question answering with optional Direct Preference Optimization (DPO) alignment.

## 🌟 Key Features

- **Adaptive Query Rewriting**: Automatically rewrites queries when confidence is low to improve retrieval
- **Dual Retrieval**: Supports both BM25 (keyword-based) and FAISS (dense vector) retrieval
- **DPO Fine-tuning**: Optional preference optimization using LLM-as-judge for improved answer quality
- **Confidence-Based Decisions**: Combines generator self-confidence and retrieval similarity scores
- **Safety Guards**: Prevents query rewrites from hallucinating facts (years, numbers, etc.)
- **Interactive UI**: Streamlit-based web interface for easy experimentation
- **Reproducible Experiments**: Complete pipeline from data to evaluation

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [System Architecture](#system-architecture)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [DPO Training](#dpo-training)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

## 🚀 Installation

### Prerequisites

- Python 3.11 or higher
- CUDA-capable GPU (optional, but recommended for local models)
- 16GB+ RAM
- 10GB+ disk space

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd coms6998-Adaptive-RAG-Project
```

### Step 2: Create Virtual Environment (Required)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**⚠️ Important**: The virtual environment is **required** for the UI launcher (`run_ui.py`). The launcher script automatically detects and uses the venv Python interpreter to ensure all dependencies are properly loaded. Running the UI without a venv may cause import errors.

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note**: Updated package versions for compatibility:
- `openai>=2.12.0` (was 1.45.0)
- `peft>=0.18.0` (was 0.12.0)
- `transformers>=4.48.0` (was 4.44.2)

## ⚡ Quick Start

### Option 1: Interactive UI (Recommended)

```bash
python run_ui.py
```

**What it does**:
- Automatically uses venv Python (required for proper dependency loading)
- Opens Streamlit UI at `http://localhost:8501`
- No need to manually activate venv - the script handles it

**UI Features**:
- Ask questions about financial documents
- Compare Baseline vs Adaptive RAG side-by-side
- Load DPO-aligned models (if trained)
- View retrieved documents and confidence scores
- Real-time confidence tracking and adaptive step visualization
- Configure retrieval and generation parameters

### Option 2: Run Complete Pipeline (Jupyter Notebook)

We provide two notebooks for different use cases:

#### Main Notebook (Recommended): `notebooks/run_all.ipynb`
Complete experimental pipeline with comprehensive evaluation

```bash
jupyter notebook notebooks/run_all.ipynb
```

**Full pipeline includes**:
1. **Data Preparation**: Download and process financial-qa-10K dataset
2. **Corpus Building**: Create searchable document chunks
3. **Index Creation**: Build BM25 and FAISS indexes
4. **Baseline RAG**: Standard retrieval-augmented generation
5. **Adaptive RAG**: Confidence-aware query rewriting
6. **DPO Training**: Preference optimization (optional)
7. **Comprehensive Evaluation**: All variants on test & noisy benchmarks
8. **Presentation Examples**: Generate demo cases
9. **Summary Tables**: Aggregate metrics and analysis

**Artifacts generated** (stored in `artifacts/`):
- `benchmarks/`: Test/train datasets, noisy benchmarks
- `corpus/chunks.jsonl`: Processed document chunks
- `indexes_bm25/`: BM25 index + metadata
- `indexes/`: FAISS index + metadata (if using dense retrieval)
- `prefs/`: DPO candidate pairs and training data
- `models/dpo_lora_v1/`: Trained DPO adapter
- `eval_runs/`: JSON result files for each experiment

#### Quick Start Notebook: `notebooks/run_all_basic.ipynb`
Streamlined pipeline for quick testing

Same core steps as main notebook, but without:
- Noisy benchmark evaluation
- Presentation examples
- Extended analysis

**Use this for**: Quick experimentation, debugging, or when you don't need full evaluation.

### Option 3: Python API

```python
from workspace.models.rag import AdaptiveRAG, RAGGenerator, QueryRewriter, RagConfig
from workspace.retrieval.bm25_searcher import BM25Searcher
from workspace.retrieval.corpus_store import CorpusStore
from workspace.models.local_generator import LocalChat

# Load retrieval components
searcher = BM25Searcher(
    index_path="artifacts/indexes_bm25/bm25_index.joblib",
    meta_path="artifacts/indexes_bm25/chunks_meta.jsonl"
)
store = CorpusStore("artifacts/corpus/chunks.jsonl")

# Initialize generator
chat = LocalChat(model_name="Qwen/Qwen2.5-3B-Instruct")
generator = RAGGenerator(chat)
rewriter = QueryRewriter(chat)

# Create adaptive RAG system
cfg = RagConfig(
    top_k=20,
    context_k=5,
    confidence_threshold=0.5,
    max_adaptive_steps=1,
    alpha_gen=0.6
)
rag = AdaptiveRAG(searcher, store, generator, rewriter, cfg)

# Ask a question
result = rag.answer_adaptive("What was Apple's revenue in 2023?")
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Adaptive triggered: {result['adaptive_triggered']}")
```

## 🏗️ System Architecture

### Components

```
┌─────────────────────────────────────────────┐
│           Adaptive RAG System               │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────┐    ┌──────────────┐         │
│  │  Query   │───▶│  Retriever   │         │
│  │          │    │  (BM25/FAISS)│         │
│  └──────────┘    └──────┬───────┘         │
│                         │                  │
│                         ▼                  │
│                  ┌──────────────┐         │
│                  │  Generator   │         │
│                  │ (LLM+Context)│         │
│                  └──────┬───────┘         │
│                         │                  │
│                         ▼                  │
│           ┌─────────────────────────┐     │
│           │ Confidence < Threshold? │     │
│           └──────┬──────────┬───────┘     │
│                  │ No       │ Yes         │
│                  ▼          ▼             │
│            [Return]   ┌──────────┐       │
│                       │ Rewriter │       │
│                       └─────┬────┘       │
│                             │             │
│                             ▼             │
│                       (Repeat loop)       │
└─────────────────────────────────────────────┘
```

### Models Used

#### Language Models
- **Default Generator**: `Qwen/Qwen2.5-3B-Instruct` (Alibaba Cloud)
  - 3 billion parameters
  - Instruction-tuned for chat-based tasks
  - Supports LoRA adapters for DPO fine-tuning
  - Used for: Answer generation, query rewriting, LLM-as-judge
  
- **Optional**: OpenAI models via API
  - `gpt-4o-mini` (default for OpenAI mode)
  - Configurable in UI and code
  - Used when local GPU unavailable or for comparison

#### Retrieval Models
- **BM25** (default): TF-IDF based keyword matching
  - Fast, no GPU required
  - Excellent for exact term matches
  - Used in all experiments
  
- **FAISS** (optional): Dense vector retrieval
  - Uses `sentence-transformers/all-MiniLM-L6-v2`
  - GPU-accelerated similarity search
  - Better for semantic matching

### Key Modules

- **`workspace/models/rag.py`**: Core RAG logic (baseline & adaptive)
- **`workspace/models/local_generator.py`**: Local LLM wrapper with LoRA support
- **`workspace/models/rewrite_guard.py`**: Query rewrite validation
- **`workspace/retrieval/`**: BM25 and FAISS retrievers
- **`workspace/rlhf/`**: DPO training pipeline
- **`workspace/evaluation/`**: Evaluation metrics and runners

## 📊 Usage

### Running Experiments

See [TUTORIAL.md](TUTORIAL.md) for detailed step-by-step instructions.

### Evaluation Metrics

The system tracks:
- **Answer Quality**: Exact Match, Token F1, ROUGE-L
- **Retrieval Quality**: Recall@K, MRR@K
- **Grounding**: Support overlap (answer in evidence)
- **Performance**: Latency (p50, p95), Throughput (QPS)
- **Adaptive**: Trigger rate, average rewrite steps

### System Variants

1. **baseline_base**: Baseline RAG + base generator
2. **adaptive_base**: Adaptive RAG + base generator
3. **baseline_dpo**: Baseline RAG + DPO-aligned generator
4. **adaptive_dpo**: Adaptive RAG + DPO-aligned generator ⭐

## 🎯 DPO Training

Optional preference optimization to improve answer quality:

```python
# 1. Generate candidate pairs (deterministic vs sampled)
from workspace.rlhf.gen_candidates import gen_candidates
gen_candidates(
    dataset_jsonl="artifacts/benchmarks/train.jsonl",
    searcher=searcher,
    store=store,
    chat_client=chat,
    out_path="artifacts/prefs/candidates.jsonl",
    n=300
)

# 2. Judge pairs using LLM-as-judge
from workspace.rlhf.judge_pairs import judge_pairs
judge_pairs(
    candidates_jsonl="artifacts/prefs/candidates.jsonl",
    judge_client=chat,
    out_dpo_jsonl="artifacts/prefs/dpo_dataset.jsonl",
    max_pairs=200
)

# 3. Train DPO LoRA adapter
from workspace.rlhf.train_dpo_lora import train_dpo_lora
train_dpo_lora(
    dpo_jsonl="artifacts/prefs/dpo_dataset.jsonl",
    base_model_name="Qwen/Qwen2.5-3B-Instruct",
    output_dir="artifacts/models/dpo_lora_v1",
    batch_size=2,
    grad_accum=8,
    lr=1e-5,
    epochs=1,
    beta=0.1
)
```

## ⚙️ Configuration

### RAG Configuration

```python
cfg = RagConfig(
    top_k=20,              # Documents to retrieve
    context_k=5,           # Documents to use for generation
    confidence_threshold=0.5,  # Rewrite threshold
    max_adaptive_steps=1,  # Max rewrite iterations
    alpha_gen=0.6         # Weight for generator confidence
)
```

### Rewrite Guard Configuration

Edit `workspace/models/rewrite_guard.py`:

```python
# Token overlap threshold (0.35 = require 35% similarity)
if overlap < 0.35:
    reasons.append(f"low_token_overlap={overlap:.2f}")

# Forbid new years/numbers
forbid_new_years=True    # Prevent year hallucinations
forbid_new_numbers=True  # Prevent number hallucinations
```

## 🐛 Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues and solutions.

### Quick Fixes

**OpenAI API Error**: 
```bash
export OPENAI_API_KEY="your-key-here"
```

**CUDA Out of Memory**:
- Reduce `batch_size` in DPO training
- Use `device="cpu"` for local models
- Use smaller model (e.g., 1B instead of 3B)

**Adaptive RAG Not Triggering**:
- Lower `confidence_threshold` (e.g., 0.3)
- Check if questions already have dates/numbers
- Review logs for validation failures

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@misc{adaptive-rag-financial-qa,
  title={Adaptive RAG for Financial Question Answering with DPO Alignment},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/your-repo}}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset: `virattt/financial-qa-10K` (HuggingFace)
- Base Model: Qwen2.5-3B-Instruct (Alibaba Cloud)
- Libraries: HuggingFace Transformers, TRL, PEFT, Streamlit

## 📧 Contact

For questions or issues, please open a GitHub issue or contact [your-email@example.com].

---

## 📁 Project Structure

### Source Code
```
workspace/
├── data/              # Dataset downloading and processing
│   ├── download_dataset.py      # Fetch financial-qa-10K from HuggingFace
│   ├── build_corpus.py          # Process into searchable chunks
│   └── build_benchmark.py       # Create train/test splits
├── retrieval/         # Search engines
│   ├── bm25_index.py            # Build BM25 index
│   ├── bm25_searcher.py         # BM25 search implementation
│   ├── faiss_index.py           # Build FAISS index (optional)
│   ├── embeddings.py            # Sentence embedding generation
│   └── corpus_store.py          # Document storage and retrieval
├── models/            # Core RAG system
│   ├── rag.py                   # Baseline & Adaptive RAG
│   ├── local_generator.py       # Local LLM + LoRA support
│   ├── openai_client.py         # OpenAI API wrapper
│   ├── rewrite_guard.py         # Query validation logic
│   ├── prompts.py               # Answer & rewrite templates
│   ├── parser.py                # Output parsing utilities
│   └── uncertainty.py           # Confidence scoring
├── rlhf/              # DPO training pipeline
│   ├── gen_candidates.py        # Generate answer pairs
│   ├── judge_pairs.py           # LLM-as-judge preferences
│   └── train_dpo_lora.py        # DPO LoRA fine-tuning
├── evaluation/        # Metrics and benchmarking
│   ├── eval_runner.py           # Run experiments
│   ├── metrics.py               # Answer quality metrics
│   ├── judge_eval.py            # LLM-as-judge evaluation
│   └── error_analysis.py        # Failure analysis
└── utils/             # Shared utilities
    ├── logging_utils.py         # Logging configuration
    └── seed.py                  # Reproducibility

notebooks/
├── run_all.ipynb              # Full pipeline ⭐
└── run_all_basic.ipynb        # Quick start
```

### Generated Artifacts

All generated data is stored in `artifacts/` (created by notebooks):

```
artifacts/
├── benchmarks/                 # Evaluation datasets
│   ├── train.jsonl            # Training examples (300)
│   ├── test.jsonl             # Test examples (100)
│   ├── test_noisy.jsonl       # Noisy test set (typos, paraphrases)
│   └── *.parquet              # Raw HuggingFace data
├── corpus/                     # Processed documents
│   └── chunks.jsonl           # ~13,000 document chunks
├── indexes_bm25/              # BM25 search index (default)
│   ├── bm25_index.joblib      # TF-IDF index
│   └── chunks_meta.jsonl      # Chunk metadata
├── indexes/                    # FAISS search index (optional)
│   ├── faiss_index.bin        # Dense vector index
│   └── chunks_meta.jsonl      # Chunk metadata
├── prefs/                      # DPO training data
│   ├── candidates.jsonl       # Generated answer pairs
│   └── dpo_dataset.jsonl      # Judged preferences (chosen/rejected)
├── models/                     # Trained models
│   └── dpo_lora_v1/           # DPO fine-tuned adapter
│       ├── adapter_model.safetensors
│       ├── adapter_config.json
│       └── ...                 # Tokenizer, training args
└── eval_runs/                  # Experiment results (Results reported in paper)
    ├── test_baseline_base.json      # Baseline on test set
    ├── test_adaptive_base.json      # Adaptive on test set
    ├── noisy_baseline.json          # Baseline on noisy set
    ├── noisy_adaptive.json          # Adaptive on noisy set
    └── *_dpo.json                   # DPO-aligned variants
    
Total size: ~5-10GB (including model checkpoints)
```

### Entry Points
```
app.py            # Streamlit web UI
run_ui.py         # UI launcher (auto-detects venv)
requirements.txt  # Python dependencies
```

### Documentation
```
README.md                  # Project overview (this file)
TUTORIAL.md                # Step-by-step guide
TROUBLESHOOTING.md         # Common issues and fixes
EXPERIMENTS.md             # Experimental setup and results
SUBMISSION_CHECKLIST.md    # Verification checklist
PROJECT_SPECIFICATIONS.md  # Original requirements
```
