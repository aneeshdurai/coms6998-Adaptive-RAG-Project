# Troubleshooting Guide

Common issues and solutions for the Adaptive RAG system.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Model Loading Issues](#model-loading-issues)
- [OpenAI API Issues](#openai-api-issues)
- [CUDA/GPU Issues](#cudagpu-issues)
- [Adaptive RAG Not Working](#adaptive-rag-not-working)
- [Performance Issues](#performance-issues)
- [DPO Training Issues](#dpo-training-issues)
- [UI Issues](#ui-issues)

---

## Installation Issues

### Problem: `pip install` fails with version conflicts

**Error**:
```
ERROR: Cannot install package X because it has conflicting dependencies
```

**Solution**:
```bash
# Upgrade pip first
pip install --upgrade pip setuptools wheel

# Install with no-cache to force fresh download
pip install --no-cache-dir -r requirements.txt

# If still failing, install packages individually
pip install torch transformers peft trl accelerate
pip install streamlit datasets pandas numpy
pip install faiss-cpu sentence-transformers rank-bm25
```

### Problem: OpenAI package incompatibility

**Error**:
```
Client.__init__() got an unexpected keyword argument 'proxies'
```

**Solution**:
```bash
# Upgrade OpenAI to latest compatible version
pip install --upgrade 'openai>=2.12.0'

# Also upgrade httpx
pip install --upgrade 'httpx>=0.28.0'
```

### Problem: PEFT LoRA config incompatibility

**Error**:
```
LoraConfig.__init__() got an unexpected keyword argument 'alora_invocation_tokens'
```

**Solution**:
```bash
# Upgrade PEFT and Transformers
pip install --upgrade 'peft>=0.18.0'
pip install --upgrade 'transformers>=4.48.0'
```

---

## Model Loading Issues

### Problem: Model download is stuck or fails

**Error**:
```
Downloading shards: 0%|          | 0/2 [00:00<?, ?it/s]
```

**Solutions**:

1. **Set HuggingFace cache location**:
```bash
export HF_HOME="/path/to/large/drive/.cache/huggingface"
```

2. **Check disk space**:
```bash
df -h  # Need ~10GB free
```

3. **Try manual download**:
```python
from transformers import AutoModel
AutoModel.from_pretrained("Qwen/Qwen2.5-3B-Instruct", cache_dir="/path/to/cache")
```

4. **Use mirror** (if in China):
```bash
export HF_ENDPOINT="https://hf-mirror.com"
```

### Problem: Model generates garbage output

**Symptoms**:
- Responses include "system", "user", "assistant" tags
- Multi-line responses when expecting single line
- Random conversational text

**Cause**: Instruct models need proper chat template formatting.

**Solution**: Already fixed in `workspace/models/local_generator.py`. Verify you have the latest version:
```python
# In local_generator.py, complete() method should use:
if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = self.tokenizer.apply_chat_template(...)
```

---

## OpenAI API Issues

### Problem: API key not recognized

**Error**:
```
ValueError: OPENAI_API_KEY not set
```

**Solutions**:

1. **Set environment variable**:
```bash
export OPENAI_API_KEY="sk-your-key-here"
```

2. **Set in Python**:
```python
import os
os.environ["OPENAI_API_KEY"] = "sk-your-key-here"
```

3. **Pass directly to client**:
```python
from workspace.models.openai_client import OpenAIChat
chat = OpenAIChat(model="gpt-4o-mini", api_key="sk-your-key-here")
```

### Problem: Rate limit exceeded

**Error**:
```
RateLimitError: You exceeded your current quota
```

**Solutions**:

1. **Switch to local model**:
```python
from workspace.models.local_generator import LocalChat
chat = LocalChat(model_name="Qwen/Qwen2.5-3B-Instruct")
```

2. **Add retry logic**:
```python
import time
for attempt in range(3):
    try:
        result = chat.complete(prompt)
        break
    except RateLimitError:
        time.sleep(2 ** attempt)
```

3. **Reduce request frequency** in DPO candidate generation.

---

## CUDA/GPU Issues

### Problem: CUDA out of memory

**Error**:
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solutions**:

1. **Reduce batch size**:
```python
# In DPO training
train_dpo_lora(..., batch_size=1, grad_accum=16)  # Instead of batch_size=2, grad_accum=8
```

2. **Use CPU instead**:
```python
chat = LocalChat(model_name="Qwen/Qwen2.5-3B-Instruct", device="cpu")
```

3. **Clear CUDA cache**:
```python
import torch
torch.cuda.empty_cache()
```

4. **Use smaller model**:
```python
chat = LocalChat(model_name="Qwen/Qwen2.5-1.5B-Instruct")
```

### Problem: CUDA not available

**Check**:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

**Solutions**:

1. **Install CUDA-enabled PyTorch**:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

2. **Use CPU** (slower but works):
```python
chat = LocalChat(model_name="Qwen/Qwen2.5-3B-Instruct", device="cpu")
```

---

## Adaptive RAG Not Working

### Problem: Adaptive always uses original query

**Symptom**:
```
Adaptive step 1: 'query' -> 'query'  # Same query
Rewrite validation failed
```

**Cause**: Rewrite guard is blocking rewrites.

**Solutions**:

1. **Check validation settings**:
```python
# In workspace/models/rewrite_guard.py
if overlap < 0.35:  # Try lowering to 0.20-0.25
```

2. **Relax validation**:
```python
# In workspace/models/rag.py, QueryRewriter.rewrite()
check = validate_rewrite(question, rewritten, 
                         forbid_new_years=False,  # Allow years
                         forbid_new_numbers=False) # Allow numbers
```

3. **Use queries with existing dates**:
```python
# These work better with strict validation:
"What was Apple's revenue in 2023?"  # Year already present
"How much debt in fiscal year 2022?" # Year already present
```

### Problem: Confidence always high (never triggers)

**Symptom**:
```
Confidence: 0.95  # Always above threshold
Adaptive triggered: False
```

**Solutions**:

1. **Lower confidence threshold**:
```python
cfg = RagConfig(confidence_threshold=0.3)  # From 0.5
```

2. **Adjust alpha (generator weight)**:
```python
cfg = RagConfig(alpha_gen=0.4)  # Give more weight to retrieval
```

3. **Check retrieval quality**: Poor retrieval → low confidence → triggers adaptive

---

## Performance Issues

### Problem: Inference is very slow

**Symptoms**:
- 30+ seconds per query
- UI freezes during generation

**Solutions**:

1. **Use GPU**:
```python
chat = LocalChat(model_name="Qwen/Qwen2.5-3B-Instruct", device="cuda")
```

2. **Reduce max_tokens**:
```python
# In prompts or generation
chat.complete(prompt, max_tokens=64)  # Instead of 128
```

3. **Use smaller model**:
```python
chat = LocalChat(model_name="Qwen/Qwen2.5-1.5B-Instruct")
```

4. **Disable adaptive mode**:
```python
result = rag.answer_baseline(question)  # No rewriting overhead
```

### Problem: Retrieval is slow

**Solutions**:

1. **Reduce top_k**:
```python
cfg = RagConfig(top_k=10)  # Instead of 20
```

2. **Use BM25 instead of FAISS** (already default, faster for small corpora)

3. **Build index on SSD** (not HDD)

---

## DPO Training Issues

### Problem: Training crashes with OOM

**Solutions**:

1. **Reduce batch size**:
```python
train_dpo_lora(..., batch_size=1, grad_accum=16)
```

2. **Reduce sequence length**:
```python
train_dpo_lora(..., max_prompt_length=512, max_length=640)
```

3. **Use gradient checkpointing** (TODO: add to code)

### Problem: No pairs kept after judging

**Symptom**:
```
Judged 300 pairs, kept 0 -> artifacts/prefs/dpo_dataset.jsonl
```

**Causes**:
- Judge returns "TIE" for all pairs
- Temperature difference too small (0.0 vs 0.7)

**Solutions**:

1. **Use better judge** (OpenAI instead of local):
```python
from workspace.models.openai_client import OpenAIChat
judge = OpenAIChat(model="gpt-4o-mini")
judge_pairs(..., judge_client=judge, ...)
```

2. **Increase temperature spread**:
```python
# In gen_candidates.py
a0 = chat_client.complete(prompt, temperature=0.0)
a1 = chat_client.complete(prompt, temperature=1.0)  # From 0.7
```

3. **Generate more candidates**:
```python
gen_candidates(..., n=500)  # Instead of 300
```

---

## UI Issues

### Problem: Streamlit UI won't start

**Error**:
```
ModuleNotFoundError: No module named 'streamlit'
```

**Solution**:
```bash
# Ensure venv is activated
source venv/bin/activate

# Reinstall streamlit
pip install --upgrade streamlit

# Run UI
python run_ui.py
```

### Problem: UI shows "Missing artifacts"

**Solution**:
Use the built-in artifact builder:

1. In UI sidebar, expand "🔧 Build Artifacts Automatically"
2. Click "🚀 Build Artifacts"
3. Wait for completion (~5-10 minutes)
4. Refresh page

Or build manually:
```bash
# Run notebook to build all artifacts
jupyter notebook notebooks/run_all.ipynb
```

### Problem: UI is unresponsive

**Solutions**:

1. **Check terminal for errors**
2. **Restart Streamlit**:
```bash
# Press Ctrl+C
python run_ui.py
```
3. **Clear browser cache**
4. **Try different browser**

---

## General Debugging Tips

### Enable Detailed Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check System Resources

```bash
# CPU and memory
htop  # or top

# GPU memory
nvidia-smi

# Disk space
df -h
```

### Verify Artifacts Exist

```bash
ls -lh artifacts/corpus/
ls -lh artifacts/indexes_bm25/
ls -lh artifacts/benchmarks/
```

### Test Individual Components

```python
# Test retrieval
from workspace.retrieval.bm25_searcher import BM25Searcher
searcher = BM25Searcher("artifacts/indexes_bm25/bm25_index.joblib", 
                        "artifacts/indexes_bm25/chunks_meta.jsonl")
hits = searcher.search("test query", k=5)
print(f"Retrieved {len(hits)} documents")

# Test generation
from workspace.models.local_generator import LocalChat
chat = LocalChat("Qwen/Qwen2.5-3B-Instruct")
response = chat.complete("Answer: What is 2+2?", max_tokens=10)
print(f"Response: {response}")
```

---

## Still Having Issues?

1. **Check logs**: Look in terminal output for error messages
2. **Review documentation**: See [README.md](README.md) and [TUTORIAL.md](TUTORIAL.md)
3. **Search existing issues**: Check GitHub issues
4. **Ask for help**: Open a new GitHub issue with:
   - Error message (full traceback)
   - System info (OS, Python version, GPU)
   - Steps to reproduce
   - What you've already tried

---

## Quick Fixes Checklist

- [ ] Virtual environment activated
- [ ] All dependencies installed (`pip list | grep -E "transformers|peft|openai"`)
- [ ] Artifacts exist (`ls artifacts/corpus/ artifacts/indexes_bm25/`)
- [ ] GPU available (if using CUDA) (`python -c "import torch; print(torch.cuda.is_available())"`)
- [ ] Sufficient disk space (`df -h`)
- [ ] OpenAI API key set (if using OpenAI) (`echo $OPENAI_API_KEY`)
- [ ] Latest code (`git pull` if using git)

**Most Common Fix**: Restart Python/Streamlit and try again! 🔄
