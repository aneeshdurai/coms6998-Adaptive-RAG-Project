# Adaptive RAG UI

A clean, interactive web interface for the Confidence-Aware Adaptive Retrieval-Augmented Generation system.

## Features

- **Interactive Query Interface**: Enter financial questions and get answers with confidence scores
- **Dual Mode Support**: Compare Baseline RAG vs Adaptive RAG results
- **Confidence Visualization**: Color-coded confidence indicators (high/medium/low)
- **Query Rewriting Details**: See when and how queries are rewritten for better retrieval
- **Retrieved Documents**: View the top retrieved documents with scores
- **Configurable Settings**: Adjust confidence thresholds, adaptive steps, and model parameters

## Prerequisites

1. **Build the artifacts first** by running the notebook:
   ```bash
   # Run notebooks/run_all_UPDATED.ipynb to build:
   # - artifacts/corpus/chunks.jsonl
   # - artifacts/indexes/faiss_index.bin
   # - artifacts/indexes/chunks_meta.jsonl
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set OpenAI API key** (if using OpenAI):
   ```bash
   export OPENAI_API_KEY="your-key-here"
   # Or set it in the .env file
   ```

## Running the UI

### Option 1: Direct Streamlit command
```bash
streamlit run app.py
```

### Option 2: Using the run script
```bash
python run_ui.py
```

The UI will open in your default browser at `http://localhost:8501`

## Usage

1. **Configure Settings** (Sidebar):
   - Choose between OpenAI or local models
   - Adjust confidence threshold (default: 0.5)
   - Set max adaptive steps (default: 1)
   - Configure retrieval parameters (top_k, context_k)

2. **Enter Query**:
   - Type your financial question in the text area
   - Examples:
     - "What criteria are used to classify loans as nonperforming?"
     - "How much cash from foreign subsidiaries is available for repatriation?"
     - "What are Etsy's main operating marketplaces?"

3. **Run Query**:
   - Click **Baseline RAG** for standard retrieval
   - Click **Adaptive RAG** for confidence-aware adaptive retrieval

4. **View Results**:
   - **Answer**: The generated answer
   - **Confidence Metrics**: Combined, generator, and retrieval confidence scores
   - **Adaptive Info**: Query rewriting details (if triggered)
   - **Retrieved Documents**: Top retrieved documents with scores
   - **Debug JSON**: Full response for debugging

## Understanding the Output

### Confidence Levels
- **High (Green)**: ≥ 70% - High confidence answer
- **Medium (Yellow)**: 40-70% - Moderate confidence
- **Low (Red)**: < 40% - Low confidence, may need review

### Adaptive Rewriting
- **Triggered**: Query was rewritten because initial confidence was below threshold
- **Not Triggered**: Initial confidence met threshold, no rewriting needed
- **Rewrite History**: Shows the sequence of query rewrites and their confidence scores
- **Comparison**: Side-by-side comparison of first pass vs rewritten results

### Retrieved Documents
- Shows top 5 retrieved documents
- Each document includes:
  - Retrieval score
  - Document ID and chunk ID
  - Full text content
  - Metadata (if available)

## Configuration Options

### Model Selection
- **OpenAI**: Use GPT models via API (requires API key)
- **Local**: Use HuggingFace models (e.g., Qwen2.5-3B-Instruct)

### RAG Settings
- **Confidence Threshold**: Below this value, adaptive rewriting is triggered
- **Max Adaptive Steps**: Maximum number of rewrite attempts
- **Generator Confidence Weight (α)**: Weight for generator confidence in combined score
- **Top K Retrieval**: Number of documents to retrieve
- **Context K**: Number of documents to use as context for generation

## Troubleshooting

### "Missing required artifacts" error
- Make sure you've run the notebook to build the indexes
- Check that `artifacts/` directory exists with:
  - `corpus/chunks.jsonl`
  - `indexes/faiss_index.bin`
  - `indexes/chunks_meta.jsonl`

### CUDA/GPU issues
- Set device to "cpu" in the sidebar if you don't have GPU
- Local models will run slower on CPU

### OpenAI API errors
- Verify your API key is set correctly
- Check your API quota/limits

## Architecture

The UI uses Streamlit for the web interface and integrates with the existing RAG system:
- Loads FAISS index and corpus store
- Initializes generator and rewriter models
- Processes queries through the adaptive RAG pipeline
- Displays results with rich formatting and visualizations

