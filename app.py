import streamlit as st
import sys
import os
from pathlib import Path
import time
import json

# Set HuggingFace cache to local directory to avoid permission issues
cache_dir = Path(__file__).parent / ".hf_cache"
cache_dir.mkdir(exist_ok=True)
os.environ["HF_HOME"] = str(cache_dir)
os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "transformers")
os.environ["HF_DATASETS_CACHE"] = str(cache_dir / "datasets")

# Add workspace to path
p = Path(__file__).parent.resolve()
while p != p.parent and not (p / "workspace").exists():
    p = p.parent

if str(p) not in sys.path:
    sys.path.insert(0, str(p))

from workspace.models.rag import RAGGenerator, QueryRewriter, AdaptiveRAG, RagConfig
from workspace.retrieval.bm25_searcher import BM25Searcher
from workspace.retrieval.bm25_index import build_bm25_index
from workspace.retrieval.corpus_store import CorpusStore
from workspace.models.local_generator import LocalChat
from workspace.models.openai_client import OpenAIChat
from workspace.data.download_dataset import download_financial_qa_10k
from workspace.data.build_corpus import build_corpus_from_parquet

# Page config
st.set_page_config(
    page_title="Adaptive RAG - Financial QA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .confidence-high {
        background-color: #d4edda;
        padding: 0.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .confidence-medium {
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .confidence-low {
        background-color: #f8d7da;
        padding: 0.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
    .metric-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
    }
    .rewrite-badge {
        background-color: #007bff;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        display: inline-block;
        margin-left: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_rag_system(
    use_openai: bool,
    openai_model: str,
    local_model: str,
    embed_model: str,
    artifacts_dir: str,
    device: str = "cuda",
    lora_path: str | None = None
):
    """Load and initialize the RAG system components"""
    # Handle relative paths
    artifacts = Path(artifacts_dir)
    if not artifacts.is_absolute():
        # Try to find repo root
        p = Path(__file__).parent.resolve()
        while p != p.parent and not (p / "workspace").exists():
            p = p.parent
        artifacts = p / artifacts_dir
    
    # Load retrieval components (BM25)
    chunks_path = artifacts / "corpus" / "chunks.jsonl"
    bm25_index_path = artifacts / "indexes_bm25" / "bm25_index.joblib"
    bm25_meta_path = artifacts / "indexes_bm25" / "chunks_meta.jsonl"
    
    if not all([chunks_path.exists(), bm25_index_path.exists(), bm25_meta_path.exists()]):
        return None, "Missing required artifacts. Please build BM25 index first (use the Build Artifacts button)."
    
    try:
        searcher = BM25Searcher(
            index_path=str(bm25_index_path),
            meta_path=str(bm25_meta_path),
        )
        store = CorpusStore(str(chunks_path))
        
        # Initialize generator and rewriter
        if use_openai:
            chat = OpenAIChat(model=openai_model)
        else:
            chat = LocalChat(model_name=local_model, device=device, lora_path=lora_path)
        
        generator = RAGGenerator(chat)
        rewriter = QueryRewriter(chat)
        
        # Default config (can be overridden in sidebar)
        cfg = RagConfig(
            top_k=20,
            context_k=5,
            confidence_threshold=0.5,
            max_adaptive_steps=1,
            alpha_gen=0.6,
        )
        
        rag = AdaptiveRAG(
            searcher=searcher,
            store=store,
            generator=generator,
            rewriter=rewriter,
            cfg=cfg,
        )
        
        return rag, None
    except Exception as e:
        return None, f"Error loading RAG system: {str(e)}"

def get_confidence_class(conf: float) -> str:
    """Get CSS class based on confidence level"""
    if conf >= 0.7:
        return "confidence-high"
    elif conf >= 0.4:
        return "confidence-medium"
    else:
        return "confidence-low"

def format_confidence(conf: float) -> str:
    """Format confidence as percentage"""
    return f"{conf * 100:.1f}%"

def main():
    # Header
    st.markdown('<p class="main-header">📊 Adaptive RAG for Financial QA</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Confidence-Aware Adaptive Retrieval-Augmented Generation</p>', unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # OpenAI API Key input
        st.subheader("🔑 OpenAI API Key")
        api_key_input = st.text_input(
            "API Key",
            type="password",
            help="Enter your OpenAI API key (or set OPENAI_API_KEY environment variable)",
            placeholder="sk-..."
        )
        
        # Track API key changes to clear cache
        if "last_api_key" not in st.session_state:
            st.session_state.last_api_key = None
        
        if api_key_input:
            if st.session_state.last_api_key != api_key_input:
                # API key changed, clear cache
                load_rag_system.clear()
                st.session_state.last_api_key = api_key_input
            os.environ["OPENAI_API_KEY"] = api_key_input
            st.success("✅ API key set for this session")
        elif os.getenv("OPENAI_API_KEY"):
            if st.session_state.last_api_key != "env_var":
                st.session_state.last_api_key = "env_var"
            st.info("ℹ️ Using API key from environment variable")
        else:
            if st.session_state.last_api_key is not None:
                st.session_state.last_api_key = None
            st.warning("⚠️ No API key set. Set it above or as OPENAI_API_KEY environment variable.")
        
        st.divider()
        
        # Artifacts path (define early since DPO section needs it)
        artifacts_dir = st.text_input(
            "Artifacts Directory",
            value="artifacts",
            help="Path to artifacts directory containing indexes and corpus (relative to repo root)"
        )
        
        st.divider()
        
        # Model selection
        use_openai = st.checkbox("Use OpenAI", value=False, help="Use OpenAI API instead of local model")
        
        # Warn if OpenAI is selected but no key
        if use_openai and not os.getenv("OPENAI_API_KEY"):
            st.error("❌ Please set your OpenAI API key above before using OpenAI models.")
        
        if use_openai:
            openai_model = st.selectbox(
                "OpenAI Model",
                ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
                index=0
            )
            local_model = "Qwen/Qwen2.5-3B-Instruct"
        else:
            local_model = st.text_input(
                "Local Model",
                value="Qwen/Qwen2.5-3B-Instruct",
                help="HuggingFace model ID"
            )
            openai_model = "gpt-4o-mini"
            
            # DPO LoRA option
            st.divider()
            st.subheader("🎯 DPO LoRA (Optional)")
            use_dpo = st.checkbox(
                "Use DPO-aligned LoRA adapter",
                value=st.session_state.get("use_dpo", False),
                help="Load a DPO-trained LoRA adapter for improved answer quality"
            )
            st.session_state.use_dpo = use_dpo
            
            dpo_lora_path = None
            if use_dpo:
                # Check for existing DPO models
                artifacts_path = Path(artifacts_dir) if Path(artifacts_dir).is_absolute() else Path(__file__).parent.resolve() / artifacts_dir
                models_dir = artifacts_path / "models"
                
                if models_dir.exists():
                    dpo_models = [d.name for d in models_dir.iterdir() if d.is_dir() and "dpo" in d.name.lower()]
                    if dpo_models:
                        selected_dpo = st.selectbox(
                            "DPO LoRA Model",
                            dpo_models,
                            index=0,
                            help="Select a trained DPO LoRA adapter"
                        )
                        dpo_lora_path = str(models_dir / selected_dpo)
                    else:
                        dpo_lora_path = st.text_input(
                            "DPO LoRA Path",
                            value=st.session_state.get("dpo_lora_path", ""),
                            placeholder="artifacts/models/dpo_lora_v1",
                            help="Path to DPO LoRA adapter directory"
                        )
                else:
                    dpo_lora_path = st.text_input(
                        "DPO LoRA Path",
                        value=st.session_state.get("dpo_lora_path", ""),
                        placeholder="artifacts/models/dpo_lora_v1",
                        help="Path to DPO LoRA adapter directory"
                    )
                
                # Store in session state
                if dpo_lora_path:
                    st.session_state.dpo_lora_path = dpo_lora_path
            else:
                # Clear if not using DPO
                if "dpo_lora_path" in st.session_state:
                    del st.session_state.dpo_lora_path
                st.session_state.dpo_lora_path = None
        
        # Note: BM25 doesn't need embedding model, but keeping for artifact builder compatibility
        embed_model = st.text_input(
            "Embedding Model (not used with BM25)",
            value="BAAI/bge-small-en-v1.5",
            help="Only used if rebuilding artifacts (BM25 doesn't need embeddings)"
        )
        st.info("ℹ️ Using BM25 (keyword-based) retrieval - no embeddings needed")
        
        # RAG configuration
        st.divider()
        st.subheader("RAG Settings")
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Threshold below which query rewriting is triggered"
        )
        
        max_adaptive_steps = st.slider(
            "Max Adaptive Steps",
            min_value=1,
            max_value=3,
            value=1,
            help="Maximum number of query rewrite attempts"
        )
        
        alpha_gen = st.slider(
            "Generator Confidence Weight (α)",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.1,
            help="Weight for generator confidence in combined score"
        )
        
        top_k = st.number_input(
            "Top K Retrieval",
            min_value=5,
            max_value=50,
            value=20,
            step=5,
            help="Number of documents to retrieve"
        )
        
        context_k = st.number_input(
            "Context K",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of retrieved documents to use as context"
        )
        
        # Check if artifacts exist
        st.divider()
        artifacts_path = Path(artifacts_dir)
        if not artifacts_path.is_absolute():
            # Find repo root
            repo_root = Path(__file__).parent.resolve()
            while repo_root != repo_root.parent and not (repo_root / "workspace").exists():
                repo_root = repo_root.parent
            artifacts_path = repo_root / artifacts_dir
        
        chunks_exist = (artifacts_path / "corpus" / "chunks.jsonl").exists()
        bm25_index_exist = (artifacts_path / "indexes_bm25" / "bm25_index.joblib").exists()
        bm25_meta_exist = (artifacts_path / "indexes_bm25" / "chunks_meta.jsonl").exists()
        
        if chunks_exist and bm25_index_exist and bm25_meta_exist:
            st.success("✅ Artifacts found (BM25)")
        else:
            st.warning("⚠️ Some artifacts missing.")
            
            # Artifact builder section
            with st.expander("🔧 Build Artifacts Automatically", expanded=not (chunks_exist and bm25_index_exist)):
                st.write("The system can automatically download the dataset and build the required artifacts.")
                
                hf_dataset_id = st.text_input(
                    "HuggingFace Dataset ID",
                    value="virattt/financial-qa-10K",
                    help="Dataset to download and use"
                )
                
                build_artifacts = st.button("🚀 Build Artifacts", type="primary", use_container_width=True)
                
                if build_artifacts:
                    with st.spinner("Building artifacts (this may take a few minutes)..."):
                        try:
                            # Step 1: Download dataset
                            data_dir = artifacts_path / "benchmarks"
                            data_dir.mkdir(parents=True, exist_ok=True)
                            
                            st.write("📥 Step 1/3: Downloading dataset...")
                            download_financial_qa_10k(
                                hf_id=hf_dataset_id,
                                out_dir=str(data_dir),
                                splits=("train", "test"),
                            )
                            
                            # Check if we have train/test parquet files
                            import pandas as pd
                            train_parquet = data_dir / f"{hf_dataset_id.replace('/','__')}_train.parquet"
                            test_parquet = data_dir / f"{hf_dataset_id.replace('/','__')}_test.parquet"
                            
                            # If only train exists, create test split
                            if train_parquet.exists() and not test_parquet.exists():
                                df = pd.read_parquet(train_parquet)
                                df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
                                cut = int(0.8 * len(df))
                                df_train = df.iloc[:cut].copy()
                                df_test = df.iloc[cut:].copy()
                                df_train.to_parquet(train_parquet, index=False)
                                df_test.to_parquet(test_parquet, index=False)
                            
                            parquet_paths = [str(p) for p in [train_parquet, test_parquet] if p.exists()]
                            
                            if not parquet_paths:
                                st.error("Failed to download dataset. Please check the dataset ID.")
                                st.stop()
                            
                            # Step 2: Build corpus
                            st.write("📚 Step 2/3: Building corpus from dataset...")
                            corpus_dir = artifacts_path / "corpus"
                            corpus_dir.mkdir(parents=True, exist_ok=True)
                            chunks_path = corpus_dir / "chunks.jsonl"
                            
                            build_corpus_from_parquet(
                                parquet_paths=parquet_paths,
                                out_chunks_path=str(chunks_path),
                                chunk_size_chars=1800,
                                chunk_overlap_chars=200,
                                min_chars=200,
                            )
                            
                            # Step 3: Build BM25 index
                            st.write("🔍 Step 3/3: Building BM25 index...")
                            bm25_index_dir = artifacts_path / "indexes_bm25"
                            bm25_index_dir.mkdir(parents=True, exist_ok=True)
                            
                            result = build_bm25_index(
                                chunks_path=str(chunks_path),
                                meta_out_path=str(bm25_index_dir / "chunks_meta.jsonl"),
                                index_out_path=str(bm25_index_dir / "bm25_index.joblib"),
                            )
                            st.write(f"✅ Built BM25 index with {result['n_chunks']} chunks")
                            
                            st.success("✅ Artifacts built successfully! Please refresh the page.")
                            st.balloons()
                            
                            # Clear cache to reload
                            if "rag" in st.session_state:
                                del st.session_state.rag
                            
                        except Exception as e:
                            st.error(f"Error building artifacts: {str(e)}")
                            with st.expander("Error Details"):
                                import traceback
                                st.code(traceback.format_exc())
        
        device = st.selectbox(
            "Device",
            ["cuda", "cpu"],
            index=0 if st.session_state.get("has_cuda", True) else 1,
            help="Device for model inference"
        )
    
    # Check if OpenAI is selected but no key
    if use_openai and not os.getenv("OPENAI_API_KEY"):
        st.error("❌ Please set your OpenAI API key in the sidebar before loading the system.")
        st.stop()
    
    # Get DPO LoRA path from sidebar (stored in session state)
    dpo_lora_path = st.session_state.get("dpo_lora_path") if not use_openai else None
    
    # Track config changes (including API key state and LoRA path)
    api_key_set = "yes" if os.getenv("OPENAI_API_KEY") else "no"
    lora_key = dpo_lora_path or "none"
    config_key = f"{use_openai}_{openai_model}_{local_model}_{artifacts_dir}_{device}_{api_key_set}_{lora_key}"
    if "last_config" not in st.session_state or st.session_state.last_config != config_key:
        st.session_state.last_config = config_key
        if "rag" in st.session_state:
            del st.session_state.rag
        # Clear cache when config changes
        load_rag_system.clear()
    
    # Load RAG system
    if "rag" not in st.session_state:
        loading_msg = "Loading RAG system (BM25)..." if use_openai else f"Loading RAG system (BM25 + {local_model})... This may take a few minutes on first run (downloading/loading model)"
        with st.spinner(loading_msg):
            try:
                rag, error = load_rag_system(
                    use_openai=use_openai,
                    openai_model=openai_model,
                    local_model=local_model,
                    embed_model=embed_model,  # Not used for BM25 but kept for compatibility
                    artifacts_dir=artifacts_dir,
                    device=device,
                    lora_path=dpo_lora_path if not use_openai else None,
                )
                
                if error:
                    st.error(error)
                    # Clear cache on error so user can retry after fixing API key
                    if "rag" in st.session_state:
                        del st.session_state.rag
                    load_rag_system.clear()
                    st.stop()
                
                st.session_state.rag = rag
                model_info = f"Using BM25 retrieval + {local_model}" if not use_openai else "Using BM25 retrieval + OpenAI"
                st.success(f"✅ RAG system loaded successfully! ({model_info})")
            except Exception as e:
                st.error(f"Error loading RAG system: {str(e)}")
                with st.expander("Error Details"):
                    import traceback
                    st.code(traceback.format_exc())
                if "rag" in st.session_state:
                    del st.session_state.rag
                load_rag_system.clear()
                st.stop()
    
    # Update config dynamically
    rag = st.session_state.rag
    rag.cfg.confidence_threshold = confidence_threshold
    rag.cfg.max_adaptive_steps = max_adaptive_steps
    rag.cfg.alpha_gen = alpha_gen
    rag.cfg.top_k = top_k
    rag.cfg.context_k = context_k
    
    # Main query interface
    st.divider()
    
    # Query input
    query = st.text_area(
        "Enter your financial question:",
        height=100,
        placeholder="e.g., What criteria are used to classify loans as nonperforming?",
        help="Ask a question about financial information from SEC 10-K filings"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        run_baseline = st.button("🔍 Baseline RAG", use_container_width=True)
    with col2:
        run_adaptive = st.button("🚀 Adaptive RAG", use_container_width=True, type="primary")
    
        # Process query
    if run_baseline or run_adaptive:
        if not query.strip():
            st.warning("Please enter a question.")
            st.stop()
        
        if "rag" not in st.session_state:
            st.error("RAG system not loaded. Please check your configuration.")
            st.stop()
        
        mode = "adaptive" if run_adaptive else "baseline"
        
        # Show progress with more detail
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            start_time = time.time()
            status_text.text("🔍 Retrieving documents...")
            progress_bar.progress(10)
            
            if mode == "adaptive":
                status_text.text("🤖 Generating answer with adaptive RAG...")
                progress_bar.progress(30)
                result = st.session_state.rag.answer_adaptive(query)
            else:
                status_text.text("🤖 Generating answer with baseline RAG...")
                progress_bar.progress(30)
                result = st.session_state.rag.answer_baseline(query)
            
            progress_bar.progress(90)
            status_text.text("✅ Processing complete!")
            latency = time.time() - start_time
            
            progress_bar.progress(100)
            status_text.empty()
            progress_bar.empty()
            
            # Show warning if it took too long (likely CPU)
            if latency > 30 and not use_openai:
                st.warning(f"⚠️ Generation took {latency:.1f}s. If you have a GPU, make sure CUDA is available and device is set to 'cuda' in the sidebar.")
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Error processing query: {str(e)}")
            with st.expander("Error Details"):
                import traceback
                st.code(traceback.format_exc())
            st.stop()
        
        # Display results
        st.divider()
        
        # Main results section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📝 Answer")
            st.markdown(f"**{result['answer']}**")
            
            if result.get("raw"):
                with st.expander("View raw model output"):
                    st.code(result["raw"], language=None)
        
        with col2:
            st.subheader("📊 Confidence Metrics")
            
            conf = result.get("confidence", 0.0)
            gen_conf = result.get("gen_conf", 0.0)
            retr_conf = result.get("retr_conf", 0.0)
            
            conf_class = get_confidence_class(conf)
            st.markdown(f'<div class="{conf_class}">', unsafe_allow_html=True)
            st.metric("Combined Confidence", format_confidence(conf))
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.metric("Generator Confidence", format_confidence(gen_conf))
            st.metric("Retrieval Confidence", format_confidence(retr_conf))
            st.metric("Latency", f"{latency:.2f}s")
        
        # Adaptive-specific information
        # Adaptive-specific information
        if mode == "adaptive":
            st.divider()
            st.subheader("🔄 Adaptive Query Rewriting")
            
            adaptive_triggered = result.get("adaptive_triggered", False)
            
            if adaptive_triggered:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.success("✅ Query Rewriting Triggered")
                    rewritten = result.get("rewritten_query")
                    if rewritten:
                        st.info(f"**Rewritten Query:** {rewritten}")
                    
                    # Show rewrite history if available
                    rewrite_history = result.get("rewrite_history", [])
                    if rewrite_history:
                        with st.expander("📜 View Rewrite History"):
                            for step in rewrite_history:
                                st.write(f"**Step {step['step']}:** `{step['query']}`")
                                st.write(f"Confidence: {format_confidence(step['confidence'])}")
                                st.divider()
                    
                    # Compare with first pass
                    first_pass = result.get("first_pass")
                    if first_pass:
                        with st.expander("📊 Compare with First Pass"):
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.write("**First Pass**")
                                st.write(f"Answer: {first_pass.get('answer', 'N/A')}")
                                st.write(f"Confidence: {format_confidence(first_pass.get('confidence', 0.0))}")
                                st.write(f"Gen Conf: {format_confidence(first_pass.get('gen_conf', 0.0))}")
                                st.write(f"Retr Conf: {format_confidence(first_pass.get('retr_conf', 0.0))}")
                            with col_b:
                                st.write("**After Rewrite**")
                                st.write(f"Answer: {result.get('answer', 'N/A')}")
                                st.write(f"Confidence: {format_confidence(conf)}")
                                st.write(f"Gen Conf: {format_confidence(gen_conf)}")
                                st.write(f"Retr Conf: {format_confidence(retr_conf)}")
                
                with col2:
                    first_pass = result.get("first_pass")
                    if first_pass:
                        conf_improvement = conf - first_pass.get("confidence", 0.0)
                        if conf_improvement > 0:
                            st.success(f"📈 Confidence improved by {format_confidence(conf_improvement)}")
                        elif conf_improvement < 0:
                            st.warning(f"📉 Confidence decreased by {format_confidence(abs(conf_improvement))}")
                        else:
                            st.info("➡️ Confidence unchanged")
            else:
                st.info("ℹ️ Confidence threshold met - no rewriting needed")
        
        # Retrieved documents
        st.divider()
        st.subheader("📚 Retrieved Documents")
        
        hits = result.get("hits", [])
        if hits:
            st.write(f"Retrieved {len(hits)} documents (showing top {min(5, len(hits))}):")
            
            for i, hit in enumerate(hits[:5], 1):
                with st.expander(f"Document {i} (Score: {hit.get('score', 0):.4f})"):
                    doc_id = hit.get("doc_id", "N/A")
                    chunk_id = hit.get("chunk_id", "N/A")
                    text = st.session_state.rag.store.get_text(doc_id, chunk_id)
                    
                    st.write(f"**Doc ID:** {doc_id}")
                    st.write(f"**Chunk ID:** {chunk_id}")
                    st.write("**Content:**")
                    st.write(text)
                    
                    meta = hit.get("meta", {})
                    if meta:
                        st.write("**Metadata:**")
                        st.json(meta)
        else:
            st.warning("No documents retrieved.")
        
        # JSON output (for debugging)
        with st.expander("🔧 Debug: Full JSON Response"):
            st.json(result)

if __name__ == "__main__":
    main()

