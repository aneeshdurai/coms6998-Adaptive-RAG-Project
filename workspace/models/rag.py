from dataclasses import dataclass
from typing import List, Dict, Any, Optional, TYPE_CHECKING, Protocol
from workspace.utils.logging_utils import get_logger
from workspace.retrieval.corpus_store import CorpusStore
from workspace.models.prompts import ANSWER_PROMPT, REWRITE_PROMPT
from workspace.models.parser import parse_answer_and_conf
from workspace.models.uncertainty import retrieval_conf, combine_conf
import math

# Protocol for searcher to avoid requiring FAISS import
class SearcherProtocol(Protocol):
    def search(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        """Search and return list of hits with score, doc_id, chunk_id"""
        ...

# Type hints for searchers (only imported during type checking)
if TYPE_CHECKING:
    from workspace.retrieval.faiss_index import FaissSearcher
    from workspace.retrieval.bm25_searcher import BM25Searcher

logger = get_logger("models.rag")

@dataclass
class RagConfig:
    top_k: int = 20
    context_k: int = 5
    confidence_threshold: float = 0.5
    max_adaptive_steps: int = 1
    alpha_gen: float = 0.6  # weight for generator confidence in combined score

class RAGGenerator:
    def __init__(self, chat_client):
        self.chat = chat_client

    def answer(self, question: str, context: str, temperature: float = 0.0) -> Dict[str, Any]:
        prompt = ANSWER_PROMPT.format(question=question, context=context)
        out = self.chat.complete(prompt, temperature=temperature, max_tokens=128)
        ans, conf = parse_answer_and_conf(out)
        return {"raw": out, "answer": ans, "gen_conf": conf}

# class QueryRewriter:
#     def __init__(self, chat_client):
#         self.chat = chat_client

#     def rewrite(self, question: str, snippets: str) -> str:
#         prompt = REWRITE_PROMPT.format(question=question, snippets=snippets)
#         out = self.chat.complete(prompt, temperature=0.0, max_tokens=64)
#         # Just take first line to avoid any accidental extra text
#         return (out or "").strip().splitlines()[0].strip()

from workspace.models.rewrite_guard import validate_rewrite

class QueryRewriter:
    def __init__(self, chat_client):
        self.chat = chat_client

    def rewrite(self, question: str, snippets: str) -> str:
        prompt = REWRITE_PROMPT.format(question=question, snippets=snippets)
        out = self.chat.complete(prompt, temperature=0.0, max_tokens=64)
        rewritten = (out or "").strip().splitlines()[0].strip()

        # Guard: do not allow the rewrite to introduce new constraints (e.g., years/numbers)
        check = validate_rewrite(question, rewritten, forbid_new_years=True, forbid_new_numbers=True)
        if not check.ok:
            # Fall back safely to original question
            # (Optional: add logging here if you have a logger)
            return question

        return rewritten

class AdaptiveRAG:
    def __init__(
        self,
        searcher: SearcherProtocol,
        store: CorpusStore,
        generator: RAGGenerator,
        rewriter: QueryRewriter,
        cfg: RagConfig,
    ):
        self.searcher = searcher
        self.store = store
        self.generator = generator
        self.rewriter = rewriter
        self.cfg = cfg

    def _build_context(self, hits: List[Dict[str, Any]]) -> str:
        parts = []
        for h in hits[: self.cfg.context_k]:
            text = self.store.get_text(h["doc_id"], h["chunk_id"])
            parts.append(f"[{h['doc_id']}:{h['chunk_id']} score={h['score']:.3f}]\n{text}")
        return "\n\n".join(parts)

    def _snippets(self, hits: List[Dict[str, Any]]) -> str:
        # short snippets for rewrite prompt
        parts = []
        for h in hits[: min(3, len(hits))]:
            text = self.store.get_text(h["doc_id"], h["chunk_id"])
            parts.append(text[:300])
        return "\n---\n".join(parts)

    def answer_baseline(self, question: str) -> Dict[str, Any]:
        hits = self.searcher.search(question, k=self.cfg.top_k)
        context = self._build_context(hits)
        gen = self.generator.answer(question, context, temperature=0.0)
        # top1 = hits[0]["score"] if hits else None
        # c_retr = retrieval_conf(top1)
        def bm25_to_conf(score: float) -> float:
            # score scale varies by corpus; this squashes reasonably
            return 1.0 / (1.0 + math.exp(-0.15 * (score - 8.0)))
            
        top1 = hits[0]["score"] if hits else 0.0
        c_retr = bm25_to_conf(top1)
        c = combine_conf(gen["gen_conf"], c_retr, alpha=self.cfg.alpha_gen)
        return {
            "question": question,
            "answer": gen["answer"],
            "gen_conf": gen["gen_conf"],
            "retr_conf": c_retr,
            "confidence": c,
            "raw": gen["raw"],
            "hits": hits,
            "adaptive_triggered": False,
            "rewritten_query": None,
        }

    # def answer_adaptive(self, question: str) -> Dict[str, Any]:
    #     res = self.answer_baseline(question)
    #     if res["confidence"] >= self.cfg.confidence_threshold:
    #         return res

    #     if self.cfg.max_adaptive_steps <= 0:
    #         return res

    #     snippets = self._snippets(res["hits"])
    #     rewritten = self.rewriter.rewrite(question, snippets)
    #     logger.info(f"Adaptive: rewriting '{question}' -> '{rewritten}'")

    #     hits2 = self.searcher.search(rewritten, k=self.cfg.top_k)
    #     context2 = self._build_context(hits2)
    #     gen2 = self.generator.answer(question, context2, temperature=0.0)
    #     top1_2 = hits2[0]["score"] if hits2 else None
    #     c_retr2 = retrieval_conf(top1_2)
    #     c2 = combine_conf(gen2["gen_conf"], c_retr2, alpha=self.cfg.alpha_gen)

    #     return {
    #         "question": question,
    #         "answer": gen2["answer"],
    #         "gen_conf": gen2["gen_conf"],
    #         "retr_conf": c_retr2,
    #         "confidence": c2,
    #         "raw": gen2["raw"],
    #         "hits": hits2,
    #         "adaptive_triggered": True,
    #         "rewritten_query": rewritten,
    #         "first_pass": res,
    #     }
    def answer_adaptive(self, question: str) -> Dict[str, Any]:
        res = self.answer_baseline(question)
    
        # If already good, stop
        if res["confidence"] >= self.cfg.confidence_threshold:
            return res
    
        history = []
        current_query = question
        current_res = res
    
        for step in range(self.cfg.max_adaptive_steps):
            snippets = self._snippets(current_res["hits"])
            rewritten = self.rewriter.rewrite(question, snippets)
            logger.info(f"Adaptive step {step+1}: '{current_query}' -> '{rewritten}'")
    
            hits = self.searcher.search(rewritten, k=self.cfg.top_k)
            context = self._build_context(hits)
            gen = self.generator.answer(question, context, temperature=0.0)
    
            # top1 = hits[0]["score"] if hits else None
            # c_retr = retrieval_conf(top1)
            def bm25_to_conf(score: float) -> float:
                # score scale varies by corpus; this squashes reasonably
                return 1.0 / (1.0 + math.exp(-0.15 * (score - 8.0)))
            
            top1 = hits[0]["score"] if hits else 0.0
            c_retr = bm25_to_conf(top1)
            c = combine_conf(gen["gen_conf"], c_retr, alpha=self.cfg.alpha_gen)
    
            new_res = {
                "question": question,
                "answer": gen["answer"],
                "gen_conf": gen["gen_conf"],
                "retr_conf": c_retr,
                "confidence": c,
                "raw": gen["raw"],
                "hits": hits,
                "adaptive_triggered": True,
                "rewritten_query": rewritten,
            }
    
            history.append({"step": step+1, "query": rewritten, "confidence": c})
    
            if c >= self.cfg.confidence_threshold:
                new_res["rewrite_history"] = history
                new_res["first_pass"] = res
                return new_res
    
            current_query = rewritten
            current_res = new_res
    
        # If never met threshold, return last attempt but include history
        current_res["rewrite_history"] = history
        current_res["first_pass"] = res
        return current_res

