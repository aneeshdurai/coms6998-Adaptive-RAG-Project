import json
from typing import Dict, Any, List, Tuple
from workspace.utils.logging_utils import get_logger

logger = get_logger("retrieval.corpus_store")

class CorpusStore:
    def __init__(self, chunks_path: str):
        self._texts: Dict[Tuple[str,int], str] = {}
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                key = (rec["doc_id"], int(rec["chunk_id"]))
                self._texts[key] = rec["text"]
        logger.info(f"Loaded chunk texts: {len(self._texts)}")

    def get_text(self, doc_id: str, chunk_id: int) -> str:
        return self._texts.get((doc_id, int(chunk_id)), "")
