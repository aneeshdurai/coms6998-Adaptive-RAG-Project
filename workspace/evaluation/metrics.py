# import re

# def normalize(s: str) -> str:
#     s = (s or "").strip().lower()
#     s = re.sub(r"\s+", " ", s)
#     return s

# # def exact_match(pred: str, gold: str) -> float:
# #     return 1.0 if normalize(pred) == normalize(gold) else 0.0
# def exact_match(pred: str, gold: str) -> float:
#     return 1.0 if normalize_numberish(pred) == normalize_numberish(gold) else 0.0

# def numeric_exact_or_close(pred: str, gold: str, rel_tol: float = 1e-3) -> float:
#     p = normalize_numberish(pred)
#     g = normalize_numberish(gold)

#     # try parse floats
#     def to_float(x):
#         x = x.replace("%", "")
#         return float(x)

#     try:
#         pf = to_float(p)
#         gf = to_float(g)
#         if gf == 0:
#             return 1.0 if pf == 0 else 0.0
#         return 1.0 if abs(pf - gf) / abs(gf) <= rel_tol else 0.0
#     except Exception:
#         return 1.0 if p == g else 0.0


# def token_f1(pred: str, gold: str) -> float:
#     p = normalize(pred).split()
#     g = normalize(gold).split()
#     if not p and not g:
#         return 1.0
#     if not p or not g:
#         return 0.0
#     common = {}
#     for t in p:
#         common[t] = common.get(t, 0) + 1
#     num_same = 0
#     for t in g:
#         if common.get(t, 0) > 0:
#             num_same += 1
#             common[t] -= 1
#     if num_same == 0:
#         return 0.0
#     precision = num_same / len(p)
#     recall = num_same / len(g)
#     return 2 * precision * recall / (precision + recall)

# import re

# def normalize_numberish(s: str) -> str:
#     s = (s or "").strip().lower()

#     # common cleanup
#     s = s.replace(",", "")          # 1,080 -> 1080
#     s = s.replace("$", "")          # $1080 -> 1080
#     s = s.replace("usd", "")        # usd 1080 -> 1080
#     s = s.replace("percent", "%")   # 12.8 percent -> 12.8%
#     s = re.sub(r"\s+", " ", s).strip()

#     return s


import re
from typing import List, Optional, Dict, Any

def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def normalize_numberish(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace(",", "")
    s = s.replace("$", "")
    s = s.replace("usd", "")
    s = s.replace("percent", "%")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def exact_match_norm(pred: str, gold: str) -> float:
    return 1.0 if normalize_numberish(pred) == normalize_numberish(gold) else 0.0

def token_f1(pred: str, gold: str) -> float:
    p = normalize_text(pred).split()
    g = normalize_text(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0

    common = {}
    for t in p:
        common[t] = common.get(t, 0) + 1
    num_same = 0
    for t in g:
        if common.get(t, 0) > 0:
            num_same += 1
            common[t] -= 1

    if num_same == 0:
        return 0.0
    precision = num_same / len(p)
    recall = num_same / len(g)
    return 2 * precision * recall / (precision + recall)

def rouge_l(pred: str, gold: str) -> float:
    # LCS-based ROUGE-L F-measure
    p = normalize_text(pred).split()
    g = normalize_text(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0

    # LCS DP (ok for short answers)
    dp = [[0]*(len(g)+1) for _ in range(len(p)+1)]
    for i in range(1, len(p)+1):
        for j in range(1, len(g)+1):
            if p[i-1] == g[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[-1][-1]
    prec = lcs / len(p)
    rec = lcs / len(g)
    if prec + rec == 0:
        return 0.0
    return (2 * prec * rec) / (prec + rec)

def answer_in_evidence(pred: str, evidence_text: str) -> float:
    # Simple support overlap proxy: does normalized predicted answer appear in evidence?
    p = normalize_numberish(pred)
    ev = normalize_numberish(evidence_text)
    if not p:
        return 0.0
    return 1.0 if p in ev else 0.0

def recall_at_k(hits: List[Dict[str, Any]], gold_doc_id: Optional[str], k: int) -> float:
    if not gold_doc_id:
        return 0.0
    for h in hits[:k]:
        if h.get("doc_id") == gold_doc_id:
            return 1.0
    return 0.0

def mrr_at_k(hits: List[Dict[str, Any]], gold_doc_id: Optional[str], k: int) -> float:
    if not gold_doc_id:
        return 0.0
    for i, h in enumerate(hits[:k], start=1):
        if h.get("doc_id") == gold_doc_id:
            return 1.0 / i
    return 0.0
