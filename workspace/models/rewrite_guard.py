import re
from dataclasses import dataclass
from typing import Set, List

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_FY_RE = re.compile(r"\bFY\s?(19|20)\d{2}\b", re.IGNORECASE)
_NUM_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?(?![A-Za-z])")

SAFE_WORDS = {
    "what","which","when","where","how","why","did","does","is","are","was","were",
    "the","a","an","of","for","to","in","on","by","and","or","from","with","at",
    "company","inc","inc.","corp","corporation","ltd","llc","plc","co","co."
}

def _extract_years(text: str) -> Set[str]:
    years = {m.group(0) for m in _YEAR_RE.finditer(text)}
    # FY2023 -> 2023
    years |= {re.sub(r"[^0-9]", "", m.group(0)) for m in _FY_RE.finditer(text)}
    return {y for y in years if y}

def _extract_numbers(text: str) -> Set[str]:
    out = set()
    for m in _NUM_RE.finditer(text):
        out.add(m.group(0).replace(",", ""))
    return out

def _tokens(text: str) -> Set[str]:
    toks = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?%?", text.lower())
    return set(toks)

@dataclass
class RewriteCheck:
    ok: bool
    reasons: List[str]

def validate_rewrite(original_q: str, rewritten_q: str, *, forbid_new_years: bool = True, forbid_new_numbers: bool = True) -> RewriteCheck:
    reasons: List[str] = []

    orig_years = _extract_years(original_q)
    new_years  = _extract_years(rewritten_q)
    added_years = new_years - orig_years
    if forbid_new_years and added_years:
        reasons.append(f"added_years={sorted(added_years)}")

    orig_nums = _extract_numbers(original_q)
    new_nums  = _extract_numbers(rewritten_q)
    added_nums = new_nums - orig_nums
    if forbid_new_numbers and added_nums:
        reasons.append(f"added_numbers={sorted(added_nums)}")

    # Mild topic drift guard: rewritten should share some non-trivial tokens
    o = _tokens(original_q) - SAFE_WORDS
    r = _tokens(rewritten_q) - SAFE_WORDS
    if o:
        overlap = len(o & r) / max(1, len(o))
        if overlap < 0.35:
            reasons.append(f"low_token_overlap={overlap:.2f}")

    return RewriteCheck(ok=(len(reasons) == 0), reasons=reasons)
