from typing import List

def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))

def retrieval_conf(top1_score: float | None) -> float:
    if top1_score is None:
        return 0.0
    return clamp(float(top1_score), 0.0, 1.0)

def combine_conf(gen_conf: float, retr_conf: float, alpha: float = 0.6) -> float:
    return clamp(alpha * gen_conf + (1 - alpha) * retr_conf, 0.0, 1.0)
