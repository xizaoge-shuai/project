from __future__ import annotations

VALID_ACTIONS = ["continue", "prune", "backtrack", "accept"]

def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))
