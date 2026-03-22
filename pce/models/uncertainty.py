from __future__ import annotations
import numpy as np

def entropy_from_prob(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(-(p * np.log(p) + (1 - p) * np.log(1 - p)))
