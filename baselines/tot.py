from __future__ import annotations
from baselines.self_consistency import run as sc_run

def run(sample: dict, generator=None, depth: int = 3, branching: int = 2) -> dict:
    # 第一版用 SC 近似 ToT 搜索
    return {**sc_run(sample, generator=generator, n=depth * branching), "method": "tot"}
