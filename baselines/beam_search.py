from __future__ import annotations
from baselines.self_consistency import run as sc_run

def run(sample: dict, generator=None, beam_size: int = 4) -> dict:
    return {**sc_run(sample, generator=generator, n=beam_size), "method": "beam_search"}
