from __future__ import annotations
from generator.local import LocalGenerator
from utils.eval_utils import extract_last_number

def run(sample: dict, generator=None, n: int = 8) -> dict:
    gen = generator or LocalGenerator(model_name="bon")
    candidates = [gen.generate_one(sample["question"], context=sample.get("context", "")) for _ in range(n)]
    # 简单启发式：选择最长推理
    best = max(candidates, key=lambda x: len(x["text"]))
    return {"prediction": best["answer"], "tokens": sum(c["tokens"] for c in candidates), "latency": sum(c["latency"] for c in candidates), "method": "bon"}
