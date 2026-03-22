from __future__ import annotations
from generator.local import LocalGenerator

def run(sample: dict, generator=None) -> dict:
    gen = generator or LocalGenerator(model_name="cot-baseline")
    out = gen.generate_one(sample["question"], context=sample.get("context", ""))
    return {"prediction": out["answer"], "tokens": out["tokens"], "latency": out["latency"], "method": "cot"}
