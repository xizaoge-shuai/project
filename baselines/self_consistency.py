from __future__ import annotations
from generator.local import LocalGenerator
from utils.eval_utils import majority_vote

def run(sample: dict, generator=None, n: int = 5) -> dict:
    gen = generator or LocalGenerator(model_name="self-consistency")
    answers, total_tokens, total_latency = [], 0, 0.0
    for _ in range(n):
        out = gen.generate_one(sample["question"], context=sample.get("context", ""))
        answers.append(out["answer"])
        total_tokens += out["tokens"]
        total_latency += out["latency"]
    pred = majority_vote(answers)
    return {"prediction": pred, "tokens": total_tokens, "latency": total_latency, "method": "self_consistency"}
