from __future__ import annotations
from generator.local import LocalGenerator
from pce.models.verifier import VerifierPCE

def run(sample: dict, generator=None, pce=None, n: int = 5) -> dict:
    gen = generator or LocalGenerator(model_name="gg-like")
    pce = pce or VerifierPCE()
    best = None
    best_score = -1.0
    total_tokens = 0
    total_latency = 0.0
    for _ in range(n):
        out = gen.generate_one(sample["question"], context=sample.get("context", ""))
        score = pce.predict(sample["question"], out["text"], out["answer"])["success_prob"] + 0.01 * len(set(out["steps"]))
        if score > best_score:
            best = out
            best_score = score
        total_tokens += out["tokens"]
        total_latency += out["latency"]
    return {"prediction": best["answer"], "tokens": total_tokens, "latency": total_latency, "method": "gg_like"}
