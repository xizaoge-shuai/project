from __future__ import annotations
from generator.local import LocalGenerator
from pce.models.verifier import VerifierPCE

def run(sample: dict, generator=None, pce=None, n: int = 5) -> dict:
    gen = generator or LocalGenerator(model_name="cisc-like")
    pce = pce or VerifierPCE()
    candidates = []
    for _ in range(n):
        out = gen.generate_one(sample["question"], context=sample.get("context", ""))
        score = pce.predict(sample["question"], "\n".join(out["steps"]), out["answer"])["success_prob"]
        candidates.append((score, out))
    candidates.sort(key=lambda x: x[0], reverse=True)
    best = candidates[0][1]
    return {"prediction": best["answer"], "tokens": sum(c[1]["tokens"] for c in candidates), "latency": sum(c[1]["latency"] for c in candidates), "method": "cisc_like"}
