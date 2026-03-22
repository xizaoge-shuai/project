from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse
import pandas as pd
from utils.io import read_jsonl

GENERATOR_SPEED = {7: 120.0, 32: 60.0, 70: 30.0}
PCE_OVERHEAD_MS = {110: 8.0, 500: 20.0, 1500: 50.0}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_preds", required=True)
    parser.add_argument("--pce_preds", required=True)
    parser.add_argument("--generator_size", type=int, default=32, choices=[7, 32, 70])
    parser.add_argument("--pce_size", type=int, default=110, choices=[110, 500, 1500])
    parser.add_argument("--out", default="outputs/metrics/overhead.csv")
    args = parser.parse_args()

    base = pd.DataFrame(read_jsonl(args.baseline_preds))
    pce = pd.DataFrame(read_jsonl(args.pce_preds))
    if len(base) == 0 or len(pce) == 0:
        raise ValueError("Prediction files are empty.")
    avg_base_tokens = base["tokens"].mean()
    avg_pce_tokens = pce["tokens"].mean()
    saved_tokens = avg_base_tokens - avg_pce_tokens
    token_speed = GENERATOR_SPEED[args.generator_size]
    saved_latency = saved_tokens / token_speed
    pce_overhead = PCE_OVERHEAD_MS[args.pce_size] / 1000.0 * pce.get("actions", pd.Series([[]]*len(pce))).apply(len).mean()
    roi = saved_latency / max(pce_overhead, 1e-6)
    out = pd.DataFrame([{
        "generator_size_b": args.generator_size,
        "pce_size_m": args.pce_size,
        "avg_base_tokens": avg_base_tokens,
        "avg_pce_tokens": avg_pce_tokens,
        "saved_tokens": saved_tokens,
        "saved_latency_sec": saved_latency,
        "pce_overhead_sec": pce_overhead,
        "roi": roi,
    }])
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
