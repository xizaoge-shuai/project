from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse
import pandas as pd
from utils.io import read_jsonl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", required=True, help="多个结果文件")
    parser.add_argument("--names", nargs="+", required=True, help="对应名称")
    parser.add_argument("--out", default="outputs/metrics/ablation.csv")
    args = parser.parse_args()
    rows = []
    for name, fp in zip(args.names, args.files):
        df = pd.DataFrame(read_jsonl(fp))
        rows.append({
            "variant": name,
            "accuracy": float(df["is_correct"].mean()) if len(df) else 0.0,
            "avg_tokens": float(df["tokens"].mean()) if len(df) else 0.0,
            "avg_latency": float(df["latency"].mean()) if len(df) else 0.0,
        })
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
