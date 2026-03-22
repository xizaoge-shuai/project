from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse
import pandas as pd
from utils.io import read_jsonl
from utils.eval_utils import is_correct_prediction

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--answer_mode", default="numeric")
    args = parser.parse_args()
    rows = read_jsonl(args.predictions)
    df = pd.DataFrame(rows)
    if "is_correct" not in df.columns:
        df["is_correct"] = [is_correct_prediction(p, g, args.answer_mode) for p, g in zip(df["prediction"], df["gold_answer"])]
    print({
        "accuracy": float(df["is_correct"].mean()) if len(df) else 0.0,
        "avg_tokens": float(df["tokens"].mean()) if len(df) else 0.0,
        "avg_latency": float(df["latency"].mean()) if len(df) else 0.0,
    })

if __name__ == "__main__":
    main()
