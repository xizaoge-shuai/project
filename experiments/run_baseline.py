from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse
from utils.io import read_jsonl, write_jsonl
from utils.eval_utils import is_correct_prediction
from baselines import cot, self_consistency, cisc_like, tot, beam_search, bon, gg_like

METHODS = {
    "cot": cot.run,
    "self_consistency": self_consistency.run,
    "cisc_like": cisc_like.run,
    "tot": tot.run,
    "beam_search": beam_search.run,
    "bon": bon.run,
    "gg_like": gg_like.run,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["gsm8k", "strategyqa", "hotpotqa"])
    parser.add_argument("--method", required=True, choices=list(METHODS))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    rows = read_jsonl(f"data/processed/unified/{args.dataset}.jsonl")
    answer_mode = "numeric" if args.dataset == "gsm8k" else ("yesno" if args.dataset == "strategyqa" else "span")
    out = []
    for row in rows:
        pred = METHODS[args.method](row)
        out.append({
            "id": row["id"],
            "question": row["question"],
            "gold_answer": row["answer"],
            "prediction": pred["prediction"],
            "tokens": pred["tokens"],
            "latency": pred["latency"],
            "is_correct": int(is_correct_prediction(pred["prediction"], row["answer"], answer_mode)),
            "method": pred["method"],
        })
    out_path = args.out or f"outputs/predictions/{args.dataset}_{args.method}.jsonl"
    write_jsonl(out_path, out)
    print(f"Saved predictions to {out_path}")

if __name__ == "__main__":
    main()
