from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse, pickle
from utils.io import read_jsonl, write_jsonl, read_yaml
from reasoning.pipeline import ReasoningPipeline
from controller.threshold import ThresholdController
from controller.budget_aware import BudgetAwareController

def load_pce(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["gsm8k", "strategyqa", "hotpotqa"])
    parser.add_argument("--controller", default="threshold", choices=["threshold", "budget_aware"])
    parser.add_argument("--checkpoint", default="outputs/checkpoints/pce.pkl")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    trajectories = read_jsonl(f"data/processed/trajectories/{args.dataset}/trajectories.jsonl")
    pce = load_pce(args.checkpoint)
    if args.controller == "budget_aware":
        c_cfg = read_yaml("configs/controller/budget_aware.yaml")
        controller = BudgetAwareController(**c_cfg)
    else:
        c_cfg = read_yaml("configs/controller/threshold.yaml")
        controller = ThresholdController(**c_cfg)
    answer_mode = "numeric" if args.dataset == "gsm8k" else ("yesno" if args.dataset == "strategyqa" else "span")
    pipeline = ReasoningPipeline(pce=pce, controller=controller, answer_mode=answer_mode)
    out = []
    for row in trajectories:
        result = pipeline.run(
            sample_id=row["sample_id"],
            question=row["question"],
            candidate_steps=row["steps"],
            gold_answer=row["gold_answer"],
            context=row.get("context", ""),
            budget_tokens=256,
        )
        result["gold_answer"] = row["gold_answer"]
        result["question"] = row["question"]
        out.append(result)
    out_path = args.out or f"outputs/predictions/{args.dataset}_pce_{args.controller}.jsonl"
    write_jsonl(out_path, out)
    print(f"Saved PCE-controlled predictions to {out_path}")

if __name__ == "__main__":
    main()
