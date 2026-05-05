from __future__ import annotations

import sys
import inspect
import pickle
import argparse
from pathlib import Path
from collections import Counter
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.io import read_jsonl, write_jsonl, read_yaml
from reasoning.pipeline import ReasoningPipeline
from controller.threshold import ThresholdController
from controller.budget_aware import BudgetAwareController


def load_pce(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def build_controller(controller_cls, cfg: dict):
    """
    只保留 controller.__init__ 真正支持的参数，避免 yaml 里有旧键时直接炸掉。
    """
    sig = inspect.signature(controller_cls.__init__)
    allowed = set(sig.parameters.keys()) - {"self"}
    filtered = {k: v for k, v in cfg.items() if k in allowed}
    dropped = sorted(k for k in cfg.keys() if k not in allowed)
    if dropped:
        print(f"[warn] dropped unsupported controller config keys: {dropped}")
    return controller_cls(**filtered)


def infer_trajectory_path(dataset: str) -> str:
    """
    优先用 trajectories.jsonl；如果没有，再回退到 test.jsonl
    """
    cand1 = Path(f"data/processed/trajectories/{dataset}/trajectories.jsonl")
    cand2 = Path(f"data/processed/trajectories/{dataset}/test.jsonl")
    if cand1.exists():
        return str(cand1)
    if cand2.exists():
        return str(cand2)
    raise FileNotFoundError(
        f"Cannot find trajectory file for dataset={dataset}. "
        f"Tried: {cand1} and {cand2}"
    )


def summarize_results(rows: list[dict]) -> dict:
    if not rows:
        return {
            "n_samples": 0,
            "final_accuracy": None,
            "avg_tokens": None,
            "avg_latency": None,
            "avg_backtracks": None,
        }

    n = len(rows)
    final_acc = mean(float(x.get("is_correct", 0)) for x in rows)

    token_vals = [float(x.get("tokens", 0.0)) for x in rows]
    latency_vals = [float(x.get("latency", 0.0)) for x in rows]
    backtrack_vals = [float(x.get("backtrack_count", 0.0)) for x in rows]

    action_counter = Counter()
    for x in rows:
        for a in x.get("actions", []):
            action_counter[a] += 1

    num_with_accept = sum(1 for x in rows if "accept" in x.get("actions", []))
    num_with_prune = sum(1 for x in rows if "prune" in x.get("actions", []))
    num_with_backtrack = sum(1 for x in rows if "backtrack" in x.get("actions", []))

    return {
        "n_samples": n,
        "final_accuracy": final_acc,
        "avg_tokens": mean(token_vals),
        "avg_latency": mean(latency_vals),
        "avg_backtracks": mean(backtrack_vals),
        "num_with_accept": num_with_accept,
        "num_with_prune": num_with_prune,
        "num_with_backtrack": num_with_backtrack,
        "accept_rate": num_with_accept / n,
        "prune_rate": num_with_prune / n,
        "backtrack_rate": num_with_backtrack / n,
        "action_counter": dict(action_counter),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["gsm8k", "strategyqa", "hotpotqa"]
    )
    parser.add_argument(
        "--controller",
        default="threshold",
        choices=["threshold", "budget_aware"]
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/checkpoints/pce.pkl"
    )
    parser.add_argument(
        "--trajectory_path",
        default=None,
        help="Optional override for trajectory jsonl path"
    )
    parser.add_argument(
        "--controller_config",
        default=None,
        help="Optional override for controller yaml path"
    )
    parser.add_argument(
        "--budget_tokens",
        type=int,
        default=256
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Only run first N samples for quick debugging; -1 means all"
    )
    parser.add_argument(
        "--out",
        default=None
    )
    parser.add_argument(
        "--summary_out",
        default=None
    )
    args = parser.parse_args()

    trajectory_path = args.trajectory_path or infer_trajectory_path(args.dataset)
    trajectories = read_jsonl(trajectory_path)

    if args.limit > 0:
        trajectories = trajectories[: args.limit]

    pce = load_pce(args.checkpoint)

    if args.controller == "budget_aware":
        cfg_path = args.controller_config or "configs/controller/budget_aware.yaml"
        c_cfg = read_yaml(cfg_path)
        controller = build_controller(BudgetAwareController, c_cfg)
    else:
        cfg_path = args.controller_config or "configs/controller/threshold.yaml"
        c_cfg = read_yaml(cfg_path)
        controller = build_controller(ThresholdController, c_cfg)

    answer_mode = (
        "numeric" if args.dataset == "gsm8k"
        else ("yesno" if args.dataset == "strategyqa" else "span")
    )

    pipeline = ReasoningPipeline(
        pce=pce,
        controller=controller,
        answer_mode=answer_mode,
    )

    out = []
    for row in trajectories:
        result = pipeline.run(
            sample_id=row["sample_id"],
            question=row["question"],
            candidate_steps=row["steps"],
            gold_answer=row["gold_answer"],
            context=row.get("context", ""),
            budget_tokens=args.budget_tokens,
        )
        result["gold_answer"] = row["gold_answer"]
        result["question"] = row["question"]
        out.append(result)

    out_path = args.out or f"outputs/predictions/{args.dataset}_pce_{args.controller}.jsonl"
    write_jsonl(out_path, out)

    summary = summarize_results(out)
    summary["dataset"] = args.dataset
    summary["controller"] = args.controller
    summary["checkpoint"] = args.checkpoint
    summary["trajectory_path"] = trajectory_path
    summary["controller_config"] = cfg_path
    summary["budget_tokens"] = args.budget_tokens

    summary_out = args.summary_out or f"outputs/metrics/{args.dataset}_pce_{args.controller}_summary.json"
    Path(summary_out).parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved PCE-controlled predictions to {out_path}")
    print(f"Saved summary metrics to {summary_out}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()