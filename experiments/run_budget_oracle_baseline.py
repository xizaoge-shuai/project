from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from collections import Counter
from statistics import mean
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.io import read_jsonl
from utils.tokenizer_utils import count_tokens
from utils.eval_utils import is_correct_prediction


def write_json(path: str, obj: Dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def answer_mode(dataset: str) -> str:
    if dataset == "gsm8k":
        return "numeric"
    if dataset == "strategyqa":
        return "yesno"
    return "span"


def extract_answer(prefix_items: List[str]) -> str:
    for s in reversed(prefix_items):
        if "Answer:" in s:
            return s.split("Answer:", 1)[-1].strip()
    return "\n".join(prefix_items).strip()


def run_one(row: Dict[str, Any], dataset: str, budget_tokens: int, strict: bool) -> Dict[str, Any]:
    steps = row.get("steps", [])
    prefix = []
    tokens = 0.0
    terminal = "end"

    for step in steps:
        step_tokens = count_tokens(step)

        if strict and tokens + step_tokens > budget_tokens:
            terminal = "budget_stop"
            break

        prefix.append(step)
        tokens += step_tokens

        if (not strict) and tokens > budget_tokens:
            terminal = "budget_exceeded"
            break

    if len(prefix) == len(steps):
        terminal = "end"

    pred = extract_answer(prefix)
    correct = int(is_correct_prediction(pred, row.get("gold_answer", ""), answer_mode=answer_mode(dataset)))

    return {
        "sample_id": row.get("sample_id", ""),
        "tokens": float(tokens),
        "terminal_action": terminal,
        "is_correct": correct,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "strategyqa", "hotpotqa"])
    parser.add_argument("--trajectory_path", required=True)
    parser.add_argument("--budget_tokens", type=int, required=True)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--strict", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.trajectory_path)
    if args.limit > 0:
        rows = rows[:args.limit]

    details = [
        run_one(
            row=r,
            dataset=args.dataset,
            budget_tokens=args.budget_tokens,
            strict=bool(args.strict),
        )
        for r in rows
    ]

    counter = Counter(d["terminal_action"] for d in details)

    summary = {
        "setting": "run_to_budget_or_end",
        "dataset": args.dataset,
        "budget_tokens": args.budget_tokens,
        "strict": bool(args.strict),
        "n_samples": len(details),
        "final_accuracy": mean(d["is_correct"] for d in details) if details else None,
        "avg_tokens": mean(d["tokens"] for d in details) if details else None,
        "terminal_action_counter": dict(counter),
    }

    write_json(args.out, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
