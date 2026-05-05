from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import Counter
from statistics import mean
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.eval_utils import is_correct_prediction
from controller.repair_safety import safe_repair_gate, normalize_num


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: str, obj: Dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def correct(ans: str, gold: str) -> bool:
    return bool(
        is_correct_prediction(
            normalize_num(ans),
            normalize_num(gold),
            answer_mode="numeric",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--details_out", default=None)
    args = parser.parse_args()

    rows = read_jsonl(args.input)

    details = []
    for r in rows:
        triggered = bool(r.get("triggered", False))
        gold = str(r.get("gold_answer", ""))
        original_is_correct = bool(r.get("original_is_correct", False))

        original_answer = str(r.get("current_answer", "") or "")
        if not original_answer:
            # 如果没有 current_answer，就用 gold 之外的信息不可得；
            # 对已有 local rewrite 文件，harmed_good/recovered 依赖 original_is_correct 字段。
            # 这里只用于 safe gate 统计，所以原答案无法恢复时保留 repaired_final_answer 前的状态。
            original_answer = str(r.get("original_final_answer", "") or "")

        repair_output = r.get("repair_output", None)
        old_repaired_final_answer = str(r.get("repaired_final_answer", "") or "")

        if not triggered:
            safe_action = "NO_TRIGGER"
            safe_final_answer = old_repaired_final_answer
            safe_reason = "not_triggered"
            safe_is_correct = original_is_correct
        else:
            gate = safe_repair_gate(
                repair_output=repair_output or "",
                original_answer=original_answer,
            )
            safe_action = gate["action"]
            safe_final_answer = gate["final_answer"]
            safe_reason = gate["reason"]

            if safe_action == "KEEP":
                safe_is_correct = original_is_correct
            else:
                safe_is_correct = correct(safe_final_answer, gold)

        recovered_bad_safe = (
            triggered
            and (not original_is_correct)
            and safe_is_correct
        )
        preserved_good_safe = (
            triggered
            and original_is_correct
            and safe_is_correct
        )
        harmed_good_safe = (
            triggered
            and original_is_correct
            and (not safe_is_correct)
        )

        details.append(
            {
                **r,
                "safe_action": safe_action,
                "safe_reason": safe_reason,
                "safe_final_answer": safe_final_answer,
                "safe_is_correct": bool(safe_is_correct),
                "safe_recovered_bad": bool(recovered_bad_safe),
                "safe_preserved_good": bool(preserved_good_safe),
                "safe_harmed_good": bool(harmed_good_safe),
            }
        )

    triggered_rows = [d for d in details if d.get("triggered")]
    triggered_bad = [d for d in triggered_rows if not d.get("original_is_correct")]
    triggered_good = [d for d in triggered_rows if d.get("original_is_correct")]

    old_harmed_good = sum(1 for d in details if d.get("harmed_good"))
    old_recovered = sum(1 for d in details if d.get("recovered"))
    old_preserved_good = sum(1 for d in details if d.get("preserved_good"))

    safe_harmed_good = sum(1 for d in details if d.get("safe_harmed_good"))
    safe_recovered_bad = sum(1 for d in details if d.get("safe_recovered_bad"))
    safe_preserved_good = sum(1 for d in details if d.get("safe_preserved_good"))

    action_counter = Counter(d.get("safe_action") for d in details)
    reason_counter = Counter(d.get("safe_reason") for d in details)

    summary = {
        "input": args.input,
        "n_total": len(details),
        "n_triggered": len(triggered_rows),
        "n_triggered_bad": len(triggered_bad),
        "n_triggered_good": len(triggered_good),

        "old_recovered": old_recovered,
        "old_preserved_good": old_preserved_good,
        "old_harmed_good": old_harmed_good,

        "safe_recovered_bad": safe_recovered_bad,
        "safe_preserved_good": safe_preserved_good,
        "safe_harmed_good": safe_harmed_good,

        "safe_recover_rate_bad": safe_recovered_bad / max(1, len(triggered_bad)),
        "safe_preserve_rate_good": safe_preserved_good / max(1, len(triggered_good)),
        "safe_harm_rate_good": safe_harmed_good / max(1, len(triggered_good)),

        "safe_action_counter": dict(action_counter),
        "safe_reason_counter": dict(reason_counter),
    }

    write_json(args.out, summary)
    if args.details_out:
        write_jsonl(args.details_out, details)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
