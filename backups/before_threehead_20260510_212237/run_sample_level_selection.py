from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.eval_utils import is_correct_prediction


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    return [json.loads(x) for x in p.open("r", encoding="utf-8") if x.strip()]


def write_json(path: str, obj: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def norm_num(x: Any) -> str:
    x = str(x or "").replace(",", "").replace("$", "")
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", x)
    if not nums:
        return str(x).strip()
    v = nums[-1]
    if "." in v:
        v = v.rstrip("0").rstrip(".")
    return v


def extract_answer_from_steps(steps: List[str]) -> str:
    for s in reversed(steps or []):
        s = str(s)
        if "Final Answer:" in s:
            return s.split("Final Answer:", 1)[-1].strip()
        if "Answer:" in s:
            return s.split("Answer:", 1)[-1].strip()
        if "####" in s:
            return s.split("####", 1)[-1].strip()

    joined = "\n".join(str(x) for x in steps or [])
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", joined.replace(",", ""))
    return nums[-1] if nums else joined.strip()


def traj_order(tid: str) -> int:
    m = re.search(r"_traj_(\d+)$", str(tid))
    return int(m.group(1)) if m else 999999


def get_score(scores_by_tid: Dict[str, List[tuple]], tid: str, mode: str) -> float:
    arr = sorted(scores_by_tid.get(tid, []), key=lambda x: x[0])
    vals = [x[1] for x in arr]
    if not vals:
        return 1.0

    if mode == "last":
        return vals[-1]
    if mode == "tail3":
        return sum(vals[-3:]) / min(3, len(vals))
    if mode == "tail5":
        return sum(vals[-5:]) / min(5, len(vals))
    if mode == "min_tail5":
        return min(vals[-5:])
    if mode == "mean":
        return sum(vals) / len(vals)
    if mode == "median":
        vals2 = sorted(vals)
        return vals2[len(vals2) // 2]

    raise ValueError(f"Unknown score mode: {mode}")


def weighted_pred(rs: List[Dict[str, Any]], field: str, scores_by_tid: Dict[str, List[tuple]], mode: str) -> str:
    votes = defaultdict(float)
    for x in rs:
        votes[x[field]] += get_score(scores_by_tid, x["tid"], mode)
    return max(votes.items(), key=lambda z: z[1])[0]


def majority_pred(rs: List[Dict[str, Any]], field: str) -> str:
    votes = Counter(x[field] for x in rs)
    return votes.most_common(1)[0][0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="Prefix-level PCE predictions jsonl.")
    parser.add_argument("--trajectories", required=True, help="Trajectory bank jsonl.")
    parser.add_argument("--repair_jsonl", default=None, help="Optional safe local rewrite result jsonl.")
    parser.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "strategyqa", "hotpotqa"])
    parser.add_argument("--out", default=None)
    parser.add_argument("--details_out", default=None)
    args = parser.parse_args()

    answer_mode = "numeric" if args.dataset == "gsm8k" else ("yesno" if args.dataset == "strategyqa" else "span")

    pred_rows = read_jsonl(args.predictions)
    traj_rows = read_jsonl(args.trajectories)

    repair_by_tid = {}
    if args.repair_jsonl:
        repair_rows = read_jsonl(args.repair_jsonl)
        repair_by_tid = {r["trajectory_id"]: r for r in repair_rows}
    else:
        repair_rows = []

    scores_by_tid = defaultdict(list)
    for r in pred_rows:
        if "success_prob" not in r:
            continue
        tid = r["trajectory_id"]
        k = int(r.get("prefix_num_units", r.get("prefix_index", 0)))
        p = float(r["success_prob"])
        scores_by_tid[tid].append((k, p))

    by_sample = defaultdict(list)

    for tr in traj_rows:
        tid = tr["trajectory_id"]
        sid = tr["sample_id"]
        gold = str(tr["gold_answer"])

        before_raw = extract_answer_from_steps(tr.get("steps", []))
        before = norm_num(before_raw)

        after = before
        repaired = False

        rr = repair_by_tid.get(tid)
        if rr and rr.get("repair_decision") == "REWRITE" and str(rr.get("repaired_final_answer", "")).strip():
            after = norm_num(rr.get("repaired_final_answer"))
            repaired = True

        by_sample[sid].append({
            "sid": sid,
            "tid": tid,
            "gold": gold,
            "before": before,
            "after": after,
            "before_raw": before_raw,
            "before_ok": int(is_correct_prediction(before, gold, answer_mode=answer_mode)),
            "after_ok": int(is_correct_prediction(after, gold, answer_mode=answer_mode)),
            "repaired": repaired,
        })

    modes = ["last", "tail3", "tail5", "min_tail5", "mean", "median"]

    n = len(by_sample)
    summary = {
        "prediction_file": args.predictions,
        "trajectory_file": args.trajectories,
        "repair_jsonl": args.repair_jsonl,
        "dataset": args.dataset,
        "n_samples": n,
        "n_trajectories": len(traj_rows),
        "n_prefix_predictions": len(pred_rows),
    }

    counts = Counter()
    details = []

    weighted_before = {m: 0 for m in modes}
    weighted_after = {m: 0 for m in modes}
    pce_top1 = {m: 0 for m in modes}

    for sid, rs in by_sample.items():
        rs = sorted(rs, key=lambda x: traj_order(x["tid"]))
        gold = rs[0]["gold"]

        first_before_ok = rs[0]["before_ok"]
        first_after_ok = rs[0]["after_ok"]

        any_before_ok = int(any(x["before_ok"] for x in rs))
        any_after_ok = int(any(x["after_ok"] for x in rs))

        before_maj = majority_pred(rs, "before")
        after_maj = majority_pred(rs, "after")

        before_maj_ok = int(is_correct_prediction(before_maj, gold, answer_mode=answer_mode))
        after_maj_ok = int(is_correct_prediction(after_maj, gold, answer_mode=answer_mode))

        before_weighted_tail5 = weighted_pred(rs, "before", scores_by_tid, "tail5")
        after_weighted_tail5 = weighted_pred(rs, "after", scores_by_tid, "tail5")

        before_weighted_tail5_ok = int(is_correct_prediction(before_weighted_tail5, gold, answer_mode=answer_mode))
        after_weighted_tail5_ok = int(is_correct_prediction(after_weighted_tail5, gold, answer_mode=answer_mode))

        counts["first_before_ok"] += first_before_ok
        counts["first_after_ok"] += first_after_ok
        counts["majority_before_ok"] += before_maj_ok
        counts["majority_after_ok"] += after_maj_ok
        counts["any_before_ok"] += any_before_ok
        counts["any_after_ok"] += any_after_ok

        for mode in modes:
            best = max(rs, key=lambda x: get_score(scores_by_tid, x["tid"], mode))
            pce_top1[mode] += int(is_correct_prediction(best["before"], gold, answer_mode=answer_mode))

            pred_b = weighted_pred(rs, "before", scores_by_tid, mode)
            pred_a = weighted_pred(rs, "after", scores_by_tid, mode)

            weighted_before[mode] += int(is_correct_prediction(pred_b, gold, answer_mode=answer_mode))
            weighted_after[mode] += int(is_correct_prediction(pred_a, gold, answer_mode=answer_mode))

        # Error decomposition.
        if (not before_maj_ok) and any_before_ok:
            counts["baseline_selection_error"] += 1
        if not any_before_ok:
            counts["baseline_generation_error"] += 1

        if (not before_maj_ok) and before_weighted_tail5_ok:
            counts["weighted_fixes_majority"] += 1
        if before_maj_ok and (not before_weighted_tail5_ok):
            counts["weighted_breaks_majority"] += 1

        if (not before_weighted_tail5_ok) and after_weighted_tail5_ok:
            counts["repair_fixes_weighted"] += 1
        if before_weighted_tail5_ok and (not after_weighted_tail5_ok):
            counts["repair_breaks_weighted"] += 1

        if (not after_weighted_tail5_ok) and any_after_ok:
            counts["remaining_selection_error_after"] += 1
        if not any_after_ok:
            counts["remaining_generation_error_after"] += 1

        details.append({
            "sample_id": sid,
            "gold": gold,
            "answers_before": [x["before"] for x in rs],
            "answers_after": [x["after"] for x in rs],
            "first_before_ok": first_before_ok,
            "first_after_ok": first_after_ok,
            "majority_before": before_maj,
            "majority_after": after_maj,
            "majority_before_ok": before_maj_ok,
            "majority_after_ok": after_maj_ok,
            "weighted_tail5_before": before_weighted_tail5,
            "weighted_tail5_after": after_weighted_tail5,
            "weighted_tail5_before_ok": before_weighted_tail5_ok,
            "weighted_tail5_after_ok": after_weighted_tail5_ok,
            "any_before_ok": any_before_ok,
            "any_after_ok": any_after_ok,
        })

    def acc(key: str) -> float:
        return counts[key] / n if n else 0.0

    summary.update({
        "first_before_acc": acc("first_before_ok"),
        "first_after_acc": acc("first_after_ok"),
        "majority_before_acc": acc("majority_before_ok"),
        "majority_after_acc": acc("majority_after_ok"),
        "any_before_acc": acc("any_before_ok"),
        "any_after_acc": acc("any_after_ok"),
    })

    for mode in modes:
        summary[f"pce_top1_{mode}_acc"] = pce_top1[mode] / n if n else 0.0
        summary[f"weighted_{mode}_before_acc"] = weighted_before[mode] / n if n else 0.0
        summary[f"weighted_{mode}_after_acc"] = weighted_after[mode] / n if n else 0.0

    for k in [
        "baseline_selection_error",
        "baseline_generation_error",
        "weighted_fixes_majority",
        "weighted_breaks_majority",
        "repair_fixes_weighted",
        "repair_breaks_weighted",
        "remaining_selection_error_after",
        "remaining_generation_error_after",
    ]:
        summary[k] = counts[k]
        summary[f"{k}_rate"] = counts[k] / n if n else 0.0

    if args.out:
        write_json(args.out, summary)

    if args.details_out:
        p = Path(args.details_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for r in details:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    print("\n| Method | Accuracy |")
    print("|---|---:|")
    print(f"| first_before | {summary['first_before_acc']:.4f} |")
    print(f"| majority_before | {summary['majority_before_acc']:.4f} |")
    print(f"| weighted_tail5_before | {summary['weighted_tail5_before_acc']:.4f} |")
    print(f"| majority_after | {summary['majority_after_acc']:.4f} |")
    print(f"| weighted_tail5_after | {summary['weighted_tail5_after_acc']:.4f} |")
    print(f"| any_after | {summary['any_after_acc']:.4f} |")

    print("\n| Error type | Count | Rate |")
    print("|---|---:|---:|")
    for k in [
        "baseline_selection_error",
        "baseline_generation_error",
        "weighted_fixes_majority",
        "weighted_breaks_majority",
        "repair_fixes_weighted",
        "repair_breaks_weighted",
        "remaining_selection_error_after",
        "remaining_generation_error_after",
    ]:
        print(f"| {k} | {summary[k]} | {summary[f'{k}_rate']:.4f} |")


if __name__ == "__main__":
    main()
