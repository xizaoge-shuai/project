from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.eval_utils import is_correct_prediction


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: str, obj: Dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def norm_answer(x: str) -> str:
    """
    GSM8K numeric answer normalization.
    """
    x = str(x or "").strip()
    x = x.replace(",", "")
    x = x.replace("$", "")
    nums = re.findall(r"-?\d+(?:\.\d+)?", x)
    if nums:
        v = nums[-1]
        if "." in v:
            v = v.rstrip("0").rstrip(".")
        return v
    return x.lower().strip()


def answer_mode(dataset: str) -> str:
    if dataset == "gsm8k":
        return "numeric"
    if dataset == "strategyqa":
        return "yesno"
    return "span"


def get_score(row: Dict[str, Any]) -> float:
    for k in ["success_prob", "score", "prob"]:
        if k in row:
            try:
                return float(row[k])
            except Exception:
                pass
    return 0.0


def get_answer(row: Dict[str, Any]) -> str:
    """
    优先使用 final_answer。
    如果没有，则退化到 current_answer / answer / prefix_text。
    """
    for k in ["final_answer", "current_answer", "answer", "prediction"]:
        v = row.get(k, "")
        if v is not None and str(v).strip():
            return str(v)
    return str(row.get("prefix_text", ""))


def get_gold(row: Dict[str, Any]) -> str:
    for k in ["gold_answer", "gold", "target"]:
        v = row.get(k, "")
        if v is not None and str(v).strip():
            return str(v)
    return ""


def pick_final_row(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    从同一个 trajectory 的多条 prefix prediction 中选最后一个 prefix。
    """
    def key_fn(r: Dict[str, Any]) -> Tuple[float, int]:
        try:
            progress = float(r.get("prefix_progress", 0.0))
        except Exception:
            progress = 0.0
        try:
            n_units = int(r.get("prefix_num_units", 0))
        except Exception:
            n_units = 0
        return progress, n_units

    return sorted(rows, key=key_fn)[-1]


def choose_majority(trs: List[Dict[str, Any]]) -> str:
    """
    多数投票。
    平票时，用该答案组的 score 总和打破平局。
    """
    cnt = Counter(t["norm_answer"] for t in trs)
    score_sum = defaultdict(float)
    for t in trs:
        score_sum[t["norm_answer"]] += float(t["score"])

    return sorted(
        cnt.keys(),
        key=lambda a: (cnt[a], score_sum[a]),
        reverse=True,
    )[0]


def choose_weighted(trs: List[Dict[str, Any]]) -> str:
    """
    PCE-weighted vote。
    """
    score_sum = defaultdict(float)
    cnt = Counter()
    for t in trs:
        score_sum[t["norm_answer"]] += float(t["score"])
        cnt[t["norm_answer"]] += 1

    return sorted(
        score_sum.keys(),
        key=lambda a: (score_sum[a], cnt[a]),
        reverse=True,
    )[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "strategyqa", "hotpotqa"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--details_out", default=None)
    args = parser.parse_args()

    rows = read_jsonl(args.predictions)

    # trajectory-level aggregation
    by_traj: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        tid = str(r.get("trajectory_id", ""))
        if tid:
            by_traj[tid].append(r)

    traj_rows: List[Dict[str, Any]] = []
    for tid, rs in by_traj.items():
        fr = pick_final_row(rs)

        ans = get_answer(fr)
        gold = get_gold(fr)
        score = get_score(fr)

        ans_norm = norm_answer(ans)
        gold_norm = norm_answer(gold)

        correct = int(
            is_correct_prediction(
                ans_norm,
                gold_norm,
                answer_mode=answer_mode(args.dataset),
            )
        )

        traj_rows.append(
            {
                "sample_id": fr.get("sample_id", ""),
                "trajectory_id": tid,
                "score": float(score),
                "answer": ans,
                "norm_answer": ans_norm,
                "gold_answer": gold,
                "norm_gold": gold_norm,
                "is_correct": correct,
                "prefix_progress": fr.get("prefix_progress", None),
                "prefix_num_units": fr.get("prefix_num_units", None),
            }
        )

    # sample-level aggregation
    by_sample: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in traj_rows:
        sid = str(r.get("sample_id", ""))
        if sid:
            by_sample[sid].append(r)

    n = len(by_sample)

    correct_first = 0
    correct_pce_top1 = 0
    correct_majority = 0
    correct_weighted = 0
    oracle_any = 0

    details: List[Dict[str, Any]] = []

    for sid, trs in by_sample.items():
        gold = trs[0]["gold_answer"]

        # first trajectory baseline
        first = trs[0]
        correct_first += int(first["is_correct"])

        # PCE top1
        top = max(trs, key=lambda x: x["score"])
        correct_pce_top1 += int(top["is_correct"])

        # Majority vote
        maj_ans = choose_majority(trs)
        maj_correct = int(
            is_correct_prediction(
                maj_ans,
                gold,
                answer_mode=answer_mode(args.dataset),
            )
        )
        correct_majority += maj_correct

        # Weighted vote
        w_ans = choose_weighted(trs)
        w_correct = int(
            is_correct_prediction(
                w_ans,
                gold,
                answer_mode=answer_mode(args.dataset),
            )
        )
        correct_weighted += w_correct

        # Correct oracle: any trajectory final answer correct
        any_correct = int(any(int(t["is_correct"]) == 1 for t in trs))
        oracle_any += any_correct

        details.append(
            {
                "sample_id": sid,
                "n_trajs": len(trs),
                "gold_answer": gold,
                "first_answer": first["answer"],
                "first_correct": int(first["is_correct"]),
                "pce_top1_answer": top["answer"],
                "pce_top1_score": top["score"],
                "pce_top1_correct": int(top["is_correct"]),
                "majority_answer": maj_ans,
                "majority_correct": maj_correct,
                "weighted_answer": w_ans,
                "weighted_correct": w_correct,
                "oracle_any_correct": any_correct,
                "trajectory_answers": [
                    {
                        "trajectory_id": t["trajectory_id"],
                        "answer": t["answer"],
                        "norm_answer": t["norm_answer"],
                        "score": t["score"],
                        "is_correct": t["is_correct"],
                    }
                    for t in trs
                ],
            }
        )

    results = {
        "prediction_file": args.predictions,
        "dataset": args.dataset,
        "n_samples": n,
        "n_trajectories": len(traj_rows),
        "avg_trajs_per_sample": (
            sum(len(v) for v in by_sample.values()) / max(1, n)
        ),
        "first_traj_acc": correct_first / max(1, n),
        "pce_top1_acc": correct_pce_top1 / max(1, n),
        "majority_vote_acc": correct_majority / max(1, n),
        "weighted_vote_acc": correct_weighted / max(1, n),
        "oracle_any_correct_acc": oracle_any / max(1, n),
    }

    write_json(args.out, results)

    if args.details_out:
        write_jsonl(args.details_out, details)

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
