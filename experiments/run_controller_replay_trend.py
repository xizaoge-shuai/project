from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from collections import defaultdict
from statistics import mean
from typing import Any, Dict, List, Optional, Set

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.io import read_jsonl
from utils.tokenizer_utils import count_tokens
from utils.eval_utils import is_correct_prediction


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


def norm_answer(x: str) -> str:
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


def answer_mode_for_dataset(dataset: str) -> str:
    dataset = dataset.lower()
    if dataset == "gsm8k":
        return "numeric"
    if dataset == "strategyqa":
        return "yesno"
    return "span"


def extract_answer_from_steps(steps: List[str]) -> str:
    for s in reversed(steps):
        if "Answer:" in s:
            return s.split("Answer:", 1)[-1].strip()
    return "\n".join(steps).strip()


def get_final_answer(pred_rows: List[Dict[str, Any]], traj_row: Optional[Dict[str, Any]]) -> str:
    if traj_row is not None and traj_row.get("steps"):
        return extract_answer_from_steps(traj_row.get("steps", []))

    fr = pred_rows[-1]
    for k in ["final_answer", "current_answer", "answer", "prediction"]:
        v = fr.get(k, "")
        if v is not None and str(v).strip():
            return str(v)

    return str(fr.get("prefix_text", ""))


def get_gold_answer(pred_rows: List[Dict[str, Any]], traj_row: Optional[Dict[str, Any]]) -> str:
    if traj_row is not None:
        for k in ["gold_answer", "gold", "target"]:
            v = traj_row.get(k, "")
            if v is not None and str(v).strip():
                return str(v)

    fr = pred_rows[-1]
    for k in ["gold_answer", "gold", "target"]:
        v = fr.get(k, "")
        if v is not None and str(v).strip():
            return str(v)

    return ""


def sort_pred_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key_fn(r: Dict[str, Any]):
        try:
            p = float(r.get("prefix_progress", 0.0))
        except Exception:
            p = 0.0
        try:
            n = int(r.get("prefix_num_units", 0))
        except Exception:
            n = 0
        return p, n

    return sorted(rows, key=key_fn)


def get_progress(r: Dict[str, Any]) -> float:
    try:
        return float(r.get("prefix_progress", 0.0))
    except Exception:
        return 0.0


def get_prob(r: Dict[str, Any]) -> float:
    try:
        return float(r.get("success_prob", 0.0))
    except Exception:
        return 0.0


def total_tokens_from_traj(traj_row: Optional[Dict[str, Any]], fallback_rows: List[Dict[str, Any]]) -> float:
    if traj_row is not None and traj_row.get("steps"):
        return float(sum(count_tokens(str(s)) for s in traj_row.get("steps", [])))

    fr = fallback_rows[-1]
    for k in ["tokens_total", "total_tokens"]:
        if k in fr:
            try:
                return float(fr[k])
            except Exception:
                pass

    txt = str(fr.get("prefix_text", ""))
    return float(count_tokens(txt))


def load_restrict_ids(path: Optional[str]) -> Optional[Set[str]]:
    if not path:
        return None

    rows = read_jsonl(path)
    ids: Set[str] = set()
    for r in rows:
        tid = str(r.get("trajectory_id", "")).strip()
        if tid:
            ids.add(tid)
    return ids


def should_trigger(
    probs: List[float],
    progresses: List[float],
    tau: float,
    mode: str,
    k: int,
    drop: float,
    gamma: float,
) -> bool:
    p = probs[-1]
    progress = progresses[-1]

    if mode == "static":
        return p < tau

    if mode == "decay":
        tau_eff = tau + gamma * progress
        return p < tau_eff

    if mode == "streak":
        if len(probs) < k:
            return False
        return all(x < tau for x in probs[-k:])

    if mode == "drop":
        if len(probs) < k:
            return False
        return (p < tau) and ((probs[-k] - p) >= drop)

    if mode == "streak_or_drop":
        if len(probs) < k:
            return False
        streak_hit = all(x < tau for x in probs[-k:])
        drop_hit = (p < tau) and ((probs[-k] - p) >= drop)
        return streak_hit or drop_hit

    raise ValueError(f"Unsupported mode: {mode}")


def safe_mean(xs: List[float]):
    xs = [x for x in xs if x is not None]
    return mean(xs) if xs else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--dataset", default="gsm8k")
    parser.add_argument("--tau", type=float, required=True)
    parser.add_argument(
        "--mode",
        default="static",
        choices=["static", "streak", "drop", "streak_or_drop", "decay"],
    )
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--drop", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--min_progress", type=float, default=0.0)
    parser.add_argument("--name", default="")
    parser.add_argument("--restrict_ids_from", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--details_out", default=None)
    args = parser.parse_args()

    pred_rows = read_jsonl(args.predictions)
    traj_rows = read_jsonl(args.trajectories)
    restrict_ids = load_restrict_ids(args.restrict_ids_from)

    pred_by_tid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in pred_rows:
        tid = str(r.get("trajectory_id", "")).strip()
        if not tid:
            continue
        if restrict_ids is not None and tid not in restrict_ids:
            continue
        pred_by_tid[tid].append(r)

    traj_by_tid: Dict[str, Dict[str, Any]] = {}
    for r in traj_rows:
        tid = str(r.get("trajectory_id", "")).strip()
        if tid:
            traj_by_tid[tid] = r

    details: List[Dict[str, Any]] = []

    for tid, rs in pred_by_tid.items():
        rs = sort_pred_rows(rs)
        if not rs:
            continue

        traj = traj_by_tid.get(tid)

        final_answer = get_final_answer(rs, traj)
        gold_answer = get_gold_answer(rs, traj)

        is_correct = int(
            is_correct_prediction(
                norm_answer(final_answer),
                norm_answer(gold_answer),
                answer_mode=answer_mode_for_dataset(args.dataset),
            )
        )
        is_bad = not bool(is_correct)

        tokens_total = total_tokens_from_traj(traj, rs)

        probs: List[float] = []
        progresses: List[float] = []

        triggered = False
        trigger_row: Optional[Dict[str, Any]] = None

        for r in rs:
            progress = get_progress(r)
            prob = get_prob(r)

            probs.append(prob)
            progresses.append(progress)

            if progress < args.min_progress:
                continue

            if should_trigger(
                probs=probs,
                progresses=progresses,
                tau=args.tau,
                mode=args.mode,
                k=args.k,
                drop=args.drop,
                gamma=args.gamma,
            ):
                triggered = True
                trigger_row = r
                break

        if triggered and trigger_row is not None:
            trigger_progress = get_progress(trigger_row)
            trigger_prob = get_prob(trigger_row)
            tokens_saved = max(0.0, tokens_total * (1.0 - trigger_progress))
            saved_ratio = tokens_saved / max(1.0, tokens_total)
        else:
            trigger_progress = None
            trigger_prob = None
            tokens_saved = 0.0
            saved_ratio = 0.0

        details.append(
            {
                "trajectory_id": tid,
                "sample_id": rs[-1].get("sample_id", ""),
                "is_correct": bool(is_correct),
                "is_bad_trajectory": bool(is_bad),
                "triggered": bool(triggered),
                "first_trigger_progress": trigger_progress,
                "first_trigger_prob": trigger_prob,
                "tokens_total": tokens_total,
                "tokens_saved": tokens_saved,
                "tokens_saved_ratio": saved_ratio,
                "final_answer": final_answer,
                "gold_answer": gold_answer,
            }
        )

    n_total = len(details)
    bad = [d for d in details if d["is_bad_trajectory"]]
    good = [d for d in details if not d["is_bad_trajectory"]]
    trig = [d for d in details if d["triggered"]]
    trig_bad = [d for d in trig if d["is_bad_trajectory"]]
    trig_good = [d for d in trig if not d["is_bad_trajectory"]]

    summary = {
        "setting": args.name or f"{args.mode}, tau={args.tau}",
        "prediction_file": args.predictions,
        "trajectory_file": args.trajectories,
        "dataset": args.dataset,
        "tau": args.tau,
        "mode": args.mode,
        "k": args.k,
        "drop": args.drop,
        "gamma": args.gamma,
        "min_progress": args.min_progress,
        "restrict_ids_from": args.restrict_ids_from,
        "n_total_trajectories": n_total,
        "n_bad_trajectories": len(bad),
        "n_good_trajectories": len(good),
        "num_interrupted_all": len(trig),
        "num_interrupted_bad": len(trig_bad),
        "num_interrupted_good": len(trig_good),
        "interrupt_rate_all": len(trig) / max(1, n_total),
        "interrupt_rate_bad": len(trig_bad) / max(1, len(bad)),
        "interrupt_rate_good": len(trig_good) / max(1, len(good)),
        "interrupt_precision_traj": len(trig_bad) / max(1, len(trig)),
        "avg_first_trigger_progress_all": safe_mean(
            [d["first_trigger_progress"] for d in trig]
        ),
        "avg_first_trigger_progress_bad": safe_mean(
            [d["first_trigger_progress"] for d in trig_bad]
        ),
        "avg_first_trigger_progress_good": safe_mean(
            [d["first_trigger_progress"] for d in trig_good]
        ),
        "avg_saved_tokens_all": safe_mean([d["tokens_saved"] for d in trig]),
        "avg_saved_tokens_bad": safe_mean([d["tokens_saved"] for d in trig_bad]),
        "avg_saved_tokens_good": safe_mean([d["tokens_saved"] for d in trig_good]),
        "avg_saved_token_ratio_all": safe_mean([d["tokens_saved_ratio"] for d in trig]),
        "avg_saved_token_ratio_bad": safe_mean([d["tokens_saved_ratio"] for d in trig_bad]),
        "avg_saved_token_ratio_good": safe_mean([d["tokens_saved_ratio"] for d in trig_good]),
        "details_preview": details[:10],
    }

    write_json(args.out, summary)

    if args.details_out:
        write_jsonl(args.details_out, details)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
