from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# =========================
# IO
# =========================


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_json(path: str, obj: Dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# =========================
# Answer matching
# =========================


def clean_text(x: str) -> str:
    x = str(x).strip()
    x = x.replace("\u00a0", " ")
    x = re.sub(r"\s+", " ", x)
    return x


def normalize_math(answer: str) -> str:
    x = clean_text(answer)

    m = re.search(r"####\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)", x)
    if m:
        return m.group(1).replace(",", "")

    x = x.strip("`$ ")
    x = x.replace(",", "")
    x = x.strip(" .,:;!?")
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", x)
    if nums:
        return nums[-1]
    return x.strip()


def answers_match(dataset: str, pred_answer: str, gold_answer: str) -> bool:
    dataset = dataset.lower()
    if dataset == "gsm8k":
        return normalize_math(pred_answer) == normalize_math(gold_answer)
    return clean_text(pred_answer).lower() == clean_text(gold_answer).lower()


# =========================
# Helpers
# =========================

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")


def approx_token_count_from_text(text: str) -> int:
    return len(TOKEN_PATTERN.findall(text or ""))


def infer_total_tokens(row: Dict[str, Any]) -> float:
    for key in ["tokens", "n_tokens", "output_tokens", "num_tokens"]:
        val = row.get(key, None)
        if val is not None:
            try:
                val = float(val)
                if val > 0:
                    return val
            except Exception:
                pass

    if "trajectory_text" in row and row["trajectory_text"]:
        n = approx_token_count_from_text(str(row["trajectory_text"]))
        if n > 0:
            return float(n)

    if "steps" in row and row["steps"]:
        text = "\n".join(str(x) for x in row["steps"])
        n = approx_token_count_from_text(text)
        if n > 0:
            return float(n)

    return 0.0


def mean_or_none(xs: List[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def extract_restrict_ids(path: Optional[str]) -> Optional[Set[str]]:
    if not path:
        return None
    rows = read_jsonl(path)
    ids = set()
    for r in rows:
        tid = r.get("trajectory_id", "")
        if tid:
            ids.add(tid)
    return ids


# =========================
# Build meta
# =========================


def build_trajectory_meta(
    trajectories_path: str,
    dataset: str,
    restrict_ids: Optional[Set[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    rows = read_jsonl(trajectories_path)
    out: Dict[str, Dict[str, Any]] = {}

    for r in rows:
        tid = r.get("trajectory_id", "")
        if not tid:
            continue
        if restrict_ids is not None and tid not in restrict_ids:
            continue

        gold = str(r.get("gold_answer", ""))
        pred = str(r.get("final_answer", ""))
        is_correct = answers_match(dataset, pred, gold)
        is_bad = not is_correct

        out[tid] = {
            "trajectory_id": tid,
            "sample_id": r.get("sample_id", ""),
            "tokens": infer_total_tokens(r),
            "gold_answer": gold,
            "final_answer": pred,
            "is_bad_trajectory": is_bad,
        }

    return out


def build_prediction_groups(
    predictions_path: str,
    restrict_ids: Optional[Set[str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    rows = read_jsonl(predictions_path)
    groups: Dict[str, List[Dict[str, Any]]] = {}

    for r in rows:
        tid = r.get("trajectory_id", "")
        if not tid:
            continue
        if restrict_ids is not None and tid not in restrict_ids:
            continue
        groups.setdefault(tid, []).append(r)

    for tid in groups:
        groups[tid] = sorted(
            groups[tid],
            key=lambda x: float(
                x.get("prefix_progress", x.get("prefix_num_units", 1.0))
            ),
        )

    return groups


def build_repair_index(
    repair_labels_path: str,
    restrict_ids: Optional[Set[str]] = None,
) -> Tuple[Dict[str, int], Dict[Tuple[str, float], int]]:
    rows = read_jsonl(repair_labels_path)

    by_prefix_id: Dict[str, int] = {}
    by_traj_progress: Dict[Tuple[str, float], int] = {}

    for r in rows:
        tid = r.get("trajectory_id", "")
        if restrict_ids is not None and tid and tid not in restrict_ids:
            continue

        repairable = int(r.get("repairable", 0))

        prefix_id = r.get("prefix_id", "")
        if prefix_id:
            by_prefix_id[prefix_id] = repairable

        progress = round(float(r.get("prefix_progress", 0.0) or 0.0), 6)
        if tid:
            by_traj_progress[(tid, progress)] = repairable

    return by_prefix_id, by_traj_progress


def lookup_repairable(
    row: Dict[str, Any],
    by_prefix_id: Dict[str, int],
    by_traj_progress: Dict[Tuple[str, float], int],
) -> int:
    prefix_id = row.get("prefix_id", "")
    if prefix_id and prefix_id in by_prefix_id:
        return by_prefix_id[prefix_id]

    tid = row.get("trajectory_id", "")
    progress = round(float(row.get("prefix_progress", 0.0) or 0.0), 6)
    return int(by_traj_progress.get((tid, progress), 0))


# =========================
# Main
# =========================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--repair_labels", required=True)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--dataset", default="gsm8k")
    parser.add_argument("--tau_trigger", type=float, required=True)
    parser.add_argument("--tau_recover", type=float, default=-1.0)
    parser.add_argument("--lookahead_prefixes", type=int, default=2)
    parser.add_argument(
        "--delta_recover",
        type=float,
        default=0.05,
        help="恢复时要求 best_prob - p_trigger >= delta_recover",
    )
    parser.add_argument(
        "--require_two_consecutive_lows",
        action="store_true",
        help="只有连续两个低分 prefix 才触发控制",
    )
    parser.add_argument("--name", default="setting")
    parser.add_argument("--restrict_ids_from", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    if args.tau_recover < 0:
        args.tau_recover = args.tau_trigger

    restrict_ids = (
        extract_restrict_ids(args.restrict_ids_from) if args.restrict_ids_from else None
    )

    traj_meta = build_trajectory_meta(args.trajectories, args.dataset, restrict_ids)
    pred_groups = build_prediction_groups(args.predictions, restrict_ids)
    by_prefix_id, by_traj_progress = build_repair_index(
        args.repair_labels, restrict_ids
    )

    common_ids = sorted(set(traj_meta.keys()) & set(pred_groups.keys()))
    if not common_ids:
        raise ValueError(
            "No common trajectory_ids between predictions and trajectories."
        )

    n_total = len(common_ids)
    n_bad = 0
    n_good = 0

    triggered = 0
    backtracked = 0
    pruned_directly = 0

    recovered = 0
    recovered_bad = 0
    recovered_good = 0

    interrupted = 0
    interrupted_bad = 0
    interrupted_good = 0

    saved_tokens_all: List[float] = []
    saved_tokens_bad: List[float] = []
    saved_tokens_good: List[float] = []

    details: List[Dict[str, Any]] = []

    for tid in common_ids:
        meta = traj_meta[tid]
        rows = pred_groups[tid]

        is_bad = bool(meta["is_bad_trajectory"])
        total_tokens = float(meta["tokens"])

        if is_bad:
            n_bad += 1
        else:
            n_good += 1

        triggered_this = False
        backtracked_this = False
        recovered_this = False
        interrupted_this = False
        direct_prune_this = False
        trigger_progress = None
        trigger_prob = None
        best_recover_prob = None
        saved_tokens = 0.0

        i = 0
        while i < len(rows):
            r = rows[i]
            p = float(r["success_prob"])

            if p >= args.tau_trigger:
                i += 1
                continue

            if args.require_two_consecutive_lows:
                if i + 1 >= len(rows):
                    i += 1
                    continue
                p_next_low = float(rows[i + 1]["success_prob"])
                if p_next_low >= args.tau_trigger:
                    i += 1
                    continue

            triggered += 1
            triggered_this = True
            trigger_progress = float(r.get("prefix_progress", 1.0))
            trigger_prob = p

            repairable = lookup_repairable(r, by_prefix_id, by_traj_progress)

            if repairable == 1:
                backtracked += 1
                backtracked_this = True

                start_j = i + 2 if args.require_two_consecutive_lows else i + 1
                upper = min(len(rows), start_j + args.lookahead_prefixes)

                future_probs = [
                    float(rows[j]["success_prob"]) for j in range(start_j, upper)
                ]

                if future_probs:
                    best_recover_prob = max(future_probs)
                    recovered_candidate = (best_recover_prob >= args.tau_recover) or (
                        (best_recover_prob - trigger_prob) >= args.delta_recover
                    )
                else:
                    recovered_candidate = False

                if recovered_candidate:
                    recovered += 1
                    recovered_this = True
                    if is_bad:
                        recovered_bad += 1
                    else:
                        recovered_good += 1
                    break

            # 不可修复，或者可修复但恢复失败 -> prune
            interrupted += 1
            interrupted_this = True
            if repairable == 0:
                pruned_directly += 1
                direct_prune_this = True

            saved_ratio = max(0.0, 1.0 - trigger_progress)
            saved_tokens = total_tokens * saved_ratio
            saved_tokens_all.append(saved_tokens)

            if is_bad:
                interrupted_bad += 1
                saved_tokens_bad.append(saved_tokens)
            else:
                interrupted_good += 1
                saved_tokens_good.append(saved_tokens)
            break

        details.append(
            {
                "trajectory_id": tid,
                "sample_id": meta["sample_id"],
                "is_bad_trajectory": is_bad,
                "triggered": triggered_this,
                "backtracked": backtracked_this,
                "recovered": recovered_this,
                "interrupted": interrupted_this,
                "direct_prune": direct_prune_this,
                "trigger_progress": trigger_progress,
                "trigger_prob": trigger_prob,
                "best_recover_prob": best_recover_prob,
                "tokens_total": total_tokens,
                "tokens_saved": saved_tokens,
            }
        )

    result = {
        "setting": args.name,
        "tau_trigger": args.tau_trigger,
        "tau_recover": args.tau_recover,
        "lookahead_prefixes": args.lookahead_prefixes,
        "delta_recover": args.delta_recover,
        "require_two_consecutive_lows": args.require_two_consecutive_lows,
        "n_total_trajectories": n_total,
        "n_bad_trajectories": n_bad,
        "n_good_trajectories": n_good,
        "num_triggered": triggered,
        "num_backtracked": backtracked,
        "num_pruned_directly": pruned_directly,
        "num_recovered": recovered,
        "num_recovered_bad": recovered_bad,
        "num_recovered_good": recovered_good,
        "recover_rate_all": recovered / backtracked if backtracked > 0 else None,
        "recover_rate_bad": recovered_bad / n_bad if n_bad > 0 else None,
        "recover_rate_good": recovered_good / n_good if n_good > 0 else None,
        "num_interrupted": interrupted,
        "num_interrupted_bad": interrupted_bad,
        "num_interrupted_good": interrupted_good,
        "interrupt_rate_bad": interrupted_bad / n_bad if n_bad > 0 else None,
        "interrupt_rate_good": interrupted_good / n_good if n_good > 0 else None,
        "interrupt_precision_traj": (
            interrupted_bad / interrupted if interrupted > 0 else None
        ),
        "avg_saved_tokens_all": mean_or_none(saved_tokens_all),
        "avg_saved_tokens_bad": mean_or_none(saved_tokens_bad),
        "avg_saved_tokens_good": mean_or_none(saved_tokens_good),
        "details_preview": details[:10],
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.out:
        write_json(args.out, result)


if __name__ == "__main__":
    main()
