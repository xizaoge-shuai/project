from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


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
    x = re.sub(r"^(the answer is)\s+", "", x, flags=re.IGNORECASE)
    x = re.sub(r"^(therefore|thus|so)\s*,?\s*", "", x, flags=re.IGNORECASE)

    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", x)
    if nums:
        return nums[-1]

    return x.strip()


def normalize_yesno(answer: str) -> str:
    x = clean_text(answer).lower()
    if "yes" in x:
        return "yes"
    if "no" in x:
        return "no"
    return x


def normalize_text_answer(answer: str) -> str:
    x = clean_text(answer).lower()
    x = re.sub(r"[^\w\s]", "", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def answers_match(dataset: str, pred_answer: str, gold_answer: str) -> bool:
    dataset = dataset.lower()

    if dataset == "gsm8k":
        return normalize_math(pred_answer) == normalize_math(gold_answer)

    if dataset == "strategyqa":
        return normalize_yesno(pred_answer) == normalize_yesno(gold_answer)

    if dataset == "hotpotqa":
        return normalize_text_answer(pred_answer) == normalize_text_answer(gold_answer)

    return normalize_text_answer(pred_answer) == normalize_text_answer(gold_answer)


# =========================
# Helpers
# =========================

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")


def approx_token_count_from_text(text: str) -> int:
    text = text or ""
    return len(TOKEN_PATTERN.findall(text))


def infer_total_tokens(row: Dict[str, Any]) -> float:
    """
    优先读取显式 token 字段；
    否则从 trajectory_text / steps 近似估计。
    """
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

    # 实在没有就返回 0
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
# Main replay logic
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument(
        "--dataset", default="gsm8k", choices=["gsm8k", "strategyqa", "hotpotqa"]
    )
    parser.add_argument("--tau", type=float, required=True)
    parser.add_argument("--name", default="setting")
    parser.add_argument(
        "--restrict_ids_from",
        default="",
        help="可选：用另一个 predictions 文件的 trajectory_id 子集来限制公平比较",
    )
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    restrict_ids = (
        extract_restrict_ids(args.restrict_ids_from) if args.restrict_ids_from else None
    )

    traj_meta = build_trajectory_meta(
        trajectories_path=args.trajectories,
        dataset=args.dataset,
        restrict_ids=restrict_ids,
    )
    pred_groups = build_prediction_groups(
        predictions_path=args.predictions,
        restrict_ids=restrict_ids,
    )

    common_ids = sorted(set(traj_meta.keys()) & set(pred_groups.keys()))
    if not common_ids:
        raise ValueError(
            "No common trajectory_ids between predictions and trajectories."
        )

    n_total = len(common_ids)
    n_bad = 0
    n_good = 0

    interrupted = 0
    interrupted_bad = 0
    interrupted_good = 0

    first_trigger_progress_all: List[float] = []
    first_trigger_progress_bad: List[float] = []
    first_trigger_progress_good: List[float] = []

    saved_tokens_all: List[float] = []
    saved_tokens_bad: List[float] = []
    saved_tokens_good: List[float] = []

    saved_ratio_all: List[float] = []
    saved_ratio_bad: List[float] = []
    saved_ratio_good: List[float] = []

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

        hit_progress = None
        hit_prob = None

        for r in rows:
            p = float(r["success_prob"])
            if p < args.tau:
                hit_progress = float(r.get("prefix_progress", 1.0))
                hit_prob = p
                break

        interrupted_this = hit_progress is not None
        saved_ratio = max(0.0, 1.0 - hit_progress) if interrupted_this else 0.0
        saved_tokens = total_tokens * saved_ratio if interrupted_this else 0.0

        if interrupted_this:
            interrupted += 1

            first_trigger_progress_all.append(hit_progress)
            saved_tokens_all.append(saved_tokens)
            saved_ratio_all.append(saved_ratio)

            if is_bad:
                interrupted_bad += 1
                first_trigger_progress_bad.append(hit_progress)
                saved_tokens_bad.append(saved_tokens)
                saved_ratio_bad.append(saved_ratio)
            else:
                interrupted_good += 1
                first_trigger_progress_good.append(hit_progress)
                saved_tokens_good.append(saved_tokens)
                saved_ratio_good.append(saved_ratio)

        details.append(
            {
                "trajectory_id": tid,
                "sample_id": meta["sample_id"],
                "is_bad_trajectory": is_bad,
                "interrupted": interrupted_this,
                "first_trigger_progress": hit_progress,
                "first_trigger_prob": hit_prob,
                "tokens_total": total_tokens,
                "tokens_saved": saved_tokens,
                "tokens_saved_ratio": saved_ratio,
            }
        )

    result = {
        "setting": args.name,
        "tau": args.tau,
        "n_total_trajectories": n_total,
        "n_bad_trajectories": n_bad,
        "n_good_trajectories": n_good,
        "num_interrupted_all": interrupted,
        "num_interrupted_bad": interrupted_bad,
        "num_interrupted_good": interrupted_good,
        "interrupt_rate_all": interrupted / n_total if n_total > 0 else None,
        "interrupt_rate_bad": interrupted_bad / n_bad if n_bad > 0 else None,
        "interrupt_rate_good": interrupted_good / n_good if n_good > 0 else None,
        "interrupt_precision_traj": (
            interrupted_bad / interrupted if interrupted > 0 else None
        ),
        "avg_first_trigger_progress_all": mean_or_none(first_trigger_progress_all),
        "avg_first_trigger_progress_bad": mean_or_none(first_trigger_progress_bad),
        "avg_first_trigger_progress_good": mean_or_none(first_trigger_progress_good),
        "avg_saved_tokens_all": mean_or_none(saved_tokens_all),
        "avg_saved_tokens_bad": mean_or_none(saved_tokens_bad),
        "avg_saved_tokens_good": mean_or_none(saved_tokens_good),
        "avg_saved_token_ratio_all": mean_or_none(saved_ratio_all),
        "avg_saved_token_ratio_bad": mean_or_none(saved_ratio_bad),
        "avg_saved_token_ratio_good": mean_or_none(saved_ratio_good),
        "details_preview": details[:10],
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.out:
        write_json(args.out, result)


if __name__ == "__main__":
    main()
