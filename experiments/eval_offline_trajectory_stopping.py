#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.eval_cross_pce_weighted_selection import clean, ok


TID_KEYS = (
    "trajectory_id",
    "traj_id",
    "trajectory",
)

PREFIX_INDEX_KEYS = (
    "unit_idx",
    "atom_idx",
    "prefix_idx",
    "step_idx",
    "position",
    "prefix_order",
    "idx",
)

PROGRESS_KEYS = (
    "progress",
    "progress_ratio",
    "prefix_progress",
    "relative_progress",
)

PROB_KEYS = (
    "success_prob",
    "prob_success",
    "p_success",
    "success_probability",
    "score",
    "prob",
    "confidence",
)

PREFIX_TEXT_KEYS = (
    "prefix_text",
    "trajectory_prefix",
    "reasoning_prefix",
    "prefix",
    "text",
    "trajectory",
    "response",
    "generated_text",
    "reasoning",
)

FULL_TEXT_KEYS = (
    "trajectory",
    "text",
    "response",
    "generated_text",
    "reasoning",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)

    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_no}"
                ) from exc
    return rows


def first_nonempty(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def trajectory_id(row: dict[str, Any]) -> str:
    for key in ("trajectory_id", "traj_id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def sample_id(row: dict[str, Any]) -> str:
    for key in ("sample_id", "id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def trajectory_rank(tid: str) -> tuple[int, int, str]:
    traj_match = re.search(r"_traj_(\d+)", str(tid))
    seed_match = re.search(r"_seed(\d+)", str(tid))

    traj_no = int(traj_match.group(1)) if traj_match else 10**9
    seed_no = int(seed_match.group(1)) if seed_match else 10**9

    return traj_no, seed_no, str(tid)


def extract_numeric(row: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue

        try:
            number = float(value)
        except (TypeError, ValueError):
            continue

        if math.isfinite(number):
            return number

    # Allow common nested prediction dictionaries.
    for container_key in ("probabilities", "probs", "prediction", "scores"):
        nested = row.get(container_key)
        if not isinstance(nested, dict):
            continue

        for key in keys:
            value = nested.get(key)
            if value is None:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                return number

    return None


def extract_prefix_index(row: dict[str, Any]) -> float | None:
    return extract_numeric(row, PREFIX_INDEX_KEYS)


def extract_progress(row: dict[str, Any]) -> float | None:
    value = extract_numeric(row, PROGRESS_KEYS)
    if value is None:
        return None

    # Support progress expressed as percentages.
    if value > 1.0 and value <= 100.0:
        value /= 100.0

    return min(1.0, max(0.0, value))


def extract_prefix_text(row: dict[str, Any]) -> str:
    for key in PREFIX_TEXT_KEYS:
        value = row.get(key)

        if value is None:
            continue

        if isinstance(value, list):
            text = "\n".join(str(x) for x in value)
        elif isinstance(value, dict):
            text = json.dumps(value, ensure_ascii=False)
        else:
            text = str(value)

        if text.strip():
            return text

    steps = row.get("steps")
    if isinstance(steps, list):
        return "\n".join(str(x) for x in steps)

    return ""


def extract_full_text(row: dict[str, Any]) -> str:
    for key in FULL_TEXT_KEYS:
        value = row.get(key)
        if value is None:
            continue

        if isinstance(value, list):
            text = "\n".join(str(x) for x in value)
        else:
            text = str(value)

        if text.strip():
            return text

    steps = row.get("steps")
    if isinstance(steps, list):
        return "\n".join(str(x) for x in steps)

    return ""


def extract_full_answer(row: dict[str, Any]) -> str:
    # final_answer must precede answer because MathQA's answer field is gold.
    for key in (
        "final_answer",
        "predicted_answer",
        "model_answer",
    ):
        value = row.get(key)
        if value is not None and str(value).strip():
            return clean(value)

    # Use answer only as a final fallback for datasets where it stores
    # the generated answer.
    value = row.get("answer")
    if value is not None and str(value).strip():
        return clean(value)

    return ""


def extract_gold_answer(row: dict[str, Any]) -> str:
    value = row.get("gold_answer")
    if value is not None and str(value).strip():
        return clean(value)

    value = row.get("answer")
    if value is not None and str(value).strip():
        return clean(value)

    return ""


def extract_explicit_answer(text: str) -> str:
    """
    Extract only explicitly committed final answers.

    We deliberately do not use the last number in an arbitrary reasoning
    prefix because that would make offline stopping unrealistically optimistic.
    """
    text = str(text or "")
    candidates: list[tuple[int, str]] = []

    # \boxed{...}; supports one level of nested braces.
    boxed_pattern = re.compile(
        r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}",
        flags=re.I,
    )
    for match in boxed_pattern.finditer(text):
        candidates.append((match.start(), match.group(1).strip()))

    patterns = [
        r"Final\s+Answer\s*:\s*([^\n\r]+)",
        r"Final\s+Answer\s+is\s*[:\-]?\s*([^\n\r]+)",
        r"####\s*([^\n\r]+)",
        r"(?<!Final\s)Answer\s*:\s*([^\n\r]+)",
        r"(?:Chosen\s+)?(?:Option|Choice)\s*[:\-]?\s*"
        r"[\(\[]?([A-Ea-e])[\)\]]?",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.I):
            candidates.append((match.start(), match.group(1).strip()))

    if not candidates:
        return ""

    _, raw = max(candidates, key=lambda item: item[0])

    # Remove trailing commentary occasionally placed on the same line.
    raw = re.split(
        r"\s+(?:Therefore|Thus|Hence|This is|You have|You've)\b",
        raw,
        maxsplit=1,
        flags=re.I,
    )[0].strip()

    return clean(raw)


def stable_majority(answers: list[str]) -> str:
    answers = [str(x) for x in answers if str(x).strip()]
    if not answers:
        return ""

    counts = Counter(answers)
    highest = max(counts.values())

    # Stable tie break: first trajectory in deterministic trajectory order.
    for answer in answers:
        if counts[answer] == highest:
            return answer

    return answers[0]


def correctness(prediction: str, gold: str) -> int:
    if not str(prediction).strip() or not str(gold).strip():
        return 0

    try:
        return int(bool(ok(prediction, gold)))
    except Exception:
        return 0


def combine_prefix_predictions(
    prefix_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(prefix_rows) != len(prediction_rows):
        raise ValueError(
            "Prefix/prediction line counts differ: "
            f"{len(prefix_rows)} vs {len(prediction_rows)}. "
            "The full-test prediction file must correspond line-by-line "
            "to the full prefix file."
        )

    combined = []
    mismatched_ids = 0

    for line_idx, (prefix, prediction) in enumerate(
        zip(prefix_rows, prediction_rows),
        1,
    ):
        prefix_tid = trajectory_id(prefix)
        prediction_tid = trajectory_id(prediction)

        if (
            prefix_tid
            and prediction_tid
            and prefix_tid != prediction_tid
        ):
            mismatched_ids += 1

        tid = prefix_tid or prediction_tid
        if not tid:
            raise ValueError(
                f"Missing trajectory_id at aligned line {line_idx}"
            )

        probability = extract_numeric(prediction, PROB_KEYS)
        if probability is None:
            probability = extract_numeric(prefix, PROB_KEYS)

        if probability is None:
            raise ValueError(
                f"No success probability found at aligned line {line_idx}. "
                f"Prediction keys={sorted(prediction)}"
            )

        text = extract_prefix_text(prefix)
        if not text:
            text = extract_prefix_text(prediction)

        index = extract_prefix_index(prefix)
        if index is None:
            index = extract_prefix_index(prediction)

        progress = extract_progress(prefix)
        if progress is None:
            progress = extract_progress(prediction)

        combined.append({
            "line_idx": line_idx,
            "trajectory_id": tid,
            "sample_id": sample_id(prefix) or sample_id(prediction),
            "probability": float(probability),
            "prefix_index": index,
            "progress": progress,
            "prefix_text": text,
            "explicit_answer": extract_explicit_answer(text),
        })

    if mismatched_ids:
        raise ValueError(
            f"{mismatched_ids} aligned rows have mismatched trajectory IDs"
        )

    return combined


def token_lengths(
    texts: list[str],
    tokenizer_path: str,
    batch_size: int,
) -> tuple[list[int], str]:
    if not tokenizer_path:
        lengths = [
            len(re.findall(r"\S+", str(text)))
            for text in texts
        ]
        return lengths, "whitespace"

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            local_files_only=True,
        )
    except Exception as exc:
        print(
            f"[WARNING] tokenizer load failed: {exc}\n"
            "[WARNING] falling back to whitespace token counts",
            file=sys.stderr,
        )
        lengths = [
            len(re.findall(r"\S+", str(text)))
            for text in texts
        ]
        return lengths, "whitespace"

    lengths: list[int] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        encoded = tokenizer(
            batch,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )

        lengths.extend(
            len(input_ids)
            for input_ids in encoded["input_ids"]
        )

        if start == 0 or (start // batch_size) % 20 == 0:
            done = min(start + batch_size, len(texts))
            print(f"[tokenize] {done}/{len(texts)}")

    return lengths, "tokenizer"


def prepare_records(
    trajectory_rows: list[dict[str, Any]],
    combined_prefixes: list[dict[str, Any]],
    tokenizer_path: str,
    tokenizer_batch_size: int,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    str,
]:
    trajectories: dict[str, dict[str, Any]] = {}

    for row in trajectory_rows:
        tid = trajectory_id(row)
        sid = sample_id(row)

        if not tid:
            raise ValueError(
                f"Trajectory row missing trajectory_id: keys={sorted(row)}"
            )
        if not sid:
            raise ValueError(
                f"Trajectory row missing sample_id: tid={tid}"
            )

        trajectories[tid] = {
            "trajectory_id": tid,
            "sample_id": sid,
            "gold_answer": extract_gold_answer(row),
            "full_answer": extract_full_answer(row),
            "full_text": extract_full_text(row),
        }

    prefixes_by_tid: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in combined_prefixes:
        prefixes_by_tid[row["trajectory_id"]].append(row)

    missing_prefixes = [
        tid for tid in trajectories
        if tid not in prefixes_by_tid
    ]
    if missing_prefixes:
        raise ValueError(
            f"{len(missing_prefixes)} trajectories have no prefix rows. "
            f"Examples={missing_prefixes[:5]}"
        )

    # Deterministic ordering and inferred progress.
    for tid, rows in prefixes_by_tid.items():
        rows.sort(
            key=lambda row: (
                row["prefix_index"]
                if row["prefix_index"] is not None
                else row["line_idx"]
            )
        )

        total = len(rows)
        for rank, row in enumerate(rows):
            row["rank"] = rank
            if row["progress"] is None:
                row["progress"] = (rank + 1) / max(1, total)

    full_tids = sorted(trajectories)
    prefix_refs: list[tuple[str, int]] = []
    texts: list[str] = []

    for tid in full_tids:
        texts.append(trajectories[tid]["full_text"])

    for tid in full_tids:
        for prefix_idx, row in enumerate(prefixes_by_tid[tid]):
            prefix_refs.append((tid, prefix_idx))
            texts.append(row["prefix_text"])

    lengths, token_mode = token_lengths(
        texts,
        tokenizer_path=tokenizer_path,
        batch_size=tokenizer_batch_size,
    )

    cursor = 0

    for tid in full_tids:
        trajectories[tid]["full_tokens"] = lengths[cursor]
        cursor += 1

    for tid, prefix_idx in prefix_refs:
        prefixes_by_tid[tid][prefix_idx]["prefix_tokens"] = lengths[cursor]
        cursor += 1

    return trajectories, prefixes_by_tid, token_mode


def choose_stop(
    rows: list[dict[str, Any]],
    threshold: float,
    min_progress: float,
    patience: int,
) -> dict[str, Any] | None:
    if patience < 1:
        raise ValueError("patience must be >= 1")

    for end_idx in range(patience - 1, len(rows)):
        window = rows[end_idx - patience + 1:end_idx + 1]

        if any(
            float(row["probability"]) < threshold
            for row in window
        ):
            continue

        if any(
            float(row["progress"]) < min_progress
            for row in window
        ):
            continue

        answers = [
            str(row["explicit_answer"]).strip()
            for row in window
        ]

        if any(not answer for answer in answers):
            continue

        if len(set(answers)) != 1:
            continue

        # With patience > 1, stopping occurs after observing the final
        # element of the stable window.
        return rows[end_idx]

    return None


def evaluate_threshold(
    dataset: str,
    trajectories: dict[str, dict[str, Any]],
    prefixes_by_tid: dict[str, list[dict[str, Any]]],
    threshold: float,
    min_progress: float,
    patience: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trajectory_details = []

    for tid in sorted(trajectories, key=trajectory_rank):
        trajectory = trajectories[tid]
        prefix_rows = prefixes_by_tid[tid]

        stop = choose_stop(
            prefix_rows,
            threshold=threshold,
            min_progress=min_progress,
            patience=patience,
        )

        full_answer = trajectory["full_answer"]
        gold = trajectory["gold_answer"]
        full_tokens = int(trajectory["full_tokens"])

        if stop is None:
            stopped = False
            stopped_answer = full_answer
            stopped_tokens = full_tokens
            stop_progress = 1.0
            stop_probability = None
            stop_prefix_index = None
        else:
            stopped = True
            stopped_answer = stop["explicit_answer"]
            stopped_tokens = min(
                full_tokens,
                int(stop["prefix_tokens"]),
            )
            stop_progress = float(stop["progress"])
            stop_probability = float(stop["probability"])
            stop_prefix_index = stop["prefix_index"]

        full_ok = correctness(full_answer, gold)
        stopped_ok = correctness(stopped_answer, gold)

        trajectory_details.append({
            "dataset": dataset,
            "threshold": threshold,
            "trajectory_id": tid,
            "sample_id": trajectory["sample_id"],
            "gold_answer": gold,
            "full_answer": full_answer,
            "stopped_answer": stopped_answer,
            "full_ok": full_ok,
            "stopped_ok": stopped_ok,
            "stopped": int(stopped),
            "stop_probability": stop_probability,
            "stop_progress": stop_progress,
            "stop_prefix_index": stop_prefix_index,
            "full_output_tokens": full_tokens,
            "stopped_output_tokens": stopped_tokens,
            "saved_output_tokens": max(
                0,
                full_tokens - stopped_tokens,
            ),
        })

    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trajectory_details:
        by_sample[row["sample_id"]].append(row)

    sample_rows = []
    for sid, rows in sorted(by_sample.items()):
        rows.sort(key=lambda row: trajectory_rank(row["trajectory_id"]))

        gold = rows[0]["gold_answer"]

        full_majority = stable_majority([
            row["full_answer"] for row in rows
        ])
        stopped_majority = stable_majority([
            row["stopped_answer"] for row in rows
        ])

        full_majority_ok = correctness(full_majority, gold)
        stopped_majority_ok = correctness(stopped_majority, gold)

        sample_rows.append({
            "sample_id": sid,
            "gold_answer": gold,
            "full_majority_answer": full_majority,
            "stopped_majority_answer": stopped_majority,
            "full_majority_ok": full_majority_ok,
            "stopped_majority_ok": stopped_majority_ok,
            "fixed": int(
                full_majority_ok == 0
                and stopped_majority_ok == 1
            ),
            "broken": int(
                full_majority_ok == 1
                and stopped_majority_ok == 0
            ),
            "changed": int(full_majority != stopped_majority),
        })

    n_trajectories = len(trajectory_details)
    n_samples = len(sample_rows)

    stopped_rows = [
        row for row in trajectory_details
        if row["stopped"]
    ]

    full_total_tokens = sum(
        row["full_output_tokens"]
        for row in trajectory_details
    )
    stopped_total_tokens = sum(
        row["stopped_output_tokens"]
        for row in trajectory_details
    )
    saved_total_tokens = full_total_tokens - stopped_total_tokens

    fixed = sum(row["fixed"] for row in sample_rows)
    broken = sum(row["broken"] for row in sample_rows)

    summary = {
        "dataset": dataset,
        "threshold": threshold,
        "min_progress": min_progress,
        "patience": patience,
        "n_samples": n_samples,
        "n_trajectories": n_trajectories,
        "stopped_trajectories": len(stopped_rows),
        "trajectory_stop_rate": (
            len(stopped_rows) / max(1, n_trajectories)
        ),
        "full_trajectory_acc": mean(
            row["full_ok"]
            for row in trajectory_details
        ),
        "stopped_trajectory_acc": mean(
            row["stopped_ok"]
            for row in trajectory_details
        ),
        "trajectory_acc_delta": (
            mean(row["stopped_ok"] for row in trajectory_details)
            - mean(row["full_ok"] for row in trajectory_details)
        ),
        "full_majority_acc": mean(
            row["full_majority_ok"]
            for row in sample_rows
        ),
        "stopped_majority_acc": mean(
            row["stopped_majority_ok"]
            for row in sample_rows
        ),
        "majority_acc_delta": (
            mean(row["stopped_majority_ok"] for row in sample_rows)
            - mean(row["full_majority_ok"] for row in sample_rows)
        ),
        "fixed_samples": fixed,
        "broken_samples": broken,
        "net_samples": fixed - broken,
        "changed_majority_samples": sum(
            row["changed"] for row in sample_rows
        ),
        "avg_full_output_tokens_per_trajectory": (
            full_total_tokens / max(1, n_trajectories)
        ),
        "avg_stopped_output_tokens_per_trajectory": (
            stopped_total_tokens / max(1, n_trajectories)
        ),
        "avg_saved_output_tokens_per_trajectory": (
            saved_total_tokens / max(1, n_trajectories)
        ),
        "avg_saved_output_tokens_per_question": (
            saved_total_tokens / max(1, n_samples)
        ),
        "output_token_saving_rate": (
            saved_total_tokens / max(1, full_total_tokens)
        ),
        "avg_stop_progress_stopped_only": (
            mean(row["stop_progress"] for row in stopped_rows)
            if stopped_rows
            else None
        ),
        "avg_stop_probability_stopped_only": (
            mean(row["stop_probability"] for row in stopped_rows)
            if stopped_rows
            else None
        ),
    }

    return summary, trajectory_details


def write_outputs(
    out_dir: Path,
    summaries: list[dict[str, Any]],
    details_by_threshold: dict[float, list[dict[str, Any]]],
    token_count_mode: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "token_count_mode": token_count_mode,
        "results": summaries,
    }

    (out_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_fields = list(summaries[0].keys())

    with (out_dir / "summary.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(summaries)

    lines = [
        f"# Offline Trajectory Stopping: {summaries[0]['dataset']}",
        "",
        f"Token count mode: `{token_count_mode}`",
        "",
        "| Tau | Full Maj. | Stop Maj. | Delta | Stop Rate | "
        "Avg. Full Tok. | Avg. Stop Tok. | Saving | Fixed | Broken | Net |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in summaries:
        lines.append(
            f"| {row['threshold']:.2f} "
            f"| {row['full_majority_acc']:.4f} "
            f"| {row['stopped_majority_acc']:.4f} "
            f"| {row['majority_acc_delta']:+.4f} "
            f"| {row['trajectory_stop_rate']:.4f} "
            f"| {row['avg_full_output_tokens_per_trajectory']:.1f} "
            f"| {row['avg_stopped_output_tokens_per_trajectory']:.1f} "
            f"| {row['output_token_saving_rate']:.4f} "
            f"| {row['fixed_samples']} "
            f"| {row['broken_samples']} "
            f"| {row['net_samples']:+d} |"
        )

    (out_dir / "summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    for threshold, details in details_by_threshold.items():
        threshold_name = f"{threshold:.2f}".replace(".", "p")
        detail_path = out_dir / f"details_tau_{threshold_name}.jsonl"

        with detail_path.open("w", encoding="utf-8") as f:
            for row in details:
                f.write(
                    json.dumps(row, ensure_ascii=False) + "\n"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--prefixes", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
    )
    parser.add_argument("--min_progress", type=float, default=0.50)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument(
        "--tokenizer",
        default="/root/autodl-tmp/models/Qwen2.5-7B-Instruct",
    )
    parser.add_argument("--tokenizer_batch_size", type=int, default=256)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    if not 0.0 <= args.min_progress <= 1.0:
        raise ValueError("min_progress must be in [0, 1]")

    for threshold in args.thresholds:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("all thresholds must be in [0, 1]")

    prefix_path = Path(args.prefixes)
    prediction_path = Path(args.predictions)
    trajectory_path = Path(args.trajectories)
    out_dir = Path(args.out_dir)

    print(f"[load] prefixes={prefix_path}")
    prefix_rows = read_jsonl(prefix_path)

    print(f"[load] predictions={prediction_path}")
    prediction_rows = read_jsonl(prediction_path)

    print(f"[load] trajectories={trajectory_path}")
    trajectory_rows = read_jsonl(trajectory_path)

    print(
        "[rows]",
        f"prefixes={len(prefix_rows)}",
        f"predictions={len(prediction_rows)}",
        f"trajectories={len(trajectory_rows)}",
    )

    combined = combine_prefix_predictions(
        prefix_rows,
        prediction_rows,
    )

    trajectories, prefixes_by_tid, token_mode = prepare_records(
        trajectory_rows,
        combined,
        tokenizer_path=args.tokenizer,
        tokenizer_batch_size=args.tokenizer_batch_size,
    )

    summaries = []
    details_by_threshold = {}

    for threshold in sorted(set(args.thresholds)):
        print()
        print("=" * 80)
        print(
            f"[evaluate] dataset={args.dataset} "
            f"tau={threshold:.2f} "
            f"min_progress={args.min_progress:.2f} "
            f"patience={args.patience}"
        )
        print("=" * 80)

        summary, details = evaluate_threshold(
            dataset=args.dataset,
            trajectories=trajectories,
            prefixes_by_tid=prefixes_by_tid,
            threshold=threshold,
            min_progress=args.min_progress,
            patience=args.patience,
        )

        summaries.append(summary)
        details_by_threshold[threshold] = details

        print(json.dumps(summary, ensure_ascii=False, indent=2))

    write_outputs(
        out_dir,
        summaries,
        details_by_threshold,
        token_count_mode=token_mode,
    )

    print()
    print(f"[DONE] {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
